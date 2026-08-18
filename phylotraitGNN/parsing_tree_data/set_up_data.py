from typing import Optional, Callable

import networkx as nx
import numpy as np
import pandas as pd
import torch
from Bio import Phylo
from torch_geometric.data import Data, Dataset
from torch_geometric.transforms import ToUndirected, FeaturePropagation, AddSelfLoops
from torch_geometric.utils import from_networkx
from sklearn.impute import SimpleImputer


class GenericPhyloDataset(Dataset):
    self_loop_fill_value = 1

    def __init__(self, feature_csv_path_with_missing_target: str,
                 target_name: str,
                 binary_or_continuous: str,
                 ground_truth_csv_path: str = None,
                 sigma: float = None, transform: Optional[Callable] = None, add_self_loops: bool = False, validation_nodes: Optional[list] = None):
        """
        Initializes an instance that manages a dataset with potential missing target values and ground truth data
        (optional). It supports handling both binary or continuous target variables. Additional functionalities
        such as data transformation, adding self-loops, and handling validation nodes are configurable through
        provided parameters.

        Attributes:
            binary_or_continuous (str): A string indicating whether the target variable is "binary" or "continuous".
                This dictates the type of analysis that can be performed.
            target_name (str): The name of the target variable in the dataset.
            feature_csv_path_with_missing_target (str): Path to the CSV file containing features along with a target
                variable that may have missing values.
            feature_with_missing_target_df (pd.DataFrame): A DataFrame holding the feature and target values from the
                CSV file provided via 'feature_csv_path_with_missing_target'. The target values may have missing entries.
            ground_truth_csv_path (str, optional): Path to the optional CSV file containing complete ground truth data
                of the target variable for validation purposes.
            ground_truth_df (pd.DataFrame or None): A DataFrame holding the ground truth target data if
                'ground_truth_csv_path' is provided, or None if it is not supplied.
            sigma (float, optional): A float value used for regularization or smoothing in algorithms where applicable.
            add_self_loops (bool): A boolean flag indicating whether self-loops should be added in graph-based analysis.
            validation_nodes (list, optional): A list of nodes (or indices) designated for validating the model's
                predictions or procedures.
        """
        self.binary_or_continuous = binary_or_continuous

        if binary_or_continuous not in ['binary', 'continuous']:
            raise ValueError(f"binary_or_continuous must be 'binary' or 'continuous', not {binary_or_continuous}")

        self.target_name = target_name

        # Read the feature CSV file. It holds
        self.feature_csv_path_with_missing_target = feature_csv_path_with_missing_target
        self.feature_with_missing_target_df = pd.read_csv(feature_csv_path_with_missing_target, index_col=0)
        self.index_name = self.feature_with_missing_target_df.index.name
        self.ground_truth_csv_path = ground_truth_csv_path
        if ground_truth_csv_path is not None:
            self.ground_truth_df = pd.read_csv(ground_truth_csv_path, index_col=0)
        else:
            self.ground_truth_df = None

        self.sigma = sigma
        self.add_self_loops = add_self_loops
        self.self_loop_fill_value = 1
        self.validation_nodes = validation_nodes

        if self.validation_nodes is not None:
            for v in self.validation_nodes:
                if v not in self.node_names:
                    raise ValueError(f"Validation node {v} not found in tree.")

        super().__init__(transform=transform)

    @staticmethod
    def get_edge_weights(edge_length, sigma: float = None):
        original_edge_std = edge_length.std()
        if sigma is None:
            # Setting sigma will mean the generated edge weights are the same as if the lengths hadn't been scaled.
            # Large values of sigma essentially shrinks the graph, so all unlabelled points are predicted to be the same.
            # Small values of sigma mean the nearest points dominate.
            # Following Zhu, 2002
            # d_zero is the shortest distance between two nodes with different labels.
            # This can give tiny values of sigma, which destroy weights
            # differing_lengths = []
            # for i in range(data.edge_index.shape[1]):
            #     src, dst = data.edge_index[0, i], data.edge_index[1, i]
            #     if data.y[src] != data.y[dst]:
            #         differing_lengths.append(edge_length_scaled[i])
            #
            # if differing_lengths:
            #     d_zero = min(differing_lengths)
            # else:
            #     d_zero = edge_length_scaled.mean()  # or fallback to some default, e.g., edge_lengths.mean()

            # Following Dengyong Zhou et al., Learning with Local and Global Consistency
            # sigma appears to just be std, which gives more reasonable weights
            if original_edge_std == 0:
                sigma = 1
            else:
                sigma = original_edge_std

        # Following Xiaojin Zhu and Zoubin Ghahramani, Learning from Labeled and Unlabeled Data with Label Propagation (Carnegie Mellon University, Pittsburgh, 2002).
        # Set edge weights to e^(-d^2/sigma^2)
        edge_weight = torch.exp(-(edge_length ** 2) / (sigma ** 2))
        assert (torch.all(edge_weight >= 0.0) and torch.all(edge_weight <= 1.0))

        return edge_weight, original_edge_std

    @staticmethod
    def transform_data(data, add_self_loops=False):
        # I found that the pytorch_geometric implementation of FeaturePropagation would break when edge weights included.
        # Without edge weights, it runs but seems to output mean feature values. Instead, just use simple imputer here.
        missing_mask = torch.isnan(data.x)
        if torch.any(missing_mask):
            # Set up a SimpleImputer to impute missing values in data.x
            x_np = data.x.cpu().numpy()  # Convert tensor to numpy array
            imputer = SimpleImputer()
            x_imputed = imputer.fit_transform(x_np)
            data.x = torch.from_numpy(x_imputed).type_as(data.x)  # Convert back to tensor

        if add_self_loops:
            assert data.edge_weight.max() <= GenericPhyloDataset.self_loop_fill_value
            data = AddSelfLoops(fill_value=GenericPhyloDataset.self_loop_fill_value)(data)

        data = ToUndirected(reduce='mean')(data)

        return data

    def checks(self):
        ## Do some checks
        self.data.validate(raise_on_error=True)

        assert self.data.is_undirected()

        for node in self.feature_with_missing_target_df.index:
            assert node in self.node_names, f"Node {node} not found in tree."
        if self.ground_truth_df is not None:
            pd.testing.assert_frame_equal(self.ground_truth_df.drop(columns=[self.target_name]),
                                          self.feature_with_missing_target_df.drop(columns=[self.target_name]))
            pd.testing.assert_frame_equal(self.ground_truth_df[~self.feature_with_missing_target_df[self.target_name].isna()],
                                          self.feature_with_missing_target_df.dropna(subset=[self.target_name]), check_dtype=False)
        if self.binary_or_continuous == 'binary':
            assert self.num_classes <= 2
        else:
            # check that data.y are floats
            assert self.data.y.dtype == torch.float
        # print(data.train_mask.sum(), data.test_mask.sum())
        assert (
                   self.data.train_mask).sum() > 0, "No training nodes found. feature_csv_path_with_missing_target is expected to have missing values for test nodes, and have some training nodes with values."
        assert (
                   self.data.test_mask).sum() > 0, "No test nodes found. feature_csv_path_with_missing_target is expected to have missing values for test nodes."
        assert (self.data.train_mask & self.data.test_mask).sum() == 0  # should be 0
        if hasattr(self.data, 'val_mask'):
            assert (self.data.val_mask & self.data.test_mask).sum() == 0  # should be 0
            assert (self.data.train_mask & self.data.val_mask).sum() == 0  # should be 0

        # No train or test nodes should have missing target values
        # Also check for the placeholder nan value
        nan_value = torch.tensor(np.array([np.nan]), dtype=torch.int64).numpy()[0]

        test_y = self.data.y[self.data.test_mask]
        if self.ground_truth_csv_path is not None:
            assert ~torch.any(torch.isnan(test_y))
            assert ~torch.any(test_y == nan_value)

        train_y = self.data.y[self.data.train_mask]
        assert ~torch.any(torch.isnan(train_y))
        assert ~torch.any(train_y == nan_value)

    def get_features_and_masks(self):

        # Create train/val/test masks from missing target values
        y_with_missing_target_df = self.feature_with_missing_target_df[[self.target_name]]

        X_feature_df = self.feature_with_missing_target_df.drop(columns=[self.target_name])

        feature_names = X_feature_df.columns

        if self.ground_truth_df is not None:
            pd.testing.assert_frame_equal(X_feature_df, self.ground_truth_df[feature_names], check_dtype=False)

            y_df = self.ground_truth_df[[self.target_name]]
        else:
            y_df = self.feature_with_missing_target_df[[self.target_name]]
        if self.binary_or_continuous == 'binary':
            avg_training_value = y_with_missing_target_df[self.target_name].dropna().mode()[0]
        elif self.binary_or_continuous == 'continuous':
            avg_training_value = y_with_missing_target_df[self.target_name].dropna().mean()
        nan_target_df = pd.DataFrame()
        nodes_without_feature_values = [name for name in self.node_names if name not in X_feature_df.index]
        if len(nodes_without_feature_values) > 0:
            ## When there are nodes that arent tips, these have no feature values.
            ## For features, give them a nan value. These will be filled in later by feature propagation.

            rows, cols = X_feature_df.shape
            NAN_features = pd.DataFrame([[np.nan] * cols] * (len(nodes_without_feature_values)), index=nodes_without_feature_values)
            NAN_features.index.name = self.index_name
            NAN_features.columns = feature_names
            X_feature_df = pd.concat([X_feature_df, NAN_features], axis=0)

            ## For target, Give these the avg training feature value.
            nan_target_df = pd.DataFrame([[avg_training_value]] * (len(nodes_without_feature_values)), index=nodes_without_feature_values,
                                         columns=[self.target_name])
            nan_target_df.index.name = self.index_name
            y_df = pd.concat([y_df, nan_target_df], axis=0)

            y_with_missing_target_df = pd.concat([y_with_missing_target_df, nan_target_df], axis=0)

        # Make sure all nodes are present in both feature and target dataframes, and  dataframes have same order
        # Both X and y are sorted by self.node_names which provides the order for thre resulting predictions.
        X = X_feature_df.loc[self.node_names].values  # Align with node names

        if self.ground_truth_df is None:
            y_df.loc[:, self.target_name] = y_df[self.target_name].fillna(avg_training_value)
        y = y_df.loc[self.node_names][self.target_name].values  # Align with node names

        y_with_missing_target_df = y_with_missing_target_df.loc[self.node_names]  # Align with node names

        if self.validation_nodes is not None:
            temp_val_mask = torch.tensor(
                np.where((y_with_missing_target_df.index.isin(self.validation_nodes)),
                         True, False), dtype=torch.bool)
            val_mask = temp_val_mask
        else:
            # Make val mask a torch.tensor of False which is the same length as X and y
            temp_val_mask = torch.zeros(len(X), dtype=torch.bool)
            val_mask = None

        train_mask = torch.tensor(
            np.where((y_with_missing_target_df[self.target_name].isna() | y_with_missing_target_df.index.isin(
                nan_target_df.index) | temp_val_mask.cpu().numpy()),
                     False, True), dtype=torch.bool)

        test_mask = torch.tensor(
            np.where((~y_with_missing_target_df[self.target_name].isna() | y_with_missing_target_df.index.isin(
                nan_target_df.index) | temp_val_mask.cpu().numpy()),
                     False, True), dtype=torch.bool)
        if self.binary_or_continuous == 'continuous':
            y_dtype = torch.float
        elif self.binary_or_continuous == 'binary':
            y_dtype = torch.int64

        else:
            raise ValueError(f"binary_or_continuous must be 'continuous' or 'binary', not {self.binary_or_continuous}")

        return X, y, train_mask, val_mask, test_mask, y_dtype

    def len(self):
        return 1  # Single graph dataset

    def get(self, idx):
        return self.data

    def get_model_prediction_outputs_in_feature_order(self, predictions) -> pd.DataFrame:
        # Predictions are output in order of self.node_names, which contains nodes not in feature data and not in same order
        # This function outputs a dataframe with the same order as the feature dataframe.
        prediction_df = pd.DataFrame(predictions.detach().cpu().numpy(), index=pd.Series(self.node_names, name=self.index_name))
        out_df = prediction_df.loc[self.feature_with_missing_target_df.index]
        return out_df


class DistanceMatrixDataset(GenericPhyloDataset):
    """Dataset from distance matrix CSV file i.e. where all tips are connected to each other by a distance matrix."""

    def __init__(self,
                 tree_distance_csv_path: str,
                 feature_csv_path_with_missing_target: str,
                 target_name: str,
                 binary_or_continuous: str,
                 threshold: Optional[float] = None,
                 k_nearest: Optional[int] = None,
                 ground_truth_csv_path: str = None,
                 sigma: float = None, add_self_loops: bool = False, validation_nodes: Optional[list] = None):
        """
        Initializes the instance by setting attributes and reading required CSV files, while validating the
        combination of parameters. Implements the base class initialization and processes required data.

        Attributes:
            tree_distance_csv_path (str): The path to the CSV file containing tree distances.
            feature_csv_path_with_missing_target (str): The path to the feature CSV file with possible missing target values.
            target_name (str): The name of the target feature.
            binary_or_continuous (str): Specifies whether the target feature is binary or continuous.
            threshold (Optional[float]): The threshold value for pruning or filtering, mutually exclusive with k_nearest.
            k_nearest (Optional[int]): The number of nearest nodes to consider, mutually exclusive with threshold.
            ground_truth_csv_path (Optional[str]): The path to the ground truth CSV file, if applicable.
            sigma (Optional[float]): A float value used for specific computations. Its default value is None.
            add_self_loops (bool): Indicates whether to add self-loops to the distance matrix. Default is False.
            validation_nodes (Optional[list]): A list of nodes that represent validation data, if provided.

        Raises:
            NotImplementedError: If both threshold and k_nearest are specified simultaneously.
        """
        self.tree_distance_csv_path = tree_distance_csv_path
        self.threshold = threshold
        self.k_nearest = k_nearest

        if threshold is not None and k_nearest is not None:
            raise NotImplementedError("Only one of threshold or k_nearest can be set.")

        # Read the tree CSV file
        self.tree_distance_df = pd.read_csv(tree_distance_csv_path, index_col=0)
        self.node_names = self.tree_distance_df.index.tolist()
        self.dist_matrix = self.tree_distance_df.values

        if add_self_loops:
            print("Might be unusual to add self loops to distance matrix dataset, but this is possible.")
        super().__init__(feature_csv_path_with_missing_target,
                         target_name,
                         binary_or_continuous,
                         ground_truth_csv_path=ground_truth_csv_path,
                         sigma=sigma, add_self_loops=add_self_loops, validation_nodes=validation_nodes)

        # Load the data
        self.data = self._process()
        self.checks()

    def _process(self):
        """Convert distance matrix to PyG Data object."""
        num_nodes = len(self.node_names)

        # Create edges from distance matrix
        if self.threshold is not None:
            # Create edges for distances below threshold
            rows, cols = np.where(self.dist_matrix <= self.threshold)
            edge_index = torch.tensor(np.array([rows, cols]), dtype=torch.long)
            edge_attr = torch.tensor(self.dist_matrix[rows, cols], dtype=torch.float).view(-1, 1)

        elif self.k_nearest is not None:
            # Create edges to k-nearest neighbors
            edge_list = []
            edge_lengths = []

            for i in range(num_nodes):
                # Get indices of k+1 smallest distances (includes self)
                k = min(self.k_nearest + 1, num_nodes)
                nearest = np.argpartition(self.dist_matrix[i], k)[:k]
                # Remove self if present
                nearest = nearest[nearest != i]
                # Take exactly k_nearest
                nearest = nearest[:self.k_nearest]

                for j in nearest:
                    edge_list.append([i, j])
                    edge_lengths.append(self.dist_matrix[i, j])

            edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
            edge_attr = torch.tensor(edge_lengths, dtype=torch.float).view(-1, 1)

        else:
            # Fully connected graph (all pairs except self-loops)
            rows, cols = np.triu_indices(num_nodes, k=1)
            # Add both directions for undirected graph
            rows_bidir = np.concatenate([rows, cols])
            cols_bidir = np.concatenate([cols, rows])
            edge_index = torch.tensor(np.array([rows_bidir, cols_bidir]), dtype=torch.long)
            edge_attr = torch.tensor(
                np.concatenate([self.dist_matrix[rows, cols],
                                self.dist_matrix[rows, cols]]),
                dtype=torch.float
            ).view(-1, 1)

        X, y, train_mask, val_mask, test_mask, y_dtype = self.get_features_and_masks()
        edge_weight, original_edge_std = GenericPhyloDataset.get_edge_weights(edge_attr, self.sigma)

        data = Data(
            x=torch.tensor(X, dtype=torch.float),
            y=torch.tensor(y, dtype=y_dtype),
            train_mask=train_mask,
            val_mask=val_mask,
            test_mask=test_mask,
            edge_index=edge_index,
            edge_weight=edge_weight,
            original_edge_std=original_edge_std,
        )
        data = self.transform_data(data, add_self_loops=self.add_self_loops)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        data.to(device)

        return data


class NewickDataset(GenericPhyloDataset):
    """Dataset from Newick tree file i.e. where all nodes from the original tree are present."""

    def __init__(self,
                 newick_tree_path: str,
                 feature_csv_path_with_missing_target: str,
                 target_name: str,
                 binary_or_continuous: str,
                 ground_truth_csv_path: str = None,
                 sigma: float = None, add_self_loops: bool = False, validation_nodes: Optional[list] = None):
        """
        Initializes an instance and validates the input data and tree structure.

        Attributes:
            newick_tree_path (str): File path to the Newick tree representation.
            networkx_tree: A networkx tree parsed from the Newick file.
            node_names (list): List of all node names in the tree.
            data: Processed data after initialization and validation.

        Parameters:
            newick_tree_path (str): Path to the Newick file containing the tree structure.
            feature_csv_path_with_missing_target (str): Path to the feature data CSV file
                with missing target information.
            target_name (str): Name of the target variable to focus on or predict.
            binary_or_continuous (str): Specifies whether the target variable is binary
                or continuous.
            ground_truth_csv_path (str | None, optional): Path to a CSV file containing
                ground truth data for validation, if any. Defaults to None.
            sigma (float | None, optional): A parameter for any required calculations
                or operations, if applicable. Defaults to None.
            add_self_loops (bool, optional): Whether to introduce self-loops in the
                network graph. Defaults to False.
            validation_nodes (list | None, optional): A list of nodes reserved for
                validation purposes. Defaults to None.

        """
        self.newick_tree_path = newick_tree_path

        # Read the tree file
        self.networkx_tree = self.newick_to_networkx(newick_tree_path)
        self.node_names = list(self.networkx_tree.nodes)

        super().__init__(feature_csv_path_with_missing_target,
                         target_name,
                         binary_or_continuous,
                         ground_truth_csv_path=ground_truth_csv_path,
                         sigma=sigma, add_self_loops=add_self_loops, validation_nodes=validation_nodes)

        # Load the data
        self.data = self._process()
        self.checks()

    @staticmethod
    def newick_to_networkx(tree_file):
        tree = Phylo.read(tree_file, 'newick')
        G = nx.DiGraph()  # or nx.Graph() for undirected

        for clade in tree.find_clades():
            if clade.name:
                G.add_node(clade.name, branch_length=clade.branch_length)

            for child in clade.clades:
                parent_name = clade.name
                child_name = child.name
                G.add_edge(parent_name, child_name,
                           length=child.branch_length)

        return G

    def _process(self):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        edge_length = torch.tensor([self.networkx_tree[u][v]['length'] for u, v in self.networkx_tree.edges()]).view(-1, 1)
        edge_weight, original_edge_std = GenericPhyloDataset.get_edge_weights(edge_length, self.sigma)
        pyg_data = from_networkx(self.networkx_tree)
        X, y, train_mask, val_mask, test_mask, y_dtype = self.get_features_and_masks()

        data = Data(
            x=torch.tensor(X, dtype=torch.float),
            y=torch.tensor(y, dtype=y_dtype),
            train_mask=train_mask,
            val_mask=val_mask,
            test_mask=test_mask,
            edge_index=pyg_data.edge_index,
            edge_weight=edge_weight,
            original_edge_std=original_edge_std,
        )

        data = self.transform_data(data, add_self_loops=self.add_self_loops)
        data.to(device)
        return data


def main():
    # dataset = Planetoid(root='data/Planetoid', name='Cora')
    # planetoid_data = dataset[0]
    # Load the dataset
    # dataset1 = DistanceMatrixDataset(
    #     tree_distance_csv_path='unittest_data/binary/tree_distances.csv',
    #     feature_csv_path_with_missing_target='unittest_data/binary/mcar_values.csv',
    #     ground_truth_csv_path='unittest_data/binary/ground_truth.csv',
    #     target_name='trait_BM_trend_scaled',
    #     binary_or_continuous='binary',
    #     k_nearest=50,  # Alternative: connect to 2 nearest neighbors
    #     transform=torch_geometric.transforms.NormalizeFeatures()
    #
    # )

    dataset1 = NewickDataset(
        newick_tree_path='unittest_data/binary/tree.tre',
        feature_csv_path_with_missing_target='unittest_data/binary/mcar_values.csv',
        ground_truth_csv_path='unittest_data/binary/ground_truth.csv',
        target_name='trait_BM_trend_scaled',
        binary_or_continuous='binary',

    )

    print(f'Number of features: {dataset1.num_features}')
    print(f'Number of classes: {dataset1.num_classes}')
    dataset1.print_summary()
    # g = torch_geometric.utils.to_networkx(data1, to_undirected=True)
    # nx.draw(g)
    # plt.show()
    data1 = dataset1[0]
    for data in [data1]:
        print(f"\nDataset Info:")
        print(f"Number of nodes: {len(dataset1.node_names)}")
        print(f"Number of edges: {data.edge_index.shape[1]}")
        print(f"Node feature shape: {data.x.shape}")
        print(f"Edge attribute shape: {data.edge_weight.shape}")
        print(f'Number of training nodes: {int(data.train_mask.sum())}')
        print(f'Number of testing nodes: {int(data.test_mask.sum())}')

        # You can now use this like any PyG dataset
        print(f"\nEdge indices (first 5): {data.edge_index[:, :5]}")
        print(f"Edge attributes (first 5): {data.edge_weight[:5].flatten()}")


if __name__ == '__main__':
    main()
