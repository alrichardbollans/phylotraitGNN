from typing import Optional, Callable

import networkx as nx
import numpy as np
import pandas as pd
import torch
from Bio import Phylo
from torch_geometric.data import Data, Dataset
from torch_geometric.transforms import ToUndirected, FeaturePropagation, AddSelfLoops
from torch_geometric.utils import from_networkx


class GenericPhyloDataset(Dataset):
    def __init__(self, feature_csv_path_with_missing_target: str,
                 target_name: str,
                 binary_or_continuous: str,
                 ground_truth_csv_path: str = None,
                 sigma: float = None, transform: Optional[Callable] = None):

        self.binary_or_continuous = binary_or_continuous
        self.target_name = target_name

        # Read the feature CSV file. It holds
        self.feature_csv_path_with_missing_target = feature_csv_path_with_missing_target
        self.feature_with_missing_target_df = pd.read_csv(feature_csv_path_with_missing_target, index_col=0)
        self.ground_truth_csv_path = ground_truth_csv_path
        if ground_truth_csv_path is not None:
            self.ground_truth_df = pd.read_csv(ground_truth_csv_path, index_col=0)
        else:
            self.ground_truth_df = None

        self.sigma = sigma
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
            sigma = original_edge_std

        # Following Xiaojin Zhu and Zoubin Ghahramani, Learning from Labeled and Unlabeled Data with Label Propagation (Carnegie Mellon University, Pittsburgh, 2002).
        # Set edge weights to e^(-d^2/sigma^2)
        edge_weight = torch.exp(-(edge_length ** 2) / (sigma ** 2))

        return edge_weight, original_edge_std

    @staticmethod
    def transform_data(data, edge_length, sigma: float = None, add_self_loops=False):
        # From Emanuele Rossi et al., ‘On the Unreasonable Effectiveness of Feature Propagation in Learning on Graphs with Missing Node Features’,
        # arXiv:2111.12128, preprint, arXiv, 23 May 2022, https://doi.org/10.48550/arXiv.2111.12128.
        # https://pytorch-geometric.readthedocs.io/en/stable/generated/torch_geometric.transforms.FeaturePropagation.html#torch_geometric.transforms.FeaturePropagation
        # if data.x.shape[1] == 0:
        #     data.x = torch.ones((data.num_nodes, 1))
        # else:
        missing_mask = torch.isnan(data.x)
        FeaturePropagation_transform = FeaturePropagation(missing_mask=missing_mask)
        if torch.any(missing_mask):
            data = FeaturePropagation_transform(data)

        # Feature propagation transform breaks with edge attributes, so add them back in before the undirected transform.
        edge_weight, original_edge_std = GenericPhyloDataset.get_edge_weights(edge_length, sigma)
        data.original_edge_std = original_edge_std
        data.edge_weight = edge_weight
        data.edge_length = edge_length

        if add_self_loops:
            data = AddSelfLoops(fill_value=1)(data)
            raise NotImplementedError("Need to check data.values are preserved correctly after adding self loops.")

        ToUndirected_transform = ToUndirected(reduce='mean')
        data = ToUndirected_transform(data)

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

        # print(data.train_mask.sum(), data.test_mask.sum())
        assert (self.data.train_mask & self.data.test_mask).sum() == 0  # should be 0

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
        # try:
        #     assert len(feature_names) > 0
        # except AssertionError:
        #     raise ValueError("Input data tables must have at least one feature (non target) column.")

        if self.ground_truth_df is not None:
            pd.testing.assert_frame_equal(X_feature_df, self.ground_truth_df[feature_names])

            y_df = self.ground_truth_df[[self.target_name]]
        else:
            y_df = self.feature_with_missing_target_df[[self.target_name]]

        mode_training_value = y_with_missing_target_df[self.target_name].dropna().mode()[0]
        node_target_df = pd.DataFrame()
        if hasattr(self, 'nodes_that_arent_tips'):
            ## When there are nodes that arent tips, these have no feature values.
            ## For features, give them a nan value. These will be filled in later by feature propagation.

            rows, cols = X_feature_df.shape
            NAN_node_features = pd.DataFrame([[np.nan] * cols] * (len(self.nodes_that_arent_tips)), index=self.nodes_that_arent_tips)
            NAN_node_features.index.name = 'accepted_species'
            NAN_node_features.columns = feature_names
            X_feature_df = pd.concat([X_feature_df, NAN_node_features], axis=0)

            ## For target, Give these the mode training feature value.
            ## The actual value shouldn't really matter as masks are used to exclude these nodes from training and testing.
            ## and also include in both masks as False.? Or in train mask?

            node_target_df = pd.DataFrame([[mode_training_value]] * (len(self.nodes_that_arent_tips)), index=self.nodes_that_arent_tips,
                                          columns=[self.target_name])
            node_target_df.index.name = 'accepted_species'
            y_df = pd.concat([y_df, node_target_df], axis=0)

            y_with_missing_target_df = pd.concat([y_with_missing_target_df, node_target_df], axis=0)

        # Make sure all nodes are present in both feature and target dataframes, and  dataframes have same order
        # Both X and y are sorted by self.node_names
        X = X_feature_df.loc[self.node_names].values  # Align with node names

        if self.ground_truth_df is None:
            y_df[self.target_name] = y_df[self.target_name].fillna(mode_training_value)
        y = y_df.loc[self.node_names][self.target_name].values  # Align with node names

        y_with_missing_target_df = y_with_missing_target_df.loc[self.node_names]  # Align with node names

        train_mask = torch.tensor(
            np.where((y_with_missing_target_df[self.target_name].isna() | y_with_missing_target_df.index.isin(node_target_df.index)),
                     False, True), dtype=torch.bool)
        test_mask = torch.tensor(
            np.where((~y_with_missing_target_df[self.target_name].isna() | y_with_missing_target_df.index.isin(node_target_df.index)),
                     False, True), dtype=torch.bool)
        if self.binary_or_continuous == 'continuous':
            y_dtype = torch.float
        elif self.binary_or_continuous == 'binary':
            y_dtype = torch.int64

        else:
            raise ValueError(f"binary_or_continuous must be 'continuous' or 'binary', not {self.binary_or_continuous}")

        return X, y, train_mask, test_mask, y_dtype

    def len(self):
        return 1  # Single graph dataset

    def get(self, idx):
        return self.data


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
                 sigma: float = None, ):
        """
        Args:
            tree_distance_csv_path: Path to CSV file with distance matrix. This is created in R with tree_distances = ape::cophenetic.phylo(out_tree) and written to a file with  write.csv.
            feature_csv_path_with_missing_target: Path to CSV file with node features. Columns should be feature names and rows should be node names. The target column should be missing for test nodes.
            ground_truth_csv_path: Path to CSV file with node features. Columns should be feature names and rows should be node names. Should include all values for training and test nodes.
            target_name: Name of target column in feature CSV file.
            binary_or_continuous: Target is binary or continuous.
            threshold: Distance threshold for creating edges (edges for distances <= threshold)
            k_nearest: Create edges to k-nearest neighbors per node
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

        super().__init__(feature_csv_path_with_missing_target,
                         target_name,
                         binary_or_continuous,
                         ground_truth_csv_path=ground_truth_csv_path,
                         sigma=sigma)

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
            edge_index = torch.tensor([rows, cols], dtype=torch.long)
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
            edge_index = torch.tensor([rows_bidir, cols_bidir], dtype=torch.long)
            edge_attr = torch.tensor(
                np.concatenate([self.dist_matrix[rows, cols],
                                self.dist_matrix[rows, cols]]),
                dtype=torch.float
            ).view(-1, 1)

        X, y, train_mask, test_mask, y_dtype = self.get_features_and_masks()

        data = Data(
            x=torch.tensor(X, dtype=torch.float),
            y=torch.tensor(y, dtype=y_dtype),
            train_mask=train_mask,
            test_mask=test_mask,
            edge_index=edge_index,
            # edge_attr=edge_attr # Feature propagation transform breaks with edge attributes, so add them back in before the undirected transform.
        )
        data = self.transform_data(data, edge_length=edge_attr, sigma=self.sigma)

        return data


class NewickDataset(GenericPhyloDataset):
    """Dataset from Newick tree file i.e. where all nodes from the original tree are present."""

    def __init__(self,
                 newick_tree_path: str,
                 feature_csv_path_with_missing_target: str,
                 target_name: str,
                 binary_or_continuous: str,
                 ground_truth_csv_path: str = None,
                 sigma: float = None):
        """
        Args:
            newick_tree_path: Path to CSV file with newick tree.
            feature_csv_path_with_missing_target: Path to CSV file with node features. Columns should be feature names and rows should be node names. The target column should be missing for test nodes.
            ground_truth_csv_path: Path to CSV file with node features. Columns should be feature names and rows should be node names. Should include all values for training and test nodes.
            target_name: Name of target column in feature CSV file.
            binary_or_continuous: Target is binary or continuous.
            transform: PyTorch Geometric transforms which transform values stored in 'x'
        """
        self.newick_tree_path = newick_tree_path

        # Read the tree CSV file
        self.networkx_tree = self.newick_to_networkx(newick_tree_path)
        # Nodes should all be named 'Node_x' where x is the node number
        # Can set this in R with `paste("Node",1L:tree$Nnode, sep='_') -> tree$node.label`
        self.node_names = list(self.networkx_tree.nodes)
        self.nodes_that_arent_tips = [n for n in self.node_names if 'Node_' in n]
        if len(self.nodes_that_arent_tips) == 0:
            raise ValueError("No nodes found in tree that aren't tips. Nodes should be named 'Node_x' where x is the node number.")

        super().__init__(feature_csv_path_with_missing_target,
                         target_name,
                         binary_or_continuous,
                         ground_truth_csv_path=ground_truth_csv_path,
                         sigma=sigma)

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
        edge_attr = torch.tensor([self.networkx_tree[u][v]['length'] for u, v in self.networkx_tree.edges()]).view(-1, 1)
        pyg_data = from_networkx(self.networkx_tree)
        X, y, train_mask, test_mask, y_dtype = self.get_features_and_masks()

        data = Data(
            x=torch.tensor(X, dtype=torch.float),
            y=torch.tensor(y, dtype=y_dtype),
            train_mask=train_mask,
            test_mask=test_mask,
            edge_index=pyg_data.edge_index,
            # edge_attr=edge_attr # Feature propagation transform breaks with edge attributes, so add them back in before the undirected transform.
        )
        data = self.transform_data(data, edge_length=edge_attr, sigma=self.sigma)

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
