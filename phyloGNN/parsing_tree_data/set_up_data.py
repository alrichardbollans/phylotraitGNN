import networkx as nx
import pandas as pd
import torch_geometric
from matplotlib import pyplot as plt
from torch_geometric.datasets import Planetoid
from torch_geometric.explain import Explanation
from torch_geometric.utils import to_edge_index
import pandas as pd
import numpy as np
import torch
from torch_geometric.data import Data, Dataset
import os
from typing import Optional, Callable
import scipy.sparse as sp
from torch_geometric.utils import from_networkx
from Bio import Phylo


class GenericPhyloDataset(Dataset):
    def __init__(self, transform: Optional[Callable] = None):
        super().__init__(transform=transform)

    def checks(self):
        ## Do some checks
        for node in self.feature_with_missing_target_df.index:
            assert node in self.node_names, f"Node {node} not found in tree."
        pd.testing.assert_frame_equal(self.ground_truth_df.drop(columns=[self.target_name]),
                                      self.feature_with_missing_target_df.drop(columns=[self.target_name]))
        pd.testing.assert_frame_equal(self.ground_truth_df[~self.feature_with_missing_target_df[self.target_name].isna()],
                                      self.feature_with_missing_target_df.dropna(subset=[self.target_name]), check_dtype=False)
        if self.binary_or_continuous == 'binary':
            assert self.num_classes == 2

        # print(data.train_mask.sum(), data.test_mask.sum())
        assert (self.data.train_mask & self.data.test_mask).sum()==0  # should be 0

    def get_features_and_masks(self):
        # Create train/val/test masks from missing target values
        y_with_missing_target_df = self.feature_with_missing_target_df[[self.target_name]]

        X_feature_df = self.ground_truth_df.drop(columns=[self.target_name])

        feature_names = X_feature_df.columns
        try:
            assert len(feature_names) > 0
        except AssertionError:
            raise ValueError("Input data tables must have at least one feature (non target) column.")

        pd.testing.assert_frame_equal(X_feature_df, self.ground_truth_df[feature_names])

        y_df = self.ground_truth_df[[self.target_name]]

        if hasattr(self, 'nodes_that_arent_tips'):
            ## When there are nodes that arent tips, these have no feature values and so need giving np.nan and also including in both masks as False.
            rows, cols = X_feature_df.shape
            NAN_node_features = pd.DataFrame([[np.nan] * cols] * (len(self.nodes_that_arent_tips)), index=self.nodes_that_arent_tips)
            NAN_node_features.index.name = 'accepted_species'
            NAN_node_features.columns = feature_names
            X_feature_df = pd.concat([X_feature_df, NAN_node_features], axis=0)

            NAN_node_target = pd.DataFrame([[np.nan]] * (len(self.nodes_that_arent_tips)), index=self.nodes_that_arent_tips,
                                           columns=[self.target_name])
            NAN_node_target.index.name = 'accepted_species'
            y_df = pd.concat([y_df, NAN_node_target], axis=0)

            y_with_missing_target_df = pd.concat([y_with_missing_target_df, NAN_node_target], axis=0)

        # Make sure all nodes are present in both feature and target dataframes, and  dataframes have same order
        X = X_feature_df.loc[self.node_names].values  # Align with node names

        y = y_df.loc[self.node_names][self.target_name].values  # Align with node names

        y_with_missing_target_df = y_with_missing_target_df.loc[self.node_names]  # Align with node names
        train_mask = torch.tensor(np.where(y_with_missing_target_df[self.target_name].isna(), False, True), dtype=torch.bool)

        if hasattr(self, 'nodes_that_arent_tips'):
            test_mask = np.where((y_with_missing_target_df[self.target_name].isna() & ~y_with_missing_target_df.index.isin(NAN_node_target.index)),
                                 True, False)
        else:
            test_mask = torch.tensor(np.invert(train_mask), dtype=torch.bool)
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
                 ground_truth_csv_path: str,
                 target_name: str,
                 binary_or_continuous: str,
                 threshold: Optional[float] = None,
                 k_nearest: Optional[int] = None,
                 transform: Optional[Callable] = None):
        """
        Args:
            tree_distance_csv_path: Path to CSV file with distance matrix. This is created in R with tree_distances = ape::cophenetic.phylo(out_tree) and written to a file with  write.csv.
            feature_csv_path_with_missing_target: Path to CSV file with node features. Columns should be feature names and rows should be node names. The target column should be missing for test nodes.
            ground_truth_csv_path: Path to CSV file with node features. Columns should be feature names and rows should be node names. Should include all values for training and test nodes.
            target_name: Name of target column in feature CSV file.
            binary_or_continuous: Target is binary or continuous.
            threshold: Distance threshold for creating edges (edges for distances <= threshold)
            k_nearest: Create edges to k-nearest neighbors per node
            transform: PyTorch Geometric transforms which transform values stored in 'x'
        """
        self.tree_distance_csv_path = tree_distance_csv_path
        self.threshold = threshold
        self.k_nearest = k_nearest

        if threshold is not None and k_nearest is not None:
            raise NotImplementedError("Only one of threshold or k_nearest can be set.")

        self.binary_or_continuous = binary_or_continuous

        # Read the tree CSV file
        self.tree_distance_df = pd.read_csv(tree_distance_csv_path, index_col=0)
        self.node_names = self.tree_distance_df.index.tolist()
        self.dist_matrix = self.tree_distance_df.values

        # Read the feature CSV file. It holds
        self.feature_with_missing_target_df = pd.read_csv(feature_csv_path_with_missing_target, index_col=0)
        self.ground_truth_df = pd.read_csv(ground_truth_csv_path, index_col=0)

        self.target_name = target_name

        super().__init__(transform=transform)

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
            edge_weights = []

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
                    edge_weights.append(self.dist_matrix[i, j])

            edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
            edge_attr = torch.tensor(edge_weights, dtype=torch.float).view(-1, 1)

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
            edge_attr=edge_attr,
            # num_nodes=num_nodes,
            # node_names=self.node_names,
            # dist_matrix=torch.tensor(self.dist_matrix, dtype=torch.float)
        )

        if self.transform is not None:
            raise NotImplementedError('This is setting everything to Nan here when some of the values are missing.')
            data = self.transform(data)

        return data


class NewickDataset(GenericPhyloDataset):
    """Dataset from Newick tree file i.e. where all nodes from the original tree are present."""

    def __init__(self,
                 newick_tree_path: str,
                 feature_csv_path_with_missing_target: str,
                 ground_truth_csv_path: str,
                 target_name: str,
                 binary_or_continuous: str,
                 transform: Optional[Callable] = None):
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

        self.binary_or_continuous = binary_or_continuous

        # Read the tree CSV file
        self.networkx_tree = self.newick_to_networkx(newick_tree_path)
        # Nodes should all be named 'Node_x' where x is the node number
        # Can set this in R with `paste("Node",1L:tree$Nnode, sep='_') -> tree$node.label`
        self.node_names = list(self.networkx_tree.nodes)
        self.nodes_that_arent_tips = [n for n in self.node_names if 'Node_' in n]

        # Read the feature CSV file. It holds
        self.feature_with_missing_target_df = pd.read_csv(feature_csv_path_with_missing_target, index_col=0)
        self.ground_truth_df = pd.read_csv(ground_truth_csv_path, index_col=0)
        self.target_name = target_name

        super().__init__(transform=transform)

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
                           weight=child.branch_length)

        return G

    def _process(self):
        edge_attr = torch.tensor([self.networkx_tree[u][v]['weight'] for u, v in self.networkx_tree.edges()]).view(-1, 1)
        pyg_data = from_networkx(self.networkx_tree)
        X, y, train_mask, test_mask, y_dtype = self.get_features_and_masks()

        data = Data(
            x=torch.tensor(X, dtype=torch.float),
            y=torch.tensor(y, dtype=y_dtype),
            train_mask=train_mask,
            test_mask=test_mask,
            edge_index=pyg_data.edge_index,
            edge_attr=edge_attr
        )

        if self.transform is not None:
            raise NotImplementedError('This is setting everything to Nan here when some of the values are missing.')
            data = self.transform(data)

        return data


def main():
    # dataset = Planetoid(root='data/Planetoid', name='Cora')
    # planetoid_data = dataset[0]
    # Load the dataset
    # dataset1 = DistanceMatrixDataset(
    #     tree_distance_csv_path='my_data/binary/tree_distances.csv',
    #     feature_csv_path_with_missing_target='my_data/binary/mcar_values.csv',
    #     ground_truth_csv_path='my_data/binary/ground_truth.csv',
    #     target_name='trait_BM_trend_scaled',
    #     binary_or_continuous='binary',
    #     k_nearest=50,  # Alternative: connect to 2 nearest neighbors
    #     transform=torch_geometric.transforms.NormalizeFeatures()
    #
    # )

    dataset1 = NewickDataset(
        newick_tree_path='my_data/binary/tree.tre',
        feature_csv_path_with_missing_target='my_data/binary/mcar_values.csv',
        ground_truth_csv_path='my_data/binary/ground_truth.csv',
        target_name='trait_BM_trend_scaled',
        binary_or_continuous='binary',
        # transform=torch_geometric.transforms.NormalizeFeatures()

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
        print(f"Edge attribute shape: {data.edge_attr.shape}")
        print(f'Number of training nodes: {int(data.train_mask.sum())}')
        print(f'Number of testing nodes: {int(data.test_mask.sum())}')

        # You can now use this like any PyG dataset
        print(f"\nEdge indices (first 5): {data.edge_index[:, :5]}")
        print(f"Edge attributes (first 5): {data.edge_attr[:5].flatten()}")


if __name__ == '__main__':
    main()
