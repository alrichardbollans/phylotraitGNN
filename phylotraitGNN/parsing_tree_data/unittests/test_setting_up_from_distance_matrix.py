import os
import unittest

import networkx as nx
import numpy as np
import pandas as pd
import torch
import torch_geometric
from matplotlib import pyplot as plt
from torch_geometric.data import DataLoader

from phylotraitGNN.parsing_tree_data.set_up_data import DistanceMatrixDataset


class TestDistanceMatrixDataset(unittest.TestCase):
    def setUp(self):
        # Create temporary test CSV files for distance, feature, and ground truth data
        self.tree_distance_csv_path = "test_tree_distance.csv"
        self.feature_csv_path = "test_feature.csv"
        self.ground_truth_csv_path = "test_ground_truth.csv"

        # Mock distance matrix data
        distance_data = pd.DataFrame(
            data=np.array([[0.0, 1.0, 2.0], [1.0, 0.0, 1.5], [2.0, 1.5, 0.0]]),
            columns=["Node1", "Node2", "Node3"],
            index=["Node3", "Node1", "Node2"]
        )
        distance_data.to_csv(self.tree_distance_csv_path)

        # Mock feature data with missing target
        feature_data = pd.DataFrame(
            data={"Feature1": [1.0, 2.0, 4], "Feature2": [2, 4, 8], "Target": [0.0, np.nan, np.nan]},
            index=["Node1", "Node2", "Node3"]
        )
        feature_data.to_csv(self.feature_csv_path)

        # Mock ground truth data
        ground_truth_data = pd.DataFrame(
            data={"Feature1": [1.0, 2.0, 4], "Feature2": [2, 4, 8], "Target": [0.0, 1.0, 1.0]},
            index=["Node1", "Node2", "Node3"]
        )
        ground_truth_data.to_csv(self.ground_truth_csv_path)

    def tearDown(self):
        # Remove temporary test CSV files
        os.remove(self.tree_distance_csv_path)
        os.remove(self.feature_csv_path)
        os.remove(self.ground_truth_csv_path)

    def test_dataset_initialization(self):
        dataset = DistanceMatrixDataset(
            tree_distance_csv_path=self.tree_distance_csv_path,
            feature_csv_path_with_missing_target=self.feature_csv_path,
            ground_truth_csv_path=self.ground_truth_csv_path,
            target_name="Target",
            binary_or_continuous="binary"
        )

        self.assertEqual(dataset.len(), 1)
        self.assertEqual(len(dataset.node_names), 3)
        self.assertIn("Node1", dataset.node_names)
        self.assertIsInstance(dataset.data.y, torch.Tensor)
        self.assertIs(dataset.data.y.dtype, torch.int64)

        g = torch_geometric.utils.to_networkx(dataset[0])
        nx.draw(g)
        plt.show()

        dataset = DistanceMatrixDataset(
            tree_distance_csv_path=self.tree_distance_csv_path,
            feature_csv_path_with_missing_target=self.feature_csv_path,
            ground_truth_csv_path=self.ground_truth_csv_path,
            target_name="Target",
            binary_or_continuous="continuous"
        )

        self.assertEqual(dataset.len(), 1)
        self.assertEqual(len(dataset.node_names), 3)
        self.assertIn("Node1", dataset.node_names)
        self.assertIsInstance(dataset.data.y, torch.Tensor)
        self.assertIs(dataset.data.y.dtype, torch.float)

    def test_dataset_process(self):
        dataset = DistanceMatrixDataset(
            tree_distance_csv_path=self.tree_distance_csv_path,
            feature_csv_path_with_missing_target=self.feature_csv_path,
            ground_truth_csv_path=self.ground_truth_csv_path,
            target_name="Target",
            binary_or_continuous="binary"
        )

        data = dataset.get(0)
        self.assertEqual(data.num_nodes, 3)
        self.assertTrue(data.train_mask[1].item())
        self.assertFalse(data.train_mask[0].item())
        self.assertFalse(data.train_mask[2].item())
        self.assertTrue(data.test_mask[0].item())
        self.assertTrue(data.test_mask[2].item())
        self.assertFalse(data.test_mask[1].item())
        self.assertEqual(len(data.edge_index[0]), 6)  # Undirected graph, i.e.

        self.assertEqual(data.node_stores[0]['x'][0, 0], 4)
        self.assertEqual(data.node_stores[0]['x'][0, 1], 8)
        self.assertEqual(data.node_stores[0]['x'][2, 0], 2)
        self.assertEqual(data.node_stores[0]['x'][2, 1], 4)

        self.assertEqual(data.node_stores[0]['y'][0], 1)
        self.assertEqual(data.node_stores[0]['y'][1], 0)
        self.assertEqual(data.node_stores[0]['y'][2], 1)

    def test_dataset_process_no_ground_truth(self):
        dataset = DistanceMatrixDataset(
            tree_distance_csv_path=self.tree_distance_csv_path,
            feature_csv_path_with_missing_target=self.feature_csv_path,
            target_name="Target",
            binary_or_continuous="binary"
        )

        data = dataset.get(0)
        self.assertEqual(data.num_nodes, 3)
        self.assertTrue(data.train_mask[1].item())
        self.assertFalse(data.train_mask[0].item())
        self.assertFalse(data.train_mask[2].item())
        self.assertTrue(data.test_mask[0].item())
        self.assertTrue(data.test_mask[2].item())
        self.assertFalse(data.test_mask[1].item())
        self.assertEqual(len(data.edge_index[0]), 6)  # Undirected graph, i.e.

        self.assertEqual(data.node_stores[0]['x'][0, 0], 4)
        self.assertEqual(data.node_stores[0]['x'][0, 1], 8)
        self.assertEqual(data.node_stores[0]['x'][2, 0], 2)
        self.assertEqual(data.node_stores[0]['x'][2, 1], 4)

        self.assertEqual(data.node_stores[0]['y'][0], 0)
        nan_value = torch.tensor(np.array([np.nan]), dtype=torch.int64).numpy()[0]
        self.assertEqual(data.node_stores[0]['y'][1], 0)
        self.assertEqual(data.node_stores[0]['y'][2], 0)

    def test_invalid_binary_or_continuous(self):
        with self.assertRaises(ValueError):
            DistanceMatrixDataset(
                tree_distance_csv_path=self.tree_distance_csv_path,
                feature_csv_path_with_missing_target=self.feature_csv_path,
                ground_truth_csv_path=self.ground_truth_csv_path,
                target_name="Target",
                binary_or_continuous="invalid"
            )

    def test_invalid_val_nodes(self):
        with self.assertRaises(ValueError):
            DistanceMatrixDataset(
                tree_distance_csv_path=self.tree_distance_csv_path,
                feature_csv_path_with_missing_target=self.feature_csv_path,
                ground_truth_csv_path=self.ground_truth_csv_path,
                target_name="Target",
                binary_or_continuous="binary",
                validation_nodes=['ppppp', 'notadnode']
            )

    def test_edge_creation_with_threshold(self):
        dataset = DistanceMatrixDataset(
            tree_distance_csv_path=self.tree_distance_csv_path,
            feature_csv_path_with_missing_target=self.feature_csv_path,
            ground_truth_csv_path=self.ground_truth_csv_path,
            target_name="Target",
            binary_or_continuous="binary",
            threshold=1.0
        )

        data = dataset.get(0)
        self.assertIsNotNone(data.edge_index)
        self.assertEqual(data.edge_index.size(1), 5)  # Only edges with distances <= 1.0

    def test_edge_creation_with_k_nearest(self):
        dataset = DistanceMatrixDataset(
            tree_distance_csv_path=self.tree_distance_csv_path,
            feature_csv_path_with_missing_target=self.feature_csv_path,
            ground_truth_csv_path=self.ground_truth_csv_path,
            target_name="Target",
            binary_or_continuous="binary",
            k_nearest=1
        )

        g = torch_geometric.utils.to_networkx(dataset[0])
        nx.draw(g)
        plt.show()

        data = dataset.get(0)
        self.assertIsNotNone(data.edge_index)
        self.assertEqual(data.edge_index.size(1), 4)  # Each node connects to 1 nearest node

    def test_including_validation_nodes(self):
        nw_dataset = DistanceMatrixDataset(
            tree_distance_csv_path=self.tree_distance_csv_path,
            feature_csv_path_with_missing_target=self.feature_csv_path,
            ground_truth_csv_path=self.ground_truth_csv_path,
            target_name="Target",
            binary_or_continuous="binary",
            k_nearest=1,
            validation_nodes=["Node2"]
        )

        assert (nw_dataset.data.val_mask).sum() == 1
        # Ensure node_names is a numpy array for boolean indexing support
        node_names = np.array(nw_dataset.node_names)
        val_node_names = node_names[nw_dataset.data.val_mask.cpu().numpy()]  # Convert mask to numpy if needed
        # Now test that the selected names match what is expected (order might matter)
        expected_names = np.array(["Node2"])
        assert np.array_equal(val_node_names, expected_names)

    def test_knearest(self):
        dataset = DistanceMatrixDataset(
            tree_distance_csv_path='../unittest_data/binary/tree_distances.csv',
            feature_csv_path_with_missing_target='../unittest_data/binary/mcar_values.csv',
            ground_truth_csv_path='../unittest_data/binary/ground_truth.csv',
            target_name='trait_BM_trend_scaled',
            binary_or_continuous='binary',
            k_nearest=10
        )

        g = torch_geometric.utils.to_networkx(dataset[0], to_undirected=True)
        nx.draw(g)
        plt.show()

    def test_dataset_initialization_proper_data(self):
        dataset = DistanceMatrixDataset(
            tree_distance_csv_path='../unittest_data/binary/tree_distances.csv',
            feature_csv_path_with_missing_target='../unittest_data/binary/mcar_values.csv',
            ground_truth_csv_path='../unittest_data/binary/ground_truth.csv',
            target_name='trait_BM_trend_scaled',
            binary_or_continuous='binary'
        )

        self.assertEqual(dataset.len(), 1)
        self.assertEqual(len(dataset.node_names), 100)
        self.assertIsInstance(dataset.data.y, torch.Tensor)
        self.assertIs(dataset.data.y.dtype, torch.int64)

        g = torch_geometric.utils.to_networkx(dataset[0], to_undirected=True)
        nx.draw(g)
        plt.show()

        import numpy as np
        from scipy.sparse.csgraph import minimum_spanning_tree

        D = dataset.dist_matrix  # dense, symmetric, shape [N, N]
        mst = minimum_spanning_tree(D)
        min_threshold = mst.data.max()  # smallest cutoff that keeps the graph connected
        print(min_threshold)

        dataset1 = DistanceMatrixDataset(
            tree_distance_csv_path='../unittest_data/binary/tree_distances.csv',
            feature_csv_path_with_missing_target='../unittest_data/binary/mcar_values.csv',
            ground_truth_csv_path='../unittest_data/binary/ground_truth.csv',
            target_name='trait_BM_trend_scaled',
            binary_or_continuous='binary',
            threshold=dataset.dist_matrix.max()/2
        )

        g = torch_geometric.utils.to_networkx(dataset1[0], to_undirected=True)
        nx.draw(g)
        plt.show()
        print(f"Number of edges: {dataset.data.edge_index.shape[1]}")
        print(f"Number of edges: {dataset1.data.edge_index.shape[1]}")
        assert dataset1.data.edge_index.shape[1]<dataset.data.edge_index.shape[1]
if __name__ == "__main__":
    unittest.main()
