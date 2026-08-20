import os
import unittest

import networkx as nx
import numpy as np
import pandas as pd
import torch
import torch_geometric
from matplotlib import pyplot as plt

from phyloGNNy.parsing_tree_data.set_up_data import NewickDataset


class TestNewickDataset(unittest.TestCase):
    def setUp(self):
        # Create temporary test CSV files for distance, feature, and ground truth data
        self.newick_tree_path = '../unittest_data/binary/tree.tre'
        self.feature_csv_path_with_missing_target = '../unittest_data/binary/mcar_values.csv'
        self.ground_truth_csv_path = '../unittest_data/binary/ground_truth.csv'
        self.target_name = 'trait_BM_trend_scaled'
        self.binary_or_continuous = 'binary'

        self.newick_tree_path_continuous = '../unittest_data/continuous/tree.tre'
        self.feature_csv_path_with_missing_target_continuous = '../unittest_data/continuous/mcar_values.csv'
        self.ground_truth_csv_path_continuous = '../unittest_data/continuous/ground_truth.csv'
        self.target_name_cont = 'trait_OU_scaled'

    def test_dataset_initialization(self):
        dataset = NewickDataset(
            newick_tree_path=self.newick_tree_path,
            feature_csv_path_with_missing_target=self.feature_csv_path_with_missing_target,
            ground_truth_csv_path=self.ground_truth_csv_path,
            target_name=self.target_name,
            binary_or_continuous="binary"
        )

        self.assertEqual(dataset.len(), 1)
        self.assertEqual(len(dataset.node_names), 199)
        self.assertIn("Node_1", dataset.node_names)
        self.assertIsInstance(dataset.data.y, torch.Tensor)
        self.assertIs(dataset.data.y.dtype, torch.int64)

        if torch.cuda.is_available():
            print(f"Number of GPUs: {torch.cuda.device_count()}")
            print(f"Current GPU name: {torch.cuda.get_device_name(0)}")
            print(f"CUDA version: {torch.version.cuda}")
            print("GPU compute capability:", torch.cuda.get_device_capability(0))  # e.g., (6, 1) for Pascal
            print("PyTorch compiled for:", torch.cuda.get_arch_list())  # e.g., ['sm_75', 'sm_80']

            self.assertTrue(dataset.data.x.device == torch.device("cuda:0"))
            self.assertTrue(dataset.data.y.device == torch.device("cuda:0"))
            self.assertTrue(dataset.data.edge_weight.device == torch.device("cuda:0"))
            self.assertTrue(dataset.data.edge_index.device == torch.device("cuda:0"))
        else:
            print("CUDA is not available. PyG will run on CPU.")

        g = torch_geometric.utils.to_networkx(dataset[0], to_undirected=True)
        nx.draw(g)
        plt.show()

        dataset = NewickDataset(
            newick_tree_path=self.newick_tree_path,
            feature_csv_path_with_missing_target=self.feature_csv_path_with_missing_target,
            ground_truth_csv_path=self.ground_truth_csv_path,
            target_name=self.target_name,
            binary_or_continuous="continuous"
        )

        self.assertEqual(dataset.len(), 1)
        self.assertEqual(len(dataset.node_names), 199)
        self.assertIn("Node_1", dataset.node_names)
        self.assertIsInstance(dataset.data.y, torch.Tensor)
        self.assertIs(dataset.data.y.dtype, torch.float)

    def test_dataset_process(self):
        dataset = NewickDataset(
            newick_tree_path=self.newick_tree_path,
            feature_csv_path_with_missing_target=self.feature_csv_path_with_missing_target,
            ground_truth_csv_path=self.ground_truth_csv_path,
            target_name=self.target_name,
            binary_or_continuous="binary"
        )

        data = dataset.get(0)
        self.assertEqual(data.num_nodes, 199)

        self.assertEqual(len(data.edge_index[0]), 396)  # Undirected graph, i.e.

    def test_dataset_process_cont(self):
        dataset = NewickDataset(
            newick_tree_path=self.newick_tree_path_continuous,
            feature_csv_path_with_missing_target=self.feature_csv_path_with_missing_target_continuous,
            ground_truth_csv_path=self.ground_truth_csv_path_continuous,
            target_name=self.target_name_cont,
            binary_or_continuous="continuous"
        )

        data = dataset.get(0)
        self.assertEqual(data.num_nodes, 199)

        self.assertEqual(len(data.edge_index[0]), 396)  # Undirected graph, i.e.

    def test_self_looping(self):
        dataset = NewickDataset(
            newick_tree_path=self.newick_tree_path_continuous,
            feature_csv_path_with_missing_target=self.feature_csv_path_with_missing_target_continuous,
            ground_truth_csv_path=self.ground_truth_csv_path_continuous,
            target_name=self.target_name_cont,
            binary_or_continuous="continuous",
            sigma=1, add_self_loops=True

        )

    def test_no_ground_truth(self):
        dataset = NewickDataset(
            newick_tree_path=self.newick_tree_path,
            feature_csv_path_with_missing_target=self.feature_csv_path_with_missing_target,
            target_name=self.target_name,
            binary_or_continuous="binary"
        )

        data = dataset.get(0)
        self.assertEqual(data.num_nodes, 199)

    def test_invalid_binary_or_continuous(self):
        with self.assertRaises(ValueError):
            NewickDataset(
                newick_tree_path=self.newick_tree_path,
                feature_csv_path_with_missing_target=self.feature_csv_path_with_missing_target,
                ground_truth_csv_path=self.ground_truth_csv_path,
                target_name=self.target_name,
                binary_or_continuous="invalid"
            )

    def test_invalid_validation_nodes(self):
        with self.assertRaises(ValueError):
            NewickDataset(
                newick_tree_path=self.newick_tree_path,
                feature_csv_path_with_missing_target=self.feature_csv_path_with_missing_target,
                ground_truth_csv_path=self.ground_truth_csv_path,
                target_name=self.target_name,
                binary_or_continuous="binary",
                validation_nodes=['ppppp', 'notadnode']
            )

    def test_including_validation_nodes(self):
        nw_dataset = NewickDataset(
            newick_tree_path=self.newick_tree_path,
            feature_csv_path_with_missing_target=self.feature_csv_path_with_missing_target,
            ground_truth_csv_path=self.ground_truth_csv_path,
            target_name=self.target_name,
            binary_or_continuous="binary",
            validation_nodes=['t43', 't100', 't29']
        )

        assert (nw_dataset.data.val_mask).sum() == 3
        # Ensure node_names is a numpy array for boolean indexing support
        node_names = np.array(nw_dataset.node_names)
        val_node_names = node_names[nw_dataset.data.val_mask.cpu().numpy()]  # Convert mask to numpy if needed
        # Now test that the selected names match what is expected (order might matter)
        expected_names = np.array(['t43', 't100', 't29'])
        assert np.array_equal(val_node_names, expected_names)


if __name__ == "__main__":
    unittest.main()
