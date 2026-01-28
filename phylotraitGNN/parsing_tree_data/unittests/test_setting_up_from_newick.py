import os
import unittest

import networkx as nx
import numpy as np
import pandas as pd
import torch
import torch_geometric
from matplotlib import pyplot as plt

from phylotraitGNN.parsing_tree_data.set_up_data import NewickDataset


class TestNewickDataset(unittest.TestCase):
    def setUp(self):
        # Create temporary test CSV files for distance, feature, and ground truth data
        self.newick_tree_path = '../unittest_data/binary/tree.tre'
        self.feature_csv_path_with_missing_target = '../unittest_data/binary/mcar_values.csv'
        self.ground_truth_csv_path = '../unittest_data/binary/ground_truth.csv'
        self.target_name = 'trait_BM_trend_scaled'
        self.binary_or_continuous = 'binary'


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

        nan_indices = torch.isnan(data.y)
        # Some should be nan, some should not
        self.assertTrue(torch.any(nan_indices))
        self.assertTrue(torch.all(~nan_indices))
        # For all indices where y is nan, train_mask and test_mask must be False
        self.assertTrue(torch.all(~data.train_mask[nan_indices]))
        self.assertTrue(torch.all(~data.test_mask[nan_indices]))


        self.assertEqual(len(data.edge_index[0]), 6)  # Undirected graph, i.e.

        self.assertEqual(data.node_stores[0]['x'][0,0],1)
        self.assertEqual(data.node_stores[0]['x'][0,1],2)
        self.assertEqual(data.node_stores[0]['x'][2,0],4)
        self.assertEqual(data.node_stores[0]['x'][2,1],8)

        self.assertEqual(data.node_stores[0]['y'][0],0)
        self.assertEqual(data.node_stores[0]['y'][1],1)
        self.assertEqual(data.node_stores[0]['y'][2],1)

    def test_invalid_binary_or_continuous(self):
        with self.assertRaises(ValueError):
            NewickDataset(
                newick_tree_path=self.newick_tree_path,
                feature_csv_path_with_missing_target=self.feature_csv_path_with_missing_target,
                ground_truth_csv_path=self.ground_truth_csv_path,
                target_name=self.target_name,
                binary_or_continuous="invalid"
            )


if __name__ == "__main__":
    unittest.main()
