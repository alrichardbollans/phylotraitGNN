import copy
import os
import unittest

import torch
import torch.nn.functional as F
from phylotraitGNN.GNN_models.GNN_node_classifier import GCN, train_gcn_model
from torch_geometric.data import Data

from phylotraitGNN.LP_models import propagate_labels
from phylotraitGNN.parsing_tree_data import DistanceMatrixDataset, NewickDataset


class TestGCN(unittest.TestCase):
    def setUp(self):
        # Mock dataset with appropriate structure for GCN
        num_nodes = 5
        num_features = 3
        num_classes = 2
        edge_index = torch.tensor([[0, 1, 2, 3, 4, 0],
                                   [1, 2, 3, 4, 0, 3]], dtype=torch.long)
        x = torch.rand((num_nodes, num_features))  # Random feature matrix
        y = torch.tensor([0, 1, 1, 0, 1], dtype=torch.long)  # Mock labels

        # Mock masks for training and testing
        train_mask = torch.tensor([True, True, False, False, False], dtype=torch.bool)
        test_mask = torch.tensor([False, False, True, True, True], dtype=torch.bool)

        self.data = Data(x=x, edge_index=edge_index, y=y, train_mask=train_mask, test_mask=test_mask)
        self.dataset = type("MockDataset", (object,), {"num_features": num_features, "num_classes": num_classes})

    def for_model_outputs(self, out_, dataset):
        data = dataset.data

        # First check that the outputs have multiple different values.
        test_outs = set([round(c, 5) for c in set(out_[data.test_mask].detach().cpu().numpy()[:, 1])])
        self.assertGreaterEqual(len(test_outs), 2)

        train_outs = set([round(c, 5) for c in set(out_[data.train_mask].detach().cpu().numpy()[:, 1])])
        self.assertGreaterEqual(len(train_outs), 2)

        self.assertEqual(out_.shape, (data.x.shape[0], dataset.num_classes))

        probs = F.softmax(out_, dim=1)  # Convert logits to probabilities.
        # Add assertion that values in probs >0 and <1
        self.assertTrue(torch.all(probs >= 0.0) and torch.all(probs <= 1.0))

        # Add assertion that rows in pred proba add up to 1
        self.assertAlmostEqual(probs.sum(dim=1).max().item(), 1.0, places=6)

    def test_Newick_with_no_features(self):
        dataset = NewickDataset(
            newick_tree_path='../../parsing_tree_data/unittest_data/binary_no_features/tree.tre',
            feature_csv_path_with_missing_target='../../parsing_tree_data/unittest_data/binary_no_features/mcar_values.csv',
            ground_truth_csv_path='../../parsing_tree_data/unittest_data/binary_no_features/ground_truth.csv',
            target_name='trait_ARD',
            binary_or_continuous='binary',

        )
        self.for_model_outputs(propagate_labels(dataset), dataset)
        self.for_model_outputs(propagate_labels(dataset, alpha=1), dataset)

    def test_distance_with_no_features(self):
        dataset = DistanceMatrixDataset(
            tree_distance_csv_path='../../parsing_tree_data/unittest_data/binary_no_features/tree_distances.csv',
            feature_csv_path_with_missing_target='../../parsing_tree_data/unittest_data/binary_no_features/mcar_values.csv',
            ground_truth_csv_path='../../parsing_tree_data/unittest_data/binary_no_features/ground_truth.csv',
            target_name='trait_ARD',
            binary_or_continuous='binary',

        )
        self.for_model_outputs(propagate_labels(dataset), dataset)

    def test_Newick_with_no_gt(self):
        dataset = NewickDataset(
            newick_tree_path='../../parsing_tree_data/unittest_data/binary_no_features/tree.tre',
            feature_csv_path_with_missing_target='../../parsing_tree_data/unittest_data/binary_no_features/mcar_values.csv',
            target_name='trait_ARD',
            binary_or_continuous='binary',

        )
        self.for_model_outputs(propagate_labels(dataset), dataset)

    def test_distance_with_no_gt(self):
        dataset = DistanceMatrixDataset(
            tree_distance_csv_path='../../parsing_tree_data/unittest_data/binary_no_features/tree_distances.csv',
            feature_csv_path_with_missing_target='../../parsing_tree_data/unittest_data/binary_no_features/mcar_values.csv',
            target_name='trait_ARD',
            binary_or_continuous='binary',

        )
        self.for_model_outputs(propagate_labels(dataset), dataset)


if __name__ == "__main__":
    unittest.main()
