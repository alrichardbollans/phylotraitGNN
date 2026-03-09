import unittest

import torch

from torch_geometric.data import Data

from phylotraitGNN.LP_models import propagate_labels, test_binary_LP_outputs
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

        # Add assertion that values in probs >0 and <1
        self.assertTrue(torch.all(out_ >= 0.0) and torch.all(out_ <= 1.0))

        # Add assertion that rows in pred proba add up to 1
        self.assertAlmostEqual(out_.sum(dim=1).max().item(), 1.0, places=6)

    def test_Newick_with_no_features(self):
        dataset = NewickDataset(
            newick_tree_path='../../parsing_tree_data/unittest_data/binary_no_features/tree.tre',
            feature_csv_path_with_missing_target='../../parsing_tree_data/unittest_data/binary_no_features/mcar_values.csv',
            ground_truth_csv_path='../../parsing_tree_data/unittest_data/binary_no_features/ground_truth.csv',
            target_name='trait_ARD',
            binary_or_continuous='binary',

        )
        self.for_model_outputs(propagate_labels(dataset), dataset)
        self.for_model_outputs(propagate_labels(dataset, alpha=0.99), dataset)

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

    def test_where_y_all_same_values(self):
        dataset = DistanceMatrixDataset(
            tree_distance_csv_path='../../parsing_tree_data/unittest_data/binary_no_features/tree_distances.csv',
            feature_csv_path_with_missing_target='../../parsing_tree_data/unittest_data/binary_no_features/mcar_values.csv',
            ground_truth_csv_path='../../parsing_tree_data/unittest_data/binary_no_features/ground_truth.csv',
            target_name='trait_ARD',
            binary_or_continuous='binary',

        )
        # make y a Tensor of shape y with all values the same
        dataset.data.y = torch.zeros(dataset.data.y.shape, dtype=torch.long)
        output = propagate_labels(dataset)
        assert output.shape == torch.Size([100, 2])
        test_binary_LP_outputs(output, dataset.data)

    def test_specific_values(self):
        dataset = NewickDataset(
            newick_tree_path='unittest_data/binary_no_features_simple/tree.tre',
            feature_csv_path_with_missing_target='unittest_data/binary_no_features_simple/mcar_values.csv',
            target_name='trait_ARD',
            binary_or_continuous='binary',
            sigma=1

        )
        tip_mask = dataset.data.train_mask
        node_mask = ~tip_mask

        # Case with alpha=0
        output = propagate_labels(dataset, alpha=0, num_layers=3)
        probs = output[tip_mask][:, 1]  # Probability for class 1
        # assert that two tensors are equal
        assert torch.equal(dataset.data.y[tip_mask], probs)

        # Case with alpha=1
        output = propagate_labels(dataset, alpha=1, num_layers=1)
        node_outputs = output[node_mask]
        assert torch.equal(node_outputs, torch.tensor([[0, 1], [1, 0], [0, 1]]))
        assert torch.equal(dataset.data.y[tip_mask], output[tip_mask][:,
                                                     1])  # This will break because of https://github.com/pyg-team/pytorch_geometric/issues/10627#issuecomment-4018763551

        raise NotImplementedError('Add tests for num_layers = 2 etc')


if __name__ == "__main__":
    unittest.main()
