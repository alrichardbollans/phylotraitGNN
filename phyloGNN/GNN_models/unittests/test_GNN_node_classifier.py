import copy
import os
import unittest

import torch
import torch.nn.functional as F
from phyloGNN.GNN_models.GNN_node_classifier import GCN
from torch_geometric.data import Data

from phyloGNN.parsing_tree_data import DistanceMatrixDataset, NewickDataset


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


    def test_train_step(self):
        model = GCN(self.dataset, hidden_channels=4)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        loss_function = F.cross_entropy

        loss = model.train_step(self.data, optimizer, loss_function)
        self.assertIsInstance(loss.item(), float)
        self.assertGreaterEqual(loss.item(), 0.0)

    def test_test_method(self):
        model = GCN(self.dataset, hidden_channels=4)

        with torch.no_grad():
            test_acc, brier_score = model.test(self.data)

        self.assertIsInstance(test_acc, float)
        self.assertGreaterEqual(test_acc, 0.0)
        self.assertLessEqual(test_acc, 1.0)
        self.assertIsInstance(brier_score, float)
        self.assertGreaterEqual(brier_score, 0.0)

    def for_a_dataset(self, dataset):
        model = GCN(dataset, hidden_channels=4)

        data = dataset.data
        loss_function = torch.nn.CrossEntropyLoss()
        optimizer_class = torch.optim.Adam
        optimizer_kwargs = {'lr': 0.01, 'weight_decay': 5e-4}
        optimizer = optimizer_class(model.parameters(), **optimizer_kwargs)
        test_acc_1, brier1 = model.test(data)

        for epoch in range(1, 100):
            loss = model.train_step(data, optimizer, loss_function)
            print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}')

        # Check training has changed scores
        test_acc_, brier = model.test(data)
        # self.assertNotEqual(test_acc_1, test_acc_)
        self.assertNotEqual(brier1, brier)

        # Check it's not just outputting the same values
        out_ = model(data.x, data.edge_index, edge_attr=data.edge_attr)
        test_pred_proba = set([round(c, 5) for c in set(out_[data.test_mask].detach().cpu().numpy()[:, 1])])
        self.assertGreaterEqual(len(test_pred_proba), 2)

        train_pred_proba = set([round(c, 5) for c in set(out_[data.train_mask].detach().cpu().numpy()[:, 1])])
        self.assertGreaterEqual(len(train_pred_proba), 2)

        self.assertEqual(out_.shape, (data.x.shape[0], dataset.num_classes))

        # Test edge attributes are being used
        with torch.no_grad():
            # Clone data, zero out edge_attr if it exists
            data_no_edge_attr = copy.deepcopy(data)
            if hasattr(data, 'edge_attr'):
                data_no_edge_attr.edge_attr = torch.zeros_like(data.edge_attr)
            out_with_none = model(data.x, data.edge_index, None)
            out_without = model(data_no_edge_attr.x, data_no_edge_attr.edge_index, data_no_edge_attr.edge_attr)
            assert not torch.allclose(out_, out_without), "Model outputs are unchanged by edge_attr!"
            assert not torch.allclose(out_, out_with_none), "Model outputs are unchanged by edge_attr!"

    def test_distance_training_process(self):

        dataset = DistanceMatrixDataset(
            tree_distance_csv_path='../../parsing_tree_data/unittest_data/binary/tree_distances.csv',
            feature_csv_path_with_missing_target='../../parsing_tree_data/unittest_data/binary/mcar_values.csv',
            ground_truth_csv_path='../../parsing_tree_data/unittest_data/binary/ground_truth.csv',
            target_name='trait_BM_trend_scaled',
            binary_or_continuous='binary',
            k_nearest=50

        )
        self.for_a_dataset(dataset)

    def test_distance_training_process_full(self):

        dataset = DistanceMatrixDataset(
            tree_distance_csv_path='../../parsing_tree_data/unittest_data/binary/tree_distances.csv',
            feature_csv_path_with_missing_target='../../parsing_tree_data/unittest_data/binary/mcar_values.csv',
            ground_truth_csv_path='../../parsing_tree_data/unittest_data/binary/ground_truth.csv',
            target_name='trait_BM_trend_scaled',
            binary_or_continuous='binary'

        )
        self.for_a_dataset(dataset)

    def test_newick_training_process(self):

        dataset = NewickDataset(
            newick_tree_path='../../parsing_tree_data/unittest_data/binary/tree.tre',
            feature_csv_path_with_missing_target='../../parsing_tree_data/unittest_data/binary/mcar_values.csv',
            ground_truth_csv_path='../../parsing_tree_data/unittest_data/binary/ground_truth.csv',
            target_name='trait_BM_trend_scaled',
            binary_or_continuous='binary'

        )
        self.for_a_dataset(dataset)

    def test_Newick_with_no_features(self):
        dataset = NewickDataset(
            newick_tree_path='../../parsing_tree_data/unittest_data/binary_no_features/tree.tre',
            feature_csv_path_with_missing_target='../../parsing_tree_data/unittest_data/binary_no_features/mcar_values.csv',
            ground_truth_csv_path='../../parsing_tree_data/unittest_data/binary_no_features/ground_truth.csv',
            target_name='trait_ARD',
            binary_or_continuous='binary',

        )
        self.for_a_dataset(dataset)
    def test_distance_with_no_features(self):
        dataset = DistanceMatrixDataset(
            tree_distance_csv_path='../../parsing_tree_data/unittest_data/binary_no_features/tree_distances.csv',
            feature_csv_path_with_missing_target='../../parsing_tree_data/unittest_data/binary_no_features/mcar_values.csv',
            ground_truth_csv_path='../../parsing_tree_data/unittest_data/binary_no_features/ground_truth.csv',
            target_name='trait_ARD',
            binary_or_continuous='binary',

        )
        self.for_a_dataset(dataset)


if __name__ == "__main__":
    unittest.main()
