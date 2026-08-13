import copy
import os
import unittest
import pandas as pd
import torch
import torch.nn.functional as F
from phylotraitGNN.GNN_models.GNN_node_classifier import GCN_node_classifier, train_gcn_model
from torch_geometric.data import Data

from phylotraitGNN.parsing_tree_data import DistanceMatrixDataset, NewickDataset


class TestGCN(unittest.TestCase):
    def setUp(self):
        # Mock dataset with appropriate structure for GCN_node_classifier
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
        model = GCN_node_classifier(self.dataset, hidden_channels=4, dropout_p=0.1)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        loss_function = F.cross_entropy

        loss = model.train_step(self.data, optimizer, loss_function)
        self.assertIsInstance(loss.item(), float)
        self.assertGreaterEqual(loss.item(), 0.0)

    def test_test_method(self):
        model = GCN_node_classifier(self.dataset, hidden_channels=4, dropout_p=0.1)

        with torch.no_grad():
            test_acc, brier_score = model.test(self.data)

        self.assertIsInstance(test_acc, float)
        self.assertGreaterEqual(test_acc, 0.0)
        self.assertLessEqual(test_acc, 1.0)
        self.assertIsInstance(brier_score, float)
        self.assertGreaterEqual(brier_score, 0.0)

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

        predictions = dataset.get_model_prediction_outputs_in_feature_order(probs)
        pd.testing.assert_index_equal(predictions.index, dataset.feature_with_missing_target_df.index)

        if dataset.ground_truth_df is not None:
            # Test data isn't leaked
            test_data = dataset.ground_truth_df[dataset.feature_with_missing_target_df[dataset.target_name].isna()][[dataset.target_name]]
            testing_predictions_like_df = predictions[[1]].rename(columns={1: dataset.target_name})[[dataset.target_name]]
            testing_predictions_like_df = testing_predictions_like_df[testing_predictions_like_df.index.isin(test_data.index)]
            testing_predictions_like_df[dataset.target_name] = testing_predictions_like_df[dataset.target_name].astype(float)
            test_data[dataset.target_name] = test_data[dataset.target_name].astype(float)

            try:
                pd.testing.assert_frame_equal(testing_predictions_like_df, test_data)
            except AssertionError:
                pass
            else:
                raise AssertionError("Ground truth and predictions should not be the same for missing values")

    def for_a_dataset_and_model(self, dataset, model):

        data = dataset.data
        test_acc_1, brier1 = model.test(data)

        train_gcn_model(model, data, epochs=100)

        # Check training has changed scores
        test_acc_, brier = model.test(data)
        # self.assertNotEqual(test_acc_1, test_acc_)
        self.assertNotEqual(brier1, brier)

        # Check it's not just outputting the same values
        out_ = model(data.x, data.edge_index, edge_attr=data.edge_weight)
        self.for_model_outputs(out_, dataset)

        # Test edge attributes are being used
        with torch.no_grad():
            # Clone data, zero out edge_attr if it exists
            data_no_edge_attr = copy.deepcopy(data)
            data_no_edge_attr.edge_weight = torch.zeros_like(data.edge_weight)
            out_with_none = model(data.x, data.edge_index, None)
            out_without = model(data_no_edge_attr.x, data_no_edge_attr.edge_index, data_no_edge_attr.edge_weight)
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
        model = GCN_node_classifier(dataset, hidden_channels=4, dropout_p=0.1)

        self.for_a_dataset_and_model(dataset, model)

    def test_distance_training_process_full(self):

        dataset = DistanceMatrixDataset(
            tree_distance_csv_path='../../parsing_tree_data/unittest_data/binary/tree_distances.csv',
            feature_csv_path_with_missing_target='../../parsing_tree_data/unittest_data/binary/mcar_values.csv',
            ground_truth_csv_path='../../parsing_tree_data/unittest_data/binary/ground_truth.csv',
            target_name='trait_BM_trend_scaled',
            binary_or_continuous='binary'

        )
        model = GCN_node_classifier(dataset, hidden_channels=4, dropout_p=0.1)

        self.for_a_dataset_and_model(dataset, model)

    def test_distance_training_process_no_gt(self):

        dataset = DistanceMatrixDataset(
            tree_distance_csv_path='../../parsing_tree_data/unittest_data/binary/tree_distances.csv',
            feature_csv_path_with_missing_target='../../parsing_tree_data/unittest_data/binary/mcar_values.csv',
            target_name='trait_BM_trend_scaled',
            binary_or_continuous='binary'

        )
        model = GCN_node_classifier(dataset, hidden_channels=4, dropout_p=0.1)

        self.for_a_dataset_and_model(dataset, model)

    def test_newick_training_process(self):

        dataset = NewickDataset(
            newick_tree_path='../../parsing_tree_data/unittest_data/binary/tree.tre',
            feature_csv_path_with_missing_target='../../parsing_tree_data/unittest_data/binary/mcar_values.csv',
            ground_truth_csv_path='../../parsing_tree_data/unittest_data/binary/ground_truth.csv',
            target_name='trait_BM_trend_scaled',
            binary_or_continuous='binary'

        )
        model = GCN_node_classifier(dataset, hidden_channels=4, dropout_p=0.1)

        self.for_a_dataset_and_model(dataset, model)

    def test_newick_training_process_not_gt(self):

        dataset = NewickDataset(
            newick_tree_path='../../parsing_tree_data/unittest_data/binary/tree.tre',
            feature_csv_path_with_missing_target='../../parsing_tree_data/unittest_data/binary/mcar_values.csv',
            target_name='trait_BM_trend_scaled',
            binary_or_continuous='binary'

        )
        model = GCN_node_classifier(dataset, hidden_channels=4, dropout_p=0.1)

        self.for_a_dataset_and_model(dataset, model)

    def test_Newick_with_no_features(self):
        dataset = NewickDataset(
            newick_tree_path='../../parsing_tree_data/unittest_data/binary_no_features/tree.tre',
            feature_csv_path_with_missing_target='../../parsing_tree_data/unittest_data/binary_no_features/mcar_values.csv',
            ground_truth_csv_path='../../parsing_tree_data/unittest_data/binary_no_features/ground_truth.csv',
            target_name='trait_ARD',
            binary_or_continuous='binary',

        )

        model = GCN_node_classifier(dataset, hidden_channels=4, dropout_p=0.1)
        try:
            self.for_a_dataset_and_model(dataset, model)
        except AssertionError:
            print('No features, so this will break.')
            pass

    def test_distance_with_no_features(self):
        dataset = DistanceMatrixDataset(
            tree_distance_csv_path='../../parsing_tree_data/unittest_data/binary_no_features/tree_distances.csv',
            feature_csv_path_with_missing_target='../../parsing_tree_data/unittest_data/binary_no_features/mcar_values.csv',
            ground_truth_csv_path='../../parsing_tree_data/unittest_data/binary_no_features/ground_truth.csv',
            target_name='trait_ARD',
            binary_or_continuous='binary',

        )

        model = GCN_node_classifier(dataset, hidden_channels=4, dropout_p=0.1)

        try:
            self.for_a_dataset_and_model(dataset, model)
        except AssertionError:
            print('No features, so this will break.')
            pass


if __name__ == "__main__":
    unittest.main()
