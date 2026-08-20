import unittest

import torch
import pandas as pd
from torch_geometric.data import Data
from torch_geometric.nn import LabelPropagation

from phyloGNNy.LP_models import propagate_labels, test_binary_LP_outputs, my_post_step
from phyloGNNy.parsing_tree_data import DistanceMatrixDataset, NewickDataset


class TestGCN(unittest.TestCase):

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

        # Check train outputs preserved
        train_outs = out_[data.train_mask][:, 1]
        train_ins = dataset.data.y[dataset.data.train_mask]
        assert torch.equal(train_outs, train_ins)

        # Check test outputs not leaked
        test_outs = out_[data.test_mask][:, 1]
        test_ins = dataset.data.y[dataset.data.test_mask]
        assert torch.any(torch.not_equal(test_outs, test_ins))

        if hasattr(dataset.data, 'val_mask'):
            val_outs = out_[data.val_mask][:, 1]
            val_ins = dataset.data.y[dataset.data.val_mask]
            assert torch.any(torch.not_equal(val_outs, val_ins))

        # Check the function getting the dataframe
        predictions = dataset.get_model_prediction_outputs_in_feature_order(out_)
        pd.testing.assert_index_equal(predictions.index, dataset.feature_with_missing_target_df.index)

        # This check ensures that the output predictions properly preserve the order of the features in the input dataset.
        # And also preserve the training values.
        if not hasattr(dataset.data, 'val_mask'):
            feature_data_without_nans = dataset.feature_with_missing_target_df.dropna(subset=[dataset.target_name])
            training_predictions_like_df = predictions[[1]].rename(columns={1: dataset.target_name})
            training_predictions_like_df = training_predictions_like_df[training_predictions_like_df.index.isin(feature_data_without_nans.index)]
            training_predictions_like_df[dataset.target_name] = training_predictions_like_df[dataset.target_name].astype(float)
            feature_data_without_nans[dataset.target_name] = feature_data_without_nans[dataset.target_name].astype(float)
            pd.testing.assert_frame_equal(training_predictions_like_df, feature_data_without_nans)

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

    def test_Newick_with_no_features(self):
        dataset = NewickDataset(
            newick_tree_path='../../parsing_tree_data/unittest_data/binary_no_features/tree.tre',
            feature_csv_path_with_missing_target='../../parsing_tree_data/unittest_data/binary_no_features/mcar_values.csv',
            ground_truth_csv_path='../../parsing_tree_data/unittest_data/binary_no_features/ground_truth.csv',
            target_name='trait_ARD',
            binary_or_continuous='binary', add_self_loops=True

        )
        self.for_model_outputs(propagate_labels(dataset), dataset)
        self.for_model_outputs(propagate_labels(dataset, alpha=0.99), dataset)

    def test_distance_with_no_features(self):
        dataset = DistanceMatrixDataset(
            tree_distance_csv_path='../../parsing_tree_data/unittest_data/binary_no_features/tree_distances.csv',
            feature_csv_path_with_missing_target='../../parsing_tree_data/unittest_data/binary_no_features/mcar_values.csv',
            ground_truth_csv_path='../../parsing_tree_data/unittest_data/binary_no_features/ground_truth.csv',
            target_name='trait_ARD',
            binary_or_continuous='binary', add_self_loops=True

        )
        self.for_model_outputs(propagate_labels(dataset), dataset)

        dataset = DistanceMatrixDataset(
            tree_distance_csv_path='../../parsing_tree_data/unittest_data/binary_no_features/tree_distances.csv',
            feature_csv_path_with_missing_target='../../parsing_tree_data/unittest_data/binary_no_features/mcar_values.csv',
            ground_truth_csv_path='../../parsing_tree_data/unittest_data/binary_no_features/ground_truth.csv',
            target_name='trait_ARD',
            binary_or_continuous='binary', add_self_loops=True, validation_nodes=['t88', 't75', 't74']

        )
        self.for_model_outputs(propagate_labels(dataset), dataset)

    def test_Newick_with_no_gt(self):
        dataset = NewickDataset(
            newick_tree_path='../../parsing_tree_data/unittest_data/binary_no_features/tree.tre',
            feature_csv_path_with_missing_target='../../parsing_tree_data/unittest_data/binary_no_features/mcar_values.csv',
            target_name='trait_ARD',
            binary_or_continuous='binary', add_self_loops=True

        )
        self.for_model_outputs(propagate_labels(dataset), dataset)

        dataset = NewickDataset(
            newick_tree_path='../../parsing_tree_data/unittest_data/binary_no_features/tree.tre',
            feature_csv_path_with_missing_target='../../parsing_tree_data/unittest_data/binary_no_features/mcar_values.csv',
            target_name='trait_ARD',
            binary_or_continuous='binary', add_self_loops=True, validation_nodes=['t88', 't75', 't2']

        )
        self.for_model_outputs(propagate_labels(dataset), dataset)

    def test_distance_with_no_gt(self):
        dataset = DistanceMatrixDataset(
            tree_distance_csv_path='../../parsing_tree_data/unittest_data/binary_no_features/tree_distances.csv',
            feature_csv_path_with_missing_target='../../parsing_tree_data/unittest_data/binary_no_features/mcar_values.csv',
            target_name='trait_ARD',
            binary_or_continuous='binary', add_self_loops=True

        )
        self.for_model_outputs(propagate_labels(dataset), dataset)

    def test_where_y_all_same_values(self):
        dataset = DistanceMatrixDataset(
            tree_distance_csv_path='../../parsing_tree_data/unittest_data/binary_no_features/tree_distances.csv',
            feature_csv_path_with_missing_target='../../parsing_tree_data/unittest_data/binary_no_features/mcar_values.csv',
            ground_truth_csv_path='../../parsing_tree_data/unittest_data/binary_no_features/ground_truth.csv',
            target_name='trait_ARD',
            binary_or_continuous='binary', add_self_loops=True

        )
        # make y a Tensor of shape y with all values the same
        dataset.data.y = torch.zeros(dataset.data.y.shape, dtype=torch.long)
        output = propagate_labels(dataset)
        assert output.shape == torch.Size([100, 2])
        test_binary_LP_outputs(output, dataset.data)

    def test_self_looping(self):
        dataset = NewickDataset(
            newick_tree_path='unittest_data/binary_no_features_simple/tree.tre',
            feature_csv_path_with_missing_target='unittest_data/binary_no_features_simple/mcar_values.csv',
            target_name='trait_ARD',
            binary_or_continuous='binary',
            sigma=1, add_self_loops=True

        )

        output = propagate_labels(dataset, alpha=0, num_layers=3)

        dataset = NewickDataset(
            newick_tree_path='unittest_data/binary_no_features_simple/tree.tre',
            feature_csv_path_with_missing_target='unittest_data/binary_no_features_simple/mcar_values.csv',
            target_name='trait_ARD',
            binary_or_continuous='binary',
            sigma=1

        )

        self.assertRaises(AssertionError, propagate_labels, dataset, alpha=0, num_layers=3)

    def test_specific_values(self):
        dataset = NewickDataset(
            newick_tree_path='unittest_data/binary_no_features_simple/tree.tre',
            feature_csv_path_with_missing_target='unittest_data/binary_no_features_simple/mcar_values.csv',
            target_name='trait_ARD',
            binary_or_continuous='binary',
            sigma=1, add_self_loops=True

        )
        train_mask = dataset.data.train_mask
        node_mask = (~train_mask & ~dataset.data.test_mask)

        # Case with alpha=0
        output = propagate_labels(dataset, alpha=0, num_layers=3)
        probs = output[train_mask][:, 1]  # Probability for class 1
        # assert that two tensors are equal
        assert torch.equal(dataset.data.y[train_mask], probs)
        #
        # # Case with alpha=1
        output = propagate_labels(dataset, alpha=1, num_layers=1)
        node_outputs = output[node_mask]
        assert torch.equal(node_outputs, torch.tensor([[0.5, 0.5], [1, 0], [0, 1]]))
        assert torch.equal(dataset.data.y[train_mask], output[train_mask][:,
                                                       1])  # This used to break because of https://github.com/pyg-team/pytorch_geometric/issues/10627#issuecomment-4018763551
        #
        # raise NotImplementedError('Add tests for num_layers = 2 etc')

        output = propagate_labels(dataset, alpha=0.8, num_layers=2)
        print(output)

    def test_labels_preserved_after_propagation(self):
        # Test that labeled node values are preserved after propagation:
        y2 = torch.tensor([0, 0, 0, 0, 0, 0, 0, 1, 1])
        edge_index2 = torch.tensor(
            [[0, 1, 2, 3, 4, 5, 6, 7, 8], [0, 0, 0, 0, 0, 0, 0, 0, 0]])
        mask2 = torch.tensor(
            [False, True, True, True, True, True, True, True, True])
        edge_weight2 = torch.ones(9)
        model2 = LabelPropagation(num_layers=2, alpha=0.9)
        from torch_geometric.utils import one_hot
        y2_oh = one_hot(y2)

        def post_step_with_mask_and_y(model_out):
            return my_post_step(model_out, mask2, y2_oh)

        out2 = model2(y=y2, edge_index=edge_index2, mask=mask2,
                      edge_weight=edge_weight2, post_step=post_step_with_mask_and_y)
        # Labeled nodes must retain their original one-hot values

        assert torch.allclose(out2[mask2], out2[mask2])
        assert torch.allclose(out2[mask2], y2_oh[mask2].float())


if __name__ == "__main__":
    unittest.main()
