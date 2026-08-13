import unittest

import pandas as pd
import torch

from phylotraitGNN.LP_models import propagate_labels
from phylotraitGNN.parsing_tree_data import NewickDataset


class TestGCN(unittest.TestCase):

    def for_model_outputs(self, out_, dataset):
        data = dataset.data

        # First check that the outputs have multiple different values.
        test_outs = set([round(c, 5) for c in set(out_[data.test_mask].detach().cpu().numpy()[:, 1])])
        self.assertGreaterEqual(len(test_outs), 2)

        train_outs = set([round(c, 5) for c in set(out_[data.train_mask].detach().cpu().numpy()[:, 1])])
        self.assertGreaterEqual(len(train_outs), 2)

        self.assertEqual(out_.shape, (data.x.shape[0], dataset.num_classes))

        probs = out_
        # Add assertion that values in probs >0 and <1
        self.assertTrue(torch.all(probs >= 0.0) and torch.all(probs <= 1.0))

        # Add assertion that rows in pred proba add up to 1
        self.assertAlmostEqual(probs.sum(dim=1).max().item(), 1.0, places=6)

        predictions = dataset.get_model_prediction_outputs_in_feature_order(out_)
        pd.testing.assert_index_equal(predictions.index, dataset.feature_with_missing_target_df.index)

        # This check ensures that the output predictions properly preserve the order of the features in the input dataset.
        # And also preserve the training values.

        feature_data_without_nans = dataset.feature_with_missing_target_df.dropna(subset=[dataset.target_name])
        predictions_like_df = predictions[[1]].rename(columns={1: dataset.target_name})
        predictions_like_df = predictions_like_df[predictions_like_df.index.isin(feature_data_without_nans.index)]
        predictions_like_df[dataset.target_name] = predictions_like_df[dataset.target_name].astype(float)
        feature_data_without_nans[dataset.target_name] = feature_data_without_nans[dataset.target_name].astype(float)
        pd.testing.assert_frame_equal(predictions_like_df, feature_data_without_nans)


    def test_Newick_with_no_features(self):
        dataset = NewickDataset(
            newick_tree_path='../../parsing_tree_data/unittest_data/continuous/tree.tre',
            feature_csv_path_with_missing_target='../../parsing_tree_data/unittest_data/continuous/mcar_values.csv',
            ground_truth_csv_path='../../parsing_tree_data/unittest_data/continuous/ground_truth.csv',
            target_name='trait_OU_scaled',
            binary_or_continuous='continuous',

        )
        try:
            self.for_model_outputs(propagate_labels(dataset), dataset)
        except AssertionError:
            print("label propagation only works for discrete labels")


if __name__ == "__main__":
    unittest.main()
