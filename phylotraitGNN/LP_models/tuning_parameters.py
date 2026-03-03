from copy import deepcopy

import numpy as np
import torch
from sklearn.metrics import make_scorer, brier_score_loss

from phylotraitGNN.LP_models import propagate_labels
from phylotraitGNN.parsing_tree_data import GenericPhyloDataset, NewickDataset

from bayes_opt import BayesianOptimization


def get_datasets_for_cross_validation(dataset: GenericPhyloDataset, number_of_splits: int = 5) -> list:
    ## For the given dataset, make five copies with where the training samples are split into train/val
    ## Do this by setting more values to false in the train mask and making a val_mask which is true for these values
    """
        Splits dataset.data.train_mask into train/val masks for cross-validation.

        Args:
            dataset (NewickDataset): Dataset with .data attribute containing .train_mask.
            number_of_splits (int): Number of cross-validation splits.

        Returns:
            List of datasets, each with split train_mask and val_mask.
        """
    data = dataset.data
    # Candidates are where train_mask == True (these can be split into train/val)
    candidates = torch.where(data.train_mask)[0].cpu().numpy()
    np.random.shuffle(candidates)  # Shuffle for random splits

    fold_sizes = np.full(number_of_splits, len(candidates) // number_of_splits, dtype=int)
    fold_sizes[:len(candidates) % number_of_splits] += 1
    current = 0

    datasets = []
    for fold in range(number_of_splits):
        val_indices = candidates[current: current + fold_sizes[fold]]
        train_indices = np.setdiff1d(candidates, val_indices)
        current += fold_sizes[fold]

        # Deepcopy dataset and data
        ds_copy = deepcopy(dataset)
        # Reset masks
        print('Note that nodes in initial training mask will have been used for feature propagation when setting up data.')
        ds_copy.data.train_mask = torch.zeros_like(data.train_mask, dtype=torch.bool)
        ds_copy.data.train_mask[train_indices] = True

        # New val_mask: same shape, True at val_indices
        val_mask = torch.zeros_like(data.train_mask, dtype=torch.bool)
        val_mask[val_indices] = True
        ds_copy.data.val_mask = val_mask

        datasets.append(ds_copy)
    return datasets


def find_LP_hyperparameters(dataset: GenericPhyloDataset, verbose: int = 2, init_points=10, n_iter=50):
    original_sigma = dataset.data.original_edge_std
    val_scorer = make_scorer(brier_score_loss, greater_is_better=False, response_method='predict_proba')
    bin_or_cont = 'binary'

    global _worst_score
    _worst_score = None

    def black_box_function(alpha, num_layers, sigma_ratio):
        """Function with unknown internals we wish to maximize.

        When NaN is returned for all values of cross_val_score, the current worst score is returned. This will break if the first try returns NaN.
        """

        main_dataset_with_new_sigma = NewickDataset(
            newick_tree_path=dataset.newick_tree_path,
            feature_csv_path_with_missing_target=dataset.feature_csv_path_with_missing_target,
            ground_truth_csv_path=dataset.ground_truth_csv_path,
            target_name=dataset.target_name,
            binary_or_continuous=dataset.binary_or_continuous,
            sigma=original_sigma * sigma_ratio,

        )
        cross_val_datasets = get_datasets_for_cross_validation(main_dataset_with_new_sigma, number_of_splits=5)

        for crossval_dataset in cross_val_datasets:
            out = propagate_labels(crossval_dataset, num_layers=num_layers, alpha=alpha)
            print(out)
        phyln = PhylNearestNeighbours(distance_matrix, clf, ratio, kappa, fill_in_unknowns_with_mean=False)

        cv_score = cross_val_score(phyln, X, y, cv=cv, scoring=scorer, n_jobs=njobs, params={'sample_weight': weights}, error_score="raise")
        out = np.mean(cv_score)
        global _worst_score
        if np.isnan(out):
            return _worst_score
        elif _worst_score is None:
            _worst_score = out
        elif out < _worst_score:
            _worst_score = out

        return out

    # Bounded region of parameter space
    pbounds = {'num_layers': (1, 50, int),
               'alpha': (0, 1.0),
               'sigma_ratio': (0.1, 10)}

    while True:

        try:
            optimizer = BayesianOptimization(
                f=black_box_function,
                pbounds=pbounds,
                random_state=None,
                verbose=verbose
            )

            optimizer.maximize(init_points=init_points, n_iter=n_iter)
            break
        except TypeError:
            print(f'WARNING: Bayesian optimization failed to initialise -- retrying..')

    print(optimizer.max)
    best_ratio = optimizer.max['params']['ratio']
    best_kappa = optimizer.max['params']['kappa']
    if best_ratio < 0.05:
        print(
            f'WARNING: Max distance set to a small ratio: {best_ratio}, this may mean unweighted means performed best in hyperparameter search and '
            f'that NaNs/mean values will be predicted for all inputs (barring polytomies).')
    if best_kappa < 0.05:
        print(
            f'WARNING: kappa set to small value: {best_kappa}, this may mean unweighted means performed best in hyperparameter search and '
            f'that distances are not useful in predictions.')

    return best_alpha, best_num_layers, best_sigma


def main():
    dataset1 = NewickDataset(
        newick_tree_path='../parsing_tree_data/unittest_data/binary/tree.tre',
        feature_csv_path_with_missing_target='../parsing_tree_data/unittest_data/binary/mcar_values.csv',
        ground_truth_csv_path='../parsing_tree_data/unittest_data/binary/ground_truth.csv',
        target_name='trait_BM_trend_scaled',
        binary_or_continuous='binary',

    )

    find_LP_hyperparameters(dataset1)


if __name__ == '__main__':
    main()
