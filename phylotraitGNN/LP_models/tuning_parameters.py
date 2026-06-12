from copy import deepcopy

import numpy as np
import torch

from phylotraitGNN.LP_models import propagate_labels, test_binary_LP_outputs
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
        ds_copy.data.test_mask = val_mask

        datasets.append(ds_copy)
    return datasets


def find_LP_hyperparameters(dataset: GenericPhyloDataset, verbose: int = 2, init_points=10, n_iter=50):
    '''
    Use bayesian optimization to find the best hyperparameters for label propagation.

    Bayesian Optimization is a method for finding global optima of black-box functions. In the case where the function is computationally expensive
    (as in doing cross-validation), Bayesian Optimization is useful as it selects new hyperparameter values to try based on previous ones,
     thus avoiding many unnecessary evaluations.
    '''

    original_sigma = dataset.data.original_edge_std
    assert dataset.binary_or_continuous == 'binary'

    # Create datasets for cross-validation
    cross_val_datasets = get_datasets_for_cross_validation(deepcopy(dataset), number_of_splits=5)

    def black_box_function(alpha, num_layers, sigma_ratio):
        """Function with unknown internals we wish to maximize.

        When NaN is returned for all values of cross_val_score, the current worst score is returned. This will break if the first try returns NaN.
        """

        cross_val_scores = []
        for crossval_dataset in cross_val_datasets:
            original_edge_weight = crossval_dataset.data.edge_weight
            # Reassign edge weights using sigma_ratio
            new_sigma = original_sigma * sigma_ratio
            new_edge_weight = crossval_dataset.data.edge_weight ** ((original_sigma**2)/(new_sigma**2))
            crossval_dataset.data.edge_weight = new_edge_weight
            probs = propagate_labels(crossval_dataset, num_layers=num_layers, alpha=alpha)
            test_acc, b_score = test_binary_LP_outputs(probs, crossval_dataset.data)
            # BayesianOptimization has no option to minimise, so we invert the brier score
            cross_val_scores.append(1 - b_score)

            # Reset edge weights
            crossval_dataset.data.edge_weight = original_edge_weight
        mean_cv_score = np.mean(cross_val_scores)

        return mean_cv_score

    # Bounded region of parameter space
    # Using categorical values for num_layers:
    # Eduardo C. Garrido-Merchán and Daniel Hernández-Lobato,(March 2020): 20–35, https://doi.org/10.1016/j.neucom.2019.11.004.
    pbounds = {'num_layers': (1, 50, int),
               'alpha': (0, 1.0),  #
               'sigma_ratio': (0.1, 10)}

    optimizer = BayesianOptimization(
        f=black_box_function,
        pbounds=pbounds,
        random_state=None,
        verbose=verbose
    )

    optimizer.maximize(init_points=init_points, n_iter=n_iter)

    print(optimizer.max)
    best_alpha = optimizer.max['params']['alpha']
    best_num_layers = optimizer.max['params']['num_layers']
    best_sigma_ratio = optimizer.max['params']['sigma_ratio']
    best_sigma = original_sigma * best_sigma_ratio
    if best_alpha < 0.05:
        print(
            f'WARNING: low value of alpha {best_alpha}. This indicates that not much information is being propagated between nodes.')
    if best_alpha > 0.98:
        print(
            f'WARNING: high value of alpha {best_alpha}. This indicates that initial label information is not retained in each step, '
            f'and all information is being translated between nodes. ')
    if best_sigma_ratio > 9:
        print(
            f'WARNING: high value of sigma {best_sigma}. This means most weights in the graph are close to 1 so all unlabelled points '
            f'are likely to have similar predictions.')

    return best_alpha, int(best_num_layers), float(best_sigma), best_sigma_ratio


def main():
    dataset1 = NewickDataset(
        newick_tree_path='../parsing_tree_data/unittest_data/binary/tree.tre',
        feature_csv_path_with_missing_target='../parsing_tree_data/unittest_data/binary/mcar_values.csv',
        ground_truth_csv_path='../parsing_tree_data/unittest_data/binary/ground_truth.csv',
        target_name='trait_BM_trend_scaled',
        binary_or_continuous='binary', add_self_loops=True,

    )

    find_LP_hyperparameters(dataset1)


if __name__ == '__main__':
    main()
