import torch
from torch_geometric.nn import LabelPropagation
from torch_geometric.transforms import AddSelfLoops, ToUndirected

from phylotraitGNN.LP_models import test_binary_LP_outputs
from phylotraitGNN.parsing_tree_data import GenericPhyloDataset, DistanceMatrixDataset


def my_post_step(out):
    # The aggregation step in Label Propagation sums over neighbouring nodes, which in some cases blows up the estimates (which means alpha loses its effectiveness).
    # This fixes that case while also fixing issues with the default clamping process. See: https://github.com/pyg-team/pytorch_geometric/issues/10627
    # This function won't inflate minimal evidence e.g. a node with [0.1,0.2] will stay the same.
    # But where  values sum>1 e.g. [0.66,1.32] will be standardised and get converted to [0.33,0.66]

    # where the sum of the values in a row is greater than 1, divide values in the row by the sum of the row.
    # out: Tensor, shape [num_nodes, num_classes]
    row_sums = out.sum(dim=1, keepdim=True) + 1e-9  # Avoid division by zero

    # Normalized output
    normalized_out = out / row_sums

    # Where mask is True, take normalized_out; otherwise keep original out
    mask = (row_sums > 1)
    clamped_out = torch.where(mask, normalized_out, out)
    return clamped_out


def normalize_outputs_for_testing(out_):
    row_sums = out_.sum(dim=1, keepdim=True)

    # Normalized output
    normalized_out = out_ / row_sums

    # where normalised out rows are nan, replace with 0.5
    mask = (row_sums == 0).squeeze()
    normalized_out[mask] = 0.5
    return normalized_out


def propagate_labels(dataset: GenericPhyloDataset, num_layers=3, alpha=0.9):
    """
    Propagates labels in a dataset using label propagation. This function applies a label
    propagation model to discrete binary labels in the provided dataset. It is particularly
    suitable for datasets with graph-like structures, where nodes are associated with
    labels and connections (edges) influence label propagation.

    The dataset should have no self-loops and edges should be symmetric (Zhou, 2003).

    Parameters:
    dataset : GenericPhyloDataset
        The dataset containing the data to which label propagation will be applied.
        The dataset must have binary labels, as label propagation is only compatible
        with discrete labels.

    num_layers : int, optional
        The number of layers in the label propagation model. Default is 3.

    alpha : float, optional
        During each iteration each point receives information from its neighbors,
        and also retains its initial information.
        The alpha parameter specifies the relative amount of the information from its neighbors
        and its initial label information
        Must be a float value between 0 and 1. Default is 0.9.

    Returns:
    Tensor
        The output resulting from label propagation, reflecting the updated labels
        after applying the propagation model.
    """
    assert dataset.binary_or_continuous == 'binary'  # label propagation only works for discrete labels
    data = dataset.data

    # Check data has self loops, following Zhu 2002
    with_self_loops = AddSelfLoops(fill_value=1)(data)
    with_self_loops = ToUndirected(reduce='mean')(with_self_loops)

    assert torch.equal(with_self_loops.edge_index, data.edge_index)

    model = LabelPropagation(num_layers=num_layers, alpha=alpha)
    # Where y values are all the same, instead of training the model, out should be a tensor with two columns, one for each class.
    # Values in the column denoting the class that appears in y should be 1, and all other values should be 0.
    if torch.unique(data.y).shape[0] == 1:
        out = torch.zeros((data.y.shape[0], 2))
        if torch.unique(data.y) == 0:
            # set first column of out to all 1s
            out[:, 0] = 1
        elif torch.unique(data.y) == 1:
            out[:, 1] = 1
        else:
            raise ValueError('y must contain only 0 or 1')
    else:
        out = model(data.y, data.edge_index, mask=data.train_mask, edge_weight=data.edge_weight, post_step=my_post_step)
    out = normalize_outputs_for_testing(out)
    return out


def main():
    dataset = DistanceMatrixDataset(
        tree_distance_csv_path='../parsing_tree_data/unittest_data/binary/tree_distances.csv',
        feature_csv_path_with_missing_target='../parsing_tree_data/unittest_data/binary/mcar_values.csv',
        ground_truth_csv_path='../parsing_tree_data/unittest_data/binary/ground_truth.csv',
        target_name='trait_BM_trend_scaled',
        binary_or_continuous='binary',
        k_nearest=50,  # Alternative: connect to 2 nearest neighbors
    )
    data = dataset.data
    print(data.original_max_edge_length)

    def evaluate_label_propagation(dataset: GenericPhyloDataset, num_layers=3, alpha=0.9):
        probs = propagate_labels(dataset, num_layers=num_layers, alpha=alpha)

        test_acc, b_score = test_binary_LP_outputs(probs, dataset.data)
        print(f'Test Accuracy: {test_acc:.4f}')
        print(f'Test Brier: {b_score:.4f}')

    # evaluate_label_propagation(dataset, num_layers=1)
    evaluate_label_propagation(dataset, alpha=0.2)
    evaluate_label_propagation(dataset, num_layers=10)
    evaluate_label_propagation(dataset, num_layers=50)
    evaluate_label_propagation(dataset, num_layers=100)


if __name__ == '__main__':
    main()
