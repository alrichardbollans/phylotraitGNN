import torch
from sklearn.metrics import make_scorer, brier_score_loss
from torch_geometric.nn import LabelPropagation
import torch.nn.functional as F
from torch_geometric.transforms import AddSelfLoops

from phylotraitGNN.GNN_models import test_binary
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


def propagate_labels(dataset: GenericPhyloDataset, num_layers=3, alpha=0.9):
    data = dataset.data
    model = LabelPropagation(num_layers=num_layers, alpha=alpha)
    out = model(data.y, data.edge_index, mask=data.train_mask, edge_weight=data.edge_weight, post_step=my_post_step)
    return out


def evaluate_label_propagation(dataset: GenericPhyloDataset, num_layers=3, alpha=0.9):
    probs = propagate_labels(dataset, num_layers=num_layers, alpha=alpha)

    test_acc, b_score = test_binary(probs, dataset.data)
    print(f'Test Accuracy: {test_acc:.4f}')
    print(f'Test Brier: {b_score:.4f}')



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
    # evaluate_label_propagation(dataset, num_layers=1)
    evaluate_label_propagation(dataset, alpha=0.2)
    evaluate_label_propagation(dataset, num_layers=10)
    evaluate_label_propagation(dataset, num_layers=50)
    evaluate_label_propagation(dataset, num_layers=100)


if __name__ == '__main__':
    main()
