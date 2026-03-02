import torch
from torch_geometric.nn import LabelPropagation
import torch.nn.functional as F
from torch_geometric.transforms import AddSelfLoops

from phylotraitGNN.GNN_models import test_binary
from phylotraitGNN.parsing_tree_data import GenericPhyloDataset, DistanceMatrixDataset

lprop_hparams = {
    'num_layers': (1, 50),
    'alpha': (0.1, 1.0),
}


def my_post_step(out):
    # The aggregation step in Label Propagation sums over neighbouring nodes, which in some cases blows up the estimates.
    # This fixes that case. This does however mean that as values are esimated for nodes, these are given more weight even
    # though they are not ground truth values (however this is counteracted by alpha).
    #
    outsum = out.sum(dim=1, keepdim=True)
    soft_out = out / (outsum + 1e-9)
    return soft_out


def propagate_labels(dataset: GenericPhyloDataset, num_layers=3, alpha=0.9):
    data = dataset.data
    data = AddSelfLoops(fill_value=1)(data)
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

    # evaluate_label_propagation(dataset, num_layers=1)
    evaluate_label_propagation(dataset)
    evaluate_label_propagation(dataset, num_layers=10)
    evaluate_label_propagation(dataset, num_layers=50)
    evaluate_label_propagation(dataset, num_layers=100)


if __name__ == '__main__':
    main()
