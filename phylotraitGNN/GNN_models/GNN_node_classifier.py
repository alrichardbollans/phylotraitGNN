import torch
from sklearn.metrics import brier_score_loss
from torch_geometric.explain import Explainer, GNNExplainer
from torch_geometric.nn import GCNConv, GATConv, GATv2Conv, LabelPropagation
import torch.nn.functional as F
from torch_geometric.transforms import FeaturePropagation

from phylotraitGNN.GNN_models import propagate_labels, test_binary
from phylotraitGNN.parsing_tree_data import DistanceMatrixDataset, NewickDataset, GenericPhyloDataset


# from phylotraitGNN.parsing_tree_data.visualising import explaining


class GCN(torch.nn.Module):
    def __init__(self, dataset, hidden_channels):
        super().__init__()
        torch.manual_seed(1234567)
        # Shaked Brody et al., ‘How Attentive Are Graph Attention Networks?’,
        # arXiv:2105.14491, preprint, arXiv, 31 January 2022, https://doi.org/10.48550/arXiv.2105.14491.
        # Note this adds self loops by default, the attention function applied to neighbours then includes the current node.
        self.conv1 = GATv2Conv(dataset.num_features, hidden_channels, edge_dim=1)
        self.conv2 = GATv2Conv(hidden_channels, dataset.num_classes, edge_dim=1)

    def forward(self, x, edge_index, edge_attr):
        x = self.conv1(x, edge_index, edge_attr=edge_attr)
        x = x.relu()
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv2(x, edge_index, edge_attr=edge_attr)
        return x

    def train_step(self, data, optimizer, loss_function):
        self.train()
        optimizer.zero_grad()  # Clear gradients.
        out_ = self(data.x, data.edge_index, data.edge_attr)  # Perform a single forward pass.
        loss_ = loss_function(out_[data.train_mask], data.y[data.train_mask])  # Compute the loss solely based on the training nodes.
        loss_.backward()  # Derive gradients.
        optimizer.step()  # Update parameters based on gradients.
        return loss_

    def test(self, data):
        self.eval()
        out_ = self(data.x, data.edge_index, edge_attr=data.edge_attr)
        test_acc, b_score = test_binary(out_, data)

        return test_acc, b_score

def train_gcn_model(model, data):
    loss_function = torch.nn.CrossEntropyLoss()
    optimizer_class = torch.optim.Adam
    optimizer_kwargs = {'lr': 0.01, 'weight_decay': 5e-4}
    optimizer = optimizer_class(model.parameters(), **optimizer_kwargs)

    for epoch in range(1, 100):
        loss = model.train_step(data, optimizer, loss_function)
        print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}')

def main():
    dataset = DistanceMatrixDataset(
        tree_distance_csv_path='../parsing_tree_data/unittest_data/binary/tree_distances.csv',
        feature_csv_path_with_missing_target='../parsing_tree_data/unittest_data/binary/mcar_values.csv',
        ground_truth_csv_path='../parsing_tree_data/unittest_data/binary/ground_truth.csv',
        target_name='trait_BM_trend_scaled',
        binary_or_continuous='binary',
        k_nearest=50,  # Alternative: connect to 2 nearest neighbors

    )

    # dataset = DistanceMatrixDataset(
    #     tree_distance_csv_path='../parsing_tree_data/unittest_data/binary_no_features/tree_distances.csv',
    #     feature_csv_path_with_missing_target='../parsing_tree_data/unittest_data/binary_no_features/mcar_values.csv',
    #     ground_truth_csv_path='../parsing_tree_data/unittest_data/binary_no_features/ground_truth.csv',
    #     target_name='trait_ARD',
    #     binary_or_continuous='binary',
    #     k_nearest=50,  # Alternative: connect to 2 nearest neighbors
    #
    # )

    # dataset = NewickDataset(
    #     newick_tree_path='../parsing_tree_data/unittest_data/binary_no_features/tree.tre',
    #     feature_csv_path_with_missing_target='../parsing_tree_data/unittest_data/binary_no_features/mcar_values.csv',
    #     ground_truth_csv_path='../parsing_tree_data/unittest_data/binary_no_features/ground_truth.csv',
    #     target_name='trait_ARD',
    #     binary_or_continuous='binary',
    #
    # )
    data = dataset.data

    propagate_labels(dataset)
    # model = GCN(dataset=dataset, hidden_channels=16)
    # print(model)
    #
    #
    # missing_mask = torch.isnan(data.x)
    # assert data.x.ndim == 2
    # assert data.edge_index.ndim == 2
    # assert missing_mask.ndim == 2
    # print("data.x shape:", data.x.shape)
    # print("missing_mask shape:", missing_mask.shape)
    #
    # print(data.x[:10])  # Print first 5 rows of features
    # print(data.y[:10])
    #
    # print(data.train_mask.sum(), data.test_mask.sum())
    # print((data.train_mask & data.test_mask).sum())  # should be 0
    #
    # print(torch.unique(data.y))
    # print(data.y.dtype)
    #
    # model = GCN(dataset=dataset, hidden_channels=16)
    #
    # ## Variable for the USER
    # binary_or_continuous = 'binary'
    # loss_function = torch.nn.CrossEntropyLoss()
    # optimizer_class = torch.optim.Adam
    # optimizer_kwargs = {'lr': 0.01, 'weight_decay': 5e-4}
    # optimizer = optimizer_class(model.parameters(), **optimizer_kwargs)
    #
    # test_acc_, brier = model.test(data)
    # print(f'Test Accuracy: {test_acc_:.4f}')
    # print(f'Test brier: {brier:.4f}')
    #
    # for epoch in range(1, 1001):
    #     loss = model.train_step(data, optimizer, loss_function)
    #     print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}')
    #
    # test_acc_, brier = model.test(data)
    # print(f'Test Accuracy: {test_acc_:.4f}')
    # print(f'Test brier: {brier:.4f}')

    # explaining(model, data)


if __name__ == '__main__':
    main()
