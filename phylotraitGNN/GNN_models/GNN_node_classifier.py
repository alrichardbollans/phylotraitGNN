import torch
import torch.nn.functional as F
import torch_geometric
from torch_geometric.nn import GATv2Conv, GCNConv, APPNP

from phylotraitGNN.GNN_models import test_binary_GNN_outputs
from phylotraitGNN.parsing_tree_data import DistanceMatrixDataset, NewickDataset


class MyGNNModels(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def train_step(self, data, optimizer, loss_function):
        self.train()
        optimizer.zero_grad()  # Clear gradients.
        out_ = self(data.x, data.edge_index, data.edge_weight)  # Perform a single forward pass.
        train_loss_ = loss_function(out_[data.train_mask], data.y[data.train_mask])  # Compute the loss solely based on the training nodes.
        train_loss_.backward()  # Derive gradients.
        optimizer.step()  # Update parameters based on gradients.

        if hasattr(data, 'val_mask'):
            self.eval()
            val_loss_ = loss_function(out_[data.val_mask], data.y[data.val_mask])
        else:
            val_loss_ = None
        return train_loss_, val_loss_


class GATv2Conv_node_classifier(MyGNNModels):

    # This only passes messages twice, so best for the distance matrix case.

    def __init__(self, dataset: DistanceMatrixDataset, hidden_channels, dropout_p):
        super().__init__()
        # Shaked Brody et al., ‘How Attentive Are Graph Attention Networks?’,
        # arXiv:2105.14491, preprint, arXiv, 31 January 2022, https://doi.org/10.48550/arXiv.2105.14491.
        # Note this adds self loops by default, the attention function applied to neighbours then includes the current node.
        # Here, the self loop weight is set to the maximum value of the edge weights.
        conv1 = GATv2Conv(dataset.num_features, hidden_channels, edge_dim=1, dropout=dropout_p, fill_value=dataset.self_loop_fill_value)

        conv2 = GATv2Conv(hidden_channels, dataset.num_classes, edge_dim=1, fill_value=dataset.self_loop_fill_value)

        self.layers = torch.nn.ModuleList([conv1, conv2])
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.to(device)
    def forward(self, x, edge_index, edge_attr):

        for l in self.layers:
            # For graph layers, we need to add the "edge_index" tensor as additional input
            # All PyTorch Geometric graph layer inherit the class "MessagePassing", hence
            # we can simply check the class type.
            if isinstance(l, torch_geometric.nn.MessagePassing):
                x = l(x, edge_index,
                      edge_attr=edge_attr)
                x.relu()
            else:
                raise ValueError(f"Unknown layer type: {type(l)}")

        return x

    def test(self, data):
        self.eval()
        out_ = self(data.x, data.edge_index, edge_attr=data.edge_weight)
        test_acc, b_score = test_binary_GNN_outputs(out_, data, data.test_mask)

        return test_acc, b_score


class APPNPNet_node_classifier(MyGNNModels):
    # https://arxiv.org/abs/1810.05997

    # Advantage of this is that it can propagate messages far in the network, which could be handy in the Newick case.

    def __init__(self, dataset: NewickDataset, hidden_channels, dropout_p: float, K: int, alpha: float, edge_dropout_p: float):
        super().__init__()
        self.lin1 = torch.nn.Linear(dataset.num_features, hidden_channels)
        self.lin2 = torch.nn.Linear(hidden_channels, dataset.num_classes)
        self.prop = APPNP(K=K, alpha=alpha, dropout=edge_dropout_p)
        self.dropout_p = dropout_p

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.to(device)
    def forward(self, x, edge_index, edge_weight=None):
        # Predictions are first generated from each node’s own features by a neural network and
        # then propagated using an adaptation of personalized PageRank
        x = F.dropout(x, p=self.dropout_p, training=self.training)
        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=self.dropout_p, training=self.training)
        x = self.lin2(x)  # predict
        x = self.prop(x, edge_index, edge_weight)  # then propagate
        return x

    def test(self, data):
        self.eval()
        out_ = self(data.x, data.edge_index, edge_weight=data.edge_weight)
        test_acc, b_score = test_binary_GNN_outputs(out_, data, data.test_mask)

        return test_acc, b_score


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
