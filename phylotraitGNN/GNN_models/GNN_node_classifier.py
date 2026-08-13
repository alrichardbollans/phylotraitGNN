import torch
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv

from phylotraitGNN.GNN_models import test_binary_GNN_outputs
from phylotraitGNN.parsing_tree_data import DistanceMatrixDataset


class GCN_node_classifier(torch.nn.Module):
    def __init__(self, dataset, hidden_channels, dropout_p):
        super().__init__()
        # Shaked Brody et al., ‘How Attentive Are Graph Attention Networks?’,
        # arXiv:2105.14491, preprint, arXiv, 31 January 2022, https://doi.org/10.48550/arXiv.2105.14491.
        # Note this adds self loops by default, the attention function applied to neighbours then includes the current node.
        self.conv1 = GATv2Conv(dataset.num_features, hidden_channels, edge_dim=1, dropout=dropout_p)
        self.conv2 = GATv2Conv(hidden_channels, dataset.num_classes, edge_dim=1, dropout=dropout_p)

    def forward(self, x, edge_index, edge_attr):
        x = self.conv1(x, edge_index, edge_attr=edge_attr)
        x = x.relu()
        x = self.conv2(x, edge_index, edge_attr=edge_attr) # There typically isn't a separate, fully-connected (linear) layer after the last GNN layer, because the final convolution is trained to map embeddings directly to class logits
        return x

    def train_step(self, data, optimizer, loss_function):
        self.train()
        optimizer.zero_grad()  # Clear gradients.
        out_ = self(data.x, data.edge_index, data.edge_weight)  # Perform a single forward pass.
        loss_ = loss_function(out_[data.train_mask], data.y[data.train_mask])  # Compute the loss solely based on the training nodes.
        loss_.backward()  # Derive gradients.
        optimizer.step()  # Update parameters based on gradients.
        return loss_

    def test(self, data):
        self.eval()
        out_ = self(data.x, data.edge_index, edge_attr=data.edge_weight)
        test_acc, b_score = test_binary_GNN_outputs(out_, data)

        return test_acc, b_score


def train_gcn_model(model, data, epochs,verbose=0):

    loss_function = torch.nn.CrossEntropyLoss()
    optimizer_class = torch.optim.Adam
    optimizer_kwargs = {'lr': 0.01, 'weight_decay': 5e-4}
    optimizer = optimizer_class(model.parameters(), **optimizer_kwargs)

    for epoch in range(1, epochs):
        loss = model.train_step(data, optimizer, loss_function)
        if verbose > 0:
            print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}')


def predict_node_classes(dataset,epochs, hidden_channels, dropout_p):
    model = GCN_node_classifier(dataset, hidden_channels=hidden_channels, dropout_p=dropout_p)
    data = dataset.data
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
        return out
    else:
        train_gcn_model(model, data, epochs)

        model.eval()  # Set model to evaluation mode.
        out_ = model(data.x, data.edge_index, edge_attr=data.edge_weight)
        probs = F.softmax(out_, dim=1)  # Convert logits to probabilities.
        assert (torch.all(probs >= 0.0) and torch.all(probs <= 1.0))

        # Add assertion that rows in pred proba add up to 1
        assert (probs.sum(dim=1).max().item() < 1.01)
        assert (probs.sum(dim=1).max().item() > 0.99)
    return probs

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
