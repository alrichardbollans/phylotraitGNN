import torch
from matplotlib import pyplot as plt
from sklearn.manifold import TSNE
from torch_geometric.nn import GATv2Conv

from phylotraitGNN.GNN_models import train_gcn_model, EarlyStopping, test_regression_GNN_outputs
from phylotraitGNN.parsing_tree_data import NewickDataset


class GATv2Conv_node_regressor(torch.nn.Module):
    def __init__(self, dataset, hidden_channels, dropout_p):
        super().__init__()
        self.conv1 = GATv2Conv(dataset.num_features, hidden_channels, edge_dim=1, dropout=dropout_p)
        self.conv2 = GATv2Conv(hidden_channels, 1, edge_dim=1)  # This seems weird, need to check this for regression.

    def forward(self, x, edge_index, edge_attr):
        x = self.conv1(x, edge_index, edge_attr=edge_attr)
        x = x.relu()
        x = self.conv2(x, edge_index,
                       edge_attr=edge_attr)
        # x will have shape [num_nodes, 1], flattening:
        return x.squeeze(-1)  # [N, 1] -> [N]

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

    def test(self, data):
        self.eval()
        out_ = self(data.x, data.edge_index, edge_attr=data.edge_weight)
        _score = test_regression_GNN_outputs(out_, data, data.test_mask)

        return _score


def visualize(h, color):
    z = TSNE(n_components=2).fit_transform(h.detach().cpu().numpy())

    plt.figure(figsize=(10, 10))
    plt.xticks([])
    plt.yticks([])

    plt.scatter(z[:, 0], z[:, 1], s=70, c=color, cmap="Set2")
    plt.show()


def main():
    # dataset = DistanceMatrixDataset(
    #     tree_distance_csv_path='../parsing_tree_data/unittest_data/continuous/tree_distances.csv',
    #     feature_csv_path_with_missing_target='../parsing_tree_data/unittest_data/continuous/mcar_values.csv',
    #     ground_truth_csv_path='../parsing_tree_data/unittest_data/continuous/ground_truth.csv',
    #     target_name='trait_OU_scaled',
    #     binary_or_continuous='continuous'
    #
    # )

    dataset = NewickDataset(
        newick_tree_path='../parsing_tree_data/unittest_data/continuous/tree.tre',
        feature_csv_path_with_missing_target='../parsing_tree_data/unittest_data/continuous/mcar_values.csv',
        ground_truth_csv_path='../parsing_tree_data/unittest_data/continuous/ground_truth.csv',
        target_name='trait_OU_scaled',
        binary_or_continuous='continuous',
        validation_nodes=['t24', 't14', 't27']

    )

    model = GATv2Conv_node_regressor(dataset, hidden_channels=16, dropout_p=0.1)
    print(model)
    loss_function = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters())

    early_stopping = EarlyStopping(patience=5, delta=0.01)
    train_gcn_model(model, dataset.data, loss_function, optimizer, epochs=100, early_stopping=early_stopping, plot_loss=True, )
    # Check training has changed scores
    mae = model.test(dataset.data)


if __name__ == '__main__':
    main()
