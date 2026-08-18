import torch
import torch_geometric
from matplotlib import pyplot as plt
from sklearn.manifold import TSNE
from sklearn.metrics import mean_absolute_error
from torch_geometric.nn import GATv2Conv
import torch.nn.functional as F


from phylotraitGNN.GNN_models import train_gcn_model, EarlyStopping, test_regression_GNN_outputs, MyGNNModels
from phylotraitGNN.parsing_tree_data import NewickDataset, DistanceMatrixDataset


class GATv2Conv_node_regressor(MyGNNModels):
    def __init__(self, dataset: DistanceMatrixDataset, hidden_channels:int, dropout_p:float):
        super().__init__()
        self.conv1 = GATv2Conv(dataset.num_features, 2*hidden_channels, edge_dim=1, dropout=dropout_p, fill_value=dataset.self_loop_fill_value)
        self.conv2 = GATv2Conv(2*hidden_channels, hidden_channels, edge_dim=1, dropout=dropout_p, fill_value=dataset.self_loop_fill_value)
        self.lin1 = torch.nn.Linear(hidden_channels, 1) # Plain linear layer for regression.
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.to(device)

    def forward(self, x, edge_index, edge_attr):
        x = self.conv1(x, edge_index,
                      edge_attr=edge_attr)
        x = F.relu(x)
        x = self.conv2(x, edge_index,
                      edge_attr=edge_attr)
        x = F.relu(x)
        x = self.lin1(x)  # predict

        # x will have shape [num_nodes, 1], flattening:
        return x.squeeze(-1)  # [N, 1] -> [N]

    def test(self, data, scorer: callable):
        self.eval()
        with torch.no_grad():
            out_ = self(data.x, data.edge_index, edge_attr=data.edge_weight)
            _score = test_regression_GNN_outputs(out_, data, data.test_mask, scorer)

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
    loss_function = torch.nn.L1Loss()
    optimizer = torch.optim.Adam(model.parameters())

    early_stopping = EarlyStopping(patience=5, delta=0.01)
    train_gcn_model(model, dataset.data, loss_function, optimizer, epochs=100, early_stopping=early_stopping, plot_loss=True, )
    # Check training has changed scores
    mae = model.test(dataset.data, mean_absolute_error)


if __name__ == '__main__':
    main()
