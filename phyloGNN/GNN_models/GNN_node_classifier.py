import torch
import torch_geometric
from matplotlib import pyplot as plt
from sklearn.manifold import TSNE
from sklearn.metrics import brier_score_loss
from torch_geometric.nn import GCNConv
import torch.nn.functional as F
from torchmetrics.functional import mean_squared_error

from phyloGNN.parsing_tree_data import DistanceMatrixDataset

## Variable for the USER
binary_or_continuous = 'binary'
loss_function = torch.nn.CrossEntropyLoss()
optimizer_class = torch.optim.Adam
optimizer_kwargs = {'lr': 0.01, 'weight_decay': 5e-4}


class GCN(torch.nn.Module):
    def __init__(self, hidden_channels):
        super().__init__()
        torch.manual_seed(1234567)
        self.conv1 = GCNConv(dataset.num_features, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, dataset.num_classes)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = x.relu()
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv2(x, edge_index)
        return x


def train():
    model.train()
    optimizer.zero_grad()  # Clear gradients.
    out = model(data.x, data.edge_index)  # Perform a single forward pass.
    loss = loss_function(out[data.train_mask], data.y[data.train_mask])  # Compute the loss solely based on the training nodes.
    loss.backward()  # Derive gradients.
    optimizer.step()  # Update parameters based on gradients.
    return loss


def test():
    model.eval()
    out = model(data.x, data.edge_index)
    test_predictions = out[data.test_mask]
    pred = test_predictions.argmax(dim=1)  # Use the class with highest probability.
    test_correct = pred == data.y[data.test_mask]  # Check against ground-truth labels.
    test_acc = int(test_correct.sum()) / int(data.test_mask.sum())  # Derive ratio of correct predictions.

    pred_proba = test_predictions[:, 1]
    # This breaks as it seems like predictions are not probabilities.
    # b_score = brier_score_loss(data.y[data.test_mask].detach().numpy(), pred_proba.detach().numpy())
    return test_acc#, b_score


def visualize(h, color):
    z = TSNE(n_components=2).fit_transform(h.detach().cpu().numpy())

    plt.figure(figsize=(10, 10))
    plt.xticks([])
    plt.yticks([])

    plt.scatter(z[:, 0], z[:, 1], s=70, c=color, cmap="Set2")
    plt.show()


if __name__ == '__main__':
    dataset = DistanceMatrixDataset(
        tree_distance_csv_path='../parsing_tree_data/my_data/binary/tree_distances.csv',
        feature_csv_path_with_missing_target='../parsing_tree_data/my_data/binary/mcar_values.csv',
        ground_truth_csv_path='../parsing_tree_data/my_data/binary/ground_truth.csv',
        target_name='trait_BM_trend_scaled',
        binary_or_continuous='binary',
        k_nearest=50,  # Alternative: connect to 2 nearest neighbors
        transform=torch_geometric.transforms.NormalizeFeatures()

    )

    model = GCN(hidden_channels=16)
    print(model)

    model.eval()

    optimizer = optimizer_class(model.parameters(), **optimizer_kwargs)

    data = dataset.data
    out_ = model(data.x, data.edge_index)
    # visualize(out, color=data.y)
    test_acc_ = test()
    print(f'Test Accuracy: {test_acc_:.4f}')
    # print(f'Test brier: {brier:.4f}')

    for epoch in range(1, 101):
        loss = train()
        print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}')

    test_acc_ = test()
    print(f'Test Accuracy: {test_acc_:.4f}')
    # print(f'Test brier: {brier:.4f}')
