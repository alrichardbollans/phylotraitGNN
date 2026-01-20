import torch
import torch_geometric
from matplotlib import pyplot as plt
from sklearn.manifold import TSNE
from sklearn.metrics import brier_score_loss
from torch_geometric.explain import Explainer, GNNExplainer
from torch_geometric.nn import GCNConv
import torch.nn.functional as F

from phyloGNN.parsing_tree_data import DistanceMatrixDataset
from phyloGNN.parsing_tree_data.visualising import explaining

## Variable for the USER
binary_or_continuous = 'binary'
loss_function = torch.nn.CrossEntropyLoss()
optimizer_class = torch.optim.Adam
optimizer_kwargs = {'lr': 0.01, 'weight_decay': 5e-4}


class GCN(torch.nn.Module):
    def __init__(self, dataset, hidden_channels):
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


def main():
    dataset = DistanceMatrixDataset(
        tree_distance_csv_path='../parsing_tree_data/my_data/binary/tree_distances.csv',
        feature_csv_path_with_missing_target='../parsing_tree_data/my_data/binary/mcar_values.csv',
        ground_truth_csv_path='../parsing_tree_data/my_data/binary/ground_truth.csv',
        target_name='trait_BM_trend_scaled',
        binary_or_continuous='binary',
        k_nearest=50,  # Alternative: connect to 2 nearest neighbors
        # transform=torch_geometric.transforms.NormalizeFeatures()

    )

    model = GCN(dataset=dataset, hidden_channels=16)
    model.eval()
    print(model)

    data = dataset.data

    print(data.x[:10])  # Print first 5 rows of features
    print(data.y[:10])

    print(data.train_mask.sum(), data.test_mask.sum())
    print((data.train_mask & data.test_mask).sum())  # should be 0

    print(torch.unique(data.y))
    print(data.y.dtype)

    model = GCN(dataset=dataset, hidden_channels=16)
    optimizer = optimizer_class(model.parameters(), **optimizer_kwargs)

    def train():
        model.train()
        optimizer.zero_grad()  # Clear gradients.
        out_ = model(data.x, data.edge_index)  # Perform a single forward pass.
        loss_ = loss_function(out_[data.train_mask], data.y[data.train_mask])  # Compute the loss solely based on the training nodes.
        loss_.backward()  # Derive gradients.
        optimizer.step()  # Update parameters based on gradients.
        return loss_

    def test():
        model.eval()
        out_ = model(data.x, data.edge_index)
        test_predictions = out_[data.test_mask]
        probs = F.softmax(test_predictions, dim=1)  # Convert logits to probabilities.
        pred_proba = probs[:, 1]  # Probability for class 1

        pred = probs.argmax(dim=1)  # Use the class with highest probability.
        test_correct = pred == data.y[data.test_mask]  # Check against ground-truth labels.
        test_acc = int(test_correct.sum()) / int(data.test_mask.sum())  # Derive ratio of correct predictions.

        # To use brier_score_loss:
        b_score = brier_score_loss(
            data.y[data.test_mask].detach().cpu().numpy(),
            pred_proba.detach().cpu().numpy()
        )

        return test_acc, b_score

    test_acc_, brier = test()
    print(f'Test Accuracy: {test_acc_:.4f}')
    print(f'Test brier: {brier:.4f}')

    for epoch in range(1, 1001):
        loss = train()
        print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}')

    test_acc_, brier = test()
    print(f'Test Accuracy: {test_acc_:.4f}')
    print(f'Test brier: {brier:.4f}')

    explaining(model, data)


if __name__ == '__main__':
    main()
