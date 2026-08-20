# Phylogenetic Graph Neural Networks for Trait Predictions

These packages are currently under development.

This repository includes methods for reading phylogenetic tree data and applying Graph Neural Networks to predict properties of taxa in the tree.

## Installation

With pip, run:

`pip install git+https://github.com/alrichardbollans/phylotraitGNN.git`

## Example Usage

First load a dataset.
DistanceMatrixDataset is recommended, i.e. where there is an edge given between every species
and edge lengths are evolutionary distances. Can be generated in R with the following:

```R
tree_distances = ape::cophenetic.phylo(tree)
write.csv(tree_distances, file = 'tree_distances.csv')
```

```python
from phyloGNNy.parsing_tree_data import DistanceMatrixDataset

dataset = DistanceMatrixDataset(
    tree_distance_csv_path='tree_distances.csv',
    feature_csv_path_with_missing_target='features_with_a_target_column.csv',
    target_name='trait_to_predict',
    binary_or_continuous='binary'
)
```

The feature csv file specifies file with a table containing predictor features and a target variable to predict, that may have missing values to
predict
The index of the feature table and the distance table should contain names of tree tips
Examples of appropriate data can be found in: [binary examples](phyloGNNy/parsing_tree_data/unittest_data/binary)
and [continuous examples](phyloGNNy/parsing_tree_data/unittest_data/continuous).

Now to train a model:

```python
import torch
from phyloGNNy.GNN_models import GATv2Conv_node_classifier, train_gcn_model

model = GATv2Conv_node_classifier(dataset, 2, hidden_channels=4, attention_dropout=0.1,
                                  dropout=0.1)

loss_function = torch.nn.CrossEntropyLoss()  # specify appropriate loss function
optimizer = torch.optim.Adam(model.parameters())
epochs = 100

train_gcn_model(model, dataset.data, loss_function, optimizer, epochs, plot_loss=True)
```

To get model predictions:

```python
model_outputs = model(dataset.data.x, dataset.data.edge_index, edge_attr=dataset.data.edge_weight)

# Model outputs are logits, if you want probabilities for class 1 (in the binary case):
from phyloGNNy.GNN_models import convert_logits_to_probs

probs = convert_logits_to_probs(model_outputs)

# Model outputs are for all tips in the tree, and ordered by the ordering induced by the tree
# To convert these to a dataframe in the same ordering as the input feature table, use a helper function:

prediction_df = dataset.get_model_prediction_outputs_in_feature_order(probs)
```

## Licence

This work is licensed under a
[Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License][cc-by-nc-sa].

[![CC BY-NC-SA 4.0][cc-by-nc-sa-image]][cc-by-nc-sa]

[cc-by-nc-sa]: http://creativecommons.org/licenses/by-nc-sa/4.0/

[cc-by-nc-sa-image]: https://licensebuttons.net/l/by-nc-sa/4.0/88x31.png

[cc-by-nc-sa-shield]: https://img.shields.io/badge/License-CC%20BY--NC--SA%204.0-lightgrey.svg