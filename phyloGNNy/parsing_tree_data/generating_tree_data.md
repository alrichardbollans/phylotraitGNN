To create the required files from a phylogenetic tree:

For distance matrices:

```R
tree_distances = ape::cophenetic.phylo(tree)
write.csv(tree_distances, file = file.path(dir_path, 'tree_distances.csv'))
```

For newick files:

```R
# Nodes should all be named. 
paste("Node",1L:tree$Nnode, sep='_') -> tree$node.label
ape::write.tree(tree, 'tree.tre')
```

For feature tables, these should be a csv file with:

- an index that contains names of tips (and possibly nodes) in the tree.
- Missing values for the target variable are used to determine test cases, and everything else is training (unless validation nodes are specified)

Note that the models generate predictions for all the nodes and tips in a tree, and will output
predictions using the ordering induced by the tree. To get predictions using the ordering from the input feature table, use the
`get_model_prediction_outputs_in_feature_order` method. 