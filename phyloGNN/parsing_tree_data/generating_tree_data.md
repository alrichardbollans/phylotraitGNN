To create the required files from a phylogenetic tree:

```R
tree_distances = ape::cophenetic.phylo(tree)
write.csv(tree_distances, file = file.path(dir_path, 'tree_distances.csv'))
```