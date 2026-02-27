To create the required files from a phylogenetic tree:

For distance matrices:
```R
tree_distances = ape::cophenetic.phylo(tree)
write.csv(tree_distances, file = file.path(dir_path, 'tree_distances.csv'))
```

For newick files:
```R
# Nodes should all be named 'Node_x' where x is the node number
paste("Node",1L:tree$Nnode, sep='_') -> tree$node.label
ape::write.tree(tree, 'tree.tre')
```