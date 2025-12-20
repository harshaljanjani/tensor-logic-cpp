# Graph Neural Network (Section 4.1, Table 1)
# This shows how GNN operations map to tensor logic equations.
# A complete GNN implementation is available in `include/gnn.hpp`

# Graph structure as Boolean tensor Neig(x, y)
# A small social network
Neig(Alice, Bob)
Neig(Bob, Alice)
Neig(Bob, Charlie)
Neig(Charlie, Bob)
Neig(Charlie, Diana)
Neig(Diana, Charlie)

# GNN message passing:
# Initialization:   Emb[n, 0, d] = X[n, d]
# MLP:              Z[n, l, d'] = relu(WP[l, d', d] Emb[n, l, d] + b)
# Aggregation:      Agg[n, l, d'] = Neig(n, n') Z[n', l, d']
# Update:           Emb[n, l+1, d] = relu(WAgg[l, d, d'] Agg + WSelf[l, d, d] Emb + b)
# Classification:   Y[n, c] = softmax(WOut[c, d] Emb[n, L, d])
# This joins the neighborhood relation with node features,
# effectively aggregating and transforming neighbor features for each node.

# Symbolic reachability using pure logic (no neural components):
Reachable(x, y) <- Neig(x, y)
Reachable(x, z) <- Reachable(x, y), Neig(y, z)

# Communities can be defined as:
# SameComponent(x, y) <- Reachable(x, y), Reachable(y, x)
