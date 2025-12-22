#include <algorithm>
#include <iomanip>
#include <iostream>

#include "embeddings.hpp"
#include "gnn.hpp"
#include "tensor_logic.hpp"
#include "transformer.hpp"

using namespace tl;

void example_temperature_reasoning() {
    std::cout << "\n";
    std::cout << "╔══════════════════════════════════════════════════════════════════╗\n";
    std::cout << "║           Example 1: Temperature-Controlled Reasoning            ║\n";
    std::cout << "╚══════════════════════════════════════════════════════════════════╝\n\n";

    std::cout << "The paper shows: σ(x, T) = 1 / (1 + e^(-x/T))\n";
    std::cout << "  - At T → 0: becomes step function (purely deductive)\n";
    std::cout << "  - At T > 0: soft threshold (analogical reasoning)\n\n";

    EmbeddedProgram prog(128, 0.0);

    prog.add_fact("CanFly", "Eagle", "True");
    prog.add_fact("CanFly", "Sparrow", "True");
    prog.add_fact("CanFly", "Penguin", "False");
    prog.add_fact("CanSwim", "Penguin", "True");
    prog.add_fact("CanSwim", "Duck", "True");
    prog.add_fact("IsBird", "Eagle", "True");
    prog.add_fact("IsBird", "Sparrow", "True");
    prog.add_fact("IsBird", "Penguin", "True");
    prog.add_fact("IsBird", "Duck", "True");

    TensorEquation rule;
    rule.lhs = TensorRef("IsAerial", {Index("x", true), Index("y", true)}, true);
    rule.rhs.push_back(TensorRef("CanFly", {Index("x", true), Index("y", true)}, true));
    rule.nonlinearity = Nonlinearity::Step;
    prog.add_rule(rule);

    prog.forward_chain();

    std::cout << "Knowledge base:\n";
    std::cout << "  CanFly(Eagle, True), CanFly(Sparrow, True), CanFly(Penguin, False)\n";
    std::cout << "  IsBird(Eagle/Sparrow/Penguin/Duck, True)\n\n";

    std::cout << "═══ T = 0 (Purely Deductive) ═══\n";
    std::cout << "At T = 0, the sigmoid becomes a step function.\n";
    std::cout << "Reasoning follows exact causal chains with NO hallucinations.\n\n";

    prog.set_temperature(0.0);

    std::cout << "Query: CanFly(Eagle, True)?\n";
    double result = prog.query("CanFly", "Eagle", "True");
    std::cout << "  Result: " << result << " (exact match: " << (result > 0.5 ? "YES" : "NO") << ")\n\n";

    std::cout << "Query: CanFly(Penguin, True)?  [NOT in knowledge base]\n";
    result = prog.query("CanFly", "Penguin", "True");
    std::cout << "  Result: " << result << " (exact match: " << (result > 0.5 ? "YES" : "NO") << ")\n";
    std::cout << "  → Correctly returns NO (deductive reasoning doesn't hallucinate)\n\n";

    std::cout << "Query: CanFly(Ostrich, True)?  [Unknown entity]\n";
    result = prog.query("CanFly", "Ostrich", "True");
    std::cout << "  Result: " << result << " (exact match: " << (result > 0.5 ? "YES" : "NO") << ")\n";
    std::cout << "  → Returns NO because Ostrich not in KB (no hallucination)\n\n";

    std::cout << "═══ T = 1.0 (Analogical Reasoning) ═══\n";
    std::cout << "At T>0, similar objects 'borrow' inferences from each other.\n";
    std::cout << "Weight is proportional to embedding similarity.\n\n";

    prog.set_temperature(1.0);

    auto& eagle_emb = prog.space.embeddings["Eagle"];
    std::vector<double> ostrich_emb = eagle_emb;
    for (size_t i = 0; i < ostrich_emb.size(); i += 4) {
        ostrich_emb[i] *= 0.9;
    }
    prog.space.set_embedding("Ostrich", ostrich_emb);

    double sim = prog.space.similarity("Eagle", "Ostrich");
    std::cout << "Embedding similarity(Eagle, Ostrich) = " << std::fixed << std::setprecision(3) << sim << "\n\n";

    std::cout << "Query: CanFly(Eagle, True)?\n";
    result = prog.query("CanFly", "Eagle", "True");
    std::cout << "  Result: " << std::fixed << std::setprecision(3) << result << "\n\n";

    std::cout << "Query: CanFly(Ostrich, True)?  [Similar to Eagle]\n";
    result = prog.query("CanFly", "Ostrich", "True");
    std::cout << "  Result: " << std::fixed << std::setprecision(3) << result << "\n";
    std::cout << "  → Non-zero because Ostrich borrows inference from similar Eagle!\n";
    std::cout << "  → This is analogical reasoning: 'Ostrich is like Eagle, Eagles fly.'\n\n";

    std::cout << "Key insight from paper:\n";
    std::cout << "  'Setting temperature T to 0 effectively reduces the Gram matrix\n";
    std::cout << "   to the identity matrix, making reasoning purely deductive.\n";
    std::cout << "   This contrasts with LLMs, which may hallucinate even at T = 0.'\n";
}

void example_deductive_chain() {
    std::cout << "\n";
    std::cout << "╔══════════════════════════════════════════════════════════════════╗\n";
    std::cout << "║            Example 2: Deductive Chain Tracing (T = 0)            ║\n";
    std::cout << "╚══════════════════════════════════════════════════════════════════╝\n\n";

    std::cout << "Demonstrating how forward chaining builds a deductive closure.\n";
    std::cout << "Each step is traceable and guaranteed to follow from the rules.\n\n";

    Program prog;

    prog.add_fact("Parent", {"Alice", "Bob"});
    prog.add_fact("Parent", {"Bob", "Charlie"});
    prog.add_fact("Parent", {"Charlie", "David"});
    prog.add_fact("Parent", {"David", "Eve"});

    TensorEquation rule1;
    rule1.lhs = TensorRef("Ancestor", {Index("x", true), Index("y", true)}, true);
    rule1.rhs.push_back(TensorRef("Parent", {Index("x", true), Index("y", true)}, true));
    rule1.nonlinearity = Nonlinearity::Step;
    prog.add_equation(rule1);

    TensorEquation rule2;
    rule2.lhs = TensorRef("Ancestor", {Index("x", true), Index("z", true)}, true);
    rule2.rhs.push_back(TensorRef("Ancestor", {Index("x", true), Index("y", true)}, true));
    rule2.rhs.push_back(TensorRef("Parent", {Index("y", true), Index("z", true)}, true));
    rule2.nonlinearity = Nonlinearity::Step;
    prog.add_equation(rule2);

    std::cout << "Facts:\n";
    std::cout << "  Parent(Alice, Bob)\n";
    std::cout << "  Parent(Bob, Charlie)\n";
    std::cout << "  Parent(Charlie, David)\n";
    std::cout << "  Parent(David, Eve)\n\n";

    std::cout << "Rules (with implicit step function, T = 0):\n";
    std::cout << "  Ancestor(x, y) <- Parent(x, y)                    [Base case]\n";
    std::cout << "  Ancestor(x, z) <- Ancestor(x, y), Parent(y, z)    [Transitive]\n\n";

    std::cout << "Forward chaining iterations:\n";
    std::cout << "────────────────────────────\n";

    for (int iter = 1; iter <= 4; ++iter) {
        size_t before = 0;
        auto it = prog.bool_tensors.find("Ancestor");
        if (it != prog.bool_tensors.end()) {
            before = it->second.size();
        }

        prog.forward_chain(1);

        it = prog.bool_tensors.find("Ancestor");
        size_t after = it != prog.bool_tensors.end() ? it->second.size() : 0;

        std::cout << "Iteration " << iter << ": " << before << " → " << after << " ancestor tuples\n";

        if (after == before) {
            std::cout << "  Fixpoint reached!\n";
            break;
        }
    }

    std::cout << "\nFinal deductive closure (all derivable facts):\n";
    auto results = prog.query("Ancestor");
    for (const auto& t : results) {
        std::cout << "  Ancestor" << t.to_string() << "\n";
    }

    std::cout << "\nDeductive chain for Ancestor(Alice, Eve):\n";
    std::cout << "  1. Parent(Alice, Bob)                  [Fact]\n";
    std::cout << "  2. Ancestor(Alice, Bob)                [Rule 1 from step 1]\n";
    std::cout << "  3. Parent(Bob, Charlie)                [Fact]\n";
    std::cout << "  4. Ancestor(Alice, Charlie)            [Rule 2 from steps 2,3]\n";
    std::cout << "  5. Parent(Charlie, David)              [Fact]\n";
    std::cout << "  6. Ancestor(Alice, David)              [Rule 2 from steps 4,5]\n";
    std::cout << "  7. Parent(David, Eve)                  [Fact]\n";
    std::cout << "  8. Ancestor(Alice, Eve)                [Rule 2 from steps 6,7] ✓\n";

    std::cout << "\nEvery derived fact has a proof trace. No hallucinations possible.\n";
}

void example_transformer() {
    std::cout << "\n";
    std::cout << "╔══════════════════════════════════════════════════════════════════╗\n";
    std::cout << "║             Example 3: Transformer in Tensor Logic               ║\n";
    std::cout << "╚══════════════════════════════════════════════════════════════════╝\n\n";

    TransformerConfig config;
    config.vocab_size = 50;
    config.max_seq_len = 16;
    config.d_model = 32;
    config.n_heads = 2;
    config.d_k = 8;
    config.d_v = 8;
    config.n_blocks = 1;
    config.d_ff = 64;

    Transformer transformer(config);

    std::cout << "Transformer Configuration:\n";
    std::cout << "  vocab_size = " << config.vocab_size << "\n";
    std::cout << "  d_model = " << config.d_model << "\n";
    std::cout << "  n_heads = " << config.n_heads << "\n";
    std::cout << "  d_k = " << config.d_k << " (key/query dim per head)\n";
    std::cout << "  d_v = " << config.d_v << " (value dim per head)\n";
    std::cout << "  n_blocks = " << config.n_blocks << "\n\n";

    std::cout << "Tensor Logic Representation:\n";
    std::cout << "────────────────────────────\n";
    std::cout << transformer.to_tensor_logic() << "\n";

    Tokenizer tokenizer;
    std::string text = "the cat sat on the mat";
    auto tokens = tokenizer.tokenize(text);

    std::cout << "Input text: \"" << text << "\"\n";
    std::cout << "Tokens: [";
    for (size_t i = 0; i < tokens.size(); ++i) {
        if (i > 0) std::cout << ", ";
        std::cout << tokens[i];
    }
    std::cout << "]\n\n";

    std::cout << "Forward pass through transformer.\n";
    auto logits = transformer.forward(tokens);

    std::cout << "Output shape: [" << logits.size() << " positions × " << logits[0].size() << " vocab]\n\n";

    std::cout << "Next token probabilities for last position:\n";
    std::cout << "  (showing top 5)\n";

    std::vector<std::pair<size_t, double>> top_tokens;
    for (size_t t = 0; t < logits.back().size(); ++t) {
        top_tokens.emplace_back(t, logits.back()[t]);
    }
    std::sort(top_tokens.begin(), top_tokens.end(), [](const auto& a, const auto& b) { return a.second > b.second; });

    for (int i = 0; i < 5 && i < static_cast<int>(top_tokens.size()); ++i) {
        std::cout << "  Token " << top_tokens[i].first << ": " << std::fixed << std::setprecision(4)
                  << top_tokens[i].second << "\n";
    }

    std::cout << "\nKey insight: The attention mechanism is just tensor operations:\n";
    std::cout << "  Comp[p, p'] = softmax(Query[p, dk] Key[p', dk] / sqrt(dk))\n";
    std::cout << "  Attn[p, dv] = Comp[p, p'] Val[p', dv]\n";
}

void example_gnn() {
    std::cout << "\n";
    std::cout << "╔══════════════════════════════════════════════════════════════════╗\n";
    std::cout << "║                 Example 4: Graph Neural Network                  ║\n";
    std::cout << "╚══════════════════════════════════════════════════════════════════╝\n\n";

    std::cout << "Table 1 - GNN in Tensor Logic:\n";
    std::cout << "  Graph structure:  Neig(x, y)\n";
    std::cout << "  Initialization:   Emb[n, 0, d] = X[n, d]\n";
    std::cout << "  MLP:              Z[n, l, d'] = relu(WP[l, d', d] Emb[n, l, d])\n";
    std::cout << "  Aggregation:      Agg[n, l, d'] = Neig(n, n') Z[n', l, d']\n";
    std::cout << "  Update:           Emb[n, l+1, d] = relu(WAgg Agg + WSelf Emb)\n";
    std::cout << "  Node class:       Y[n] = softmax(WOut[c, d] Emb[n, L, d])\n\n";

    // Create GNN configuration
    GNNConfig config;
    config.d_node = 16;
    config.d_hidden = 32;
    config.n_layers = 3;
    config.n_classes = 2;
    config.use_self_loops = true;

    GNN gnn(config);

    std::cout << gnn.to_tensor_logic() << "\n";

    std::cout << "Graph structure (Neig relation):\n";
    std::cout << "      B\n";
    std::cout << "     / \\\n";
    std::cout << "    A---C    D---A\n\n";

    // Initialize nodes with random features
    std::mt19937 feature_rng(123);
    std::normal_distribution<double> feature_dist(0.0, 1.0);

    auto random_features = [&]() {
        std::vector<double> features(config.d_node);
        for (auto& f : features) f = feature_dist(feature_rng);
        return features;
    };

    gnn.add_node("A", random_features());
    gnn.add_node("B", random_features());
    gnn.add_node("C", random_features());
    gnn.add_node("D", random_features());

    // Build graph structure
    gnn.add_edge("A", "B");
    gnn.add_edge("B", "A");
    gnn.add_edge("B", "C");
    gnn.add_edge("C", "B");
    gnn.add_edge("C", "A");
    gnn.add_edge("A", "C");
    gnn.add_edge("A", "D");
    gnn.add_edge("D", "A");

    std::cout << "Message passing as tensor operation:\n";
    std::cout << "  Agg[n, l, d'] = Neig(n, n') Z[n', l, d']\n\n";

    std::cout << "For node A, neighbors are: B, C, D\n";
    std::cout << "  Agg[A, d'] = Z[B, d'] + Z[C, d'] + Z[D, d']\n\n";

    std::cout << "For node D, only neighbor is: A\n";
    std::cout << "  Agg[D, d'] = Z[A, d']\n\n";

    std::cout << "Running " << config.n_layers << " layers of message passing...\n\n";

    // Perform message passing
    gnn.forward();

    // Get node classifications
    auto classifications = gnn.classify_all();

    std::cout << "Node classifications after " << config.n_layers << " message passing layers:\n";
    for (const auto& [node, probs] : classifications) {
        std::cout << "  Node " << node << ": [";
        for (size_t c = 0; c < probs.size(); ++c) {
            if (c > 0) std::cout << ", ";
            std::cout << "class_" << c << "=" << std::fixed << std::setprecision(3) << probs[c];
        }
        std::cout << "]\n";
    }

    std::cout << "\nFinal node embeddings Emb[n, L=" << config.n_layers << ", d=" << config.d_node << "]:\n";
    std::vector<std::string> node_names = {"A", "B", "C", "D"};
    for (const std::string& node : node_names) {
        const auto& emb = gnn.get_embedding(node);
        std::cout << "  Node " << node << ": [";
        for (size_t i = 0; i < std::min(size_t(5), emb.size()); ++i) {
            if (i > 0) std::cout << ", ";
            std::cout << std::fixed << std::setprecision(3) << emb[i];
        }
        if (emb.size() > 5) std::cout << ", ...";
        std::cout << "]\n";
    }

    // Also demonstrate symbolic reachability
    std::cout << "\n═══ Symbolic Reachability (Deductive Reasoning) ═══\n\n";

    Program prog;
    prog.add_fact("Neig", {"A", "B"});
    prog.add_fact("Neig", {"B", "A"});
    prog.add_fact("Neig", {"B", "C"});
    prog.add_fact("Neig", {"C", "B"});
    prog.add_fact("Neig", {"C", "A"});
    prog.add_fact("Neig", {"A", "C"});
    prog.add_fact("Neig", {"A", "D"});
    prog.add_fact("Neig", {"D", "A"});

    TensorEquation reach1;
    reach1.lhs = TensorRef("Reach", {Index("x", true), Index("y", true)}, true);
    reach1.rhs.push_back(TensorRef("Neig", {Index("x", true), Index("y", true)}, true));
    reach1.nonlinearity = Nonlinearity::Step;
    prog.add_equation(reach1);

    TensorEquation reach2;
    reach2.lhs = TensorRef("Reach", {Index("x", true), Index("z", true)}, true);
    reach2.rhs.push_back(TensorRef("Reach", {Index("x", true), Index("y", true)}, true));
    reach2.rhs.push_back(TensorRef("Neig", {Index("y", true), Index("z", true)}, true));
    reach2.nonlinearity = Nonlinearity::Step;
    prog.add_equation(reach2);

    prog.forward_chain();

    std::cout << "Symbolic reachability (computed via forward chaining):\n";
    auto reach = prog.query("Reach");
    std::cout << "  Can reach from any node to any other: " << reach.size() << " pairs\n";

    bool d_to_c = false;
    for (const auto& t : reach) {
        if (std::get<std::string>(t.values[0]) == "D" && std::get<std::string>(t.values[1]) == "C") {
            d_to_c = true;
            break;
        }
    }
    std::cout << "  Reach(D, C)? " << (d_to_c ? "Yes" : "No") << " (via D->A->C)\n";

    std::cout << "\nKey insight: GNNs combine neural (embeddings, weights) and symbolic\n";
    std::cout << "(graph structure) reasoning in a unified tensor logic framework.\n";
}

void example_kernel_machine() {
    std::cout << "\n";
    std::cout << "╔══════════════════════════════════════════════════════════════════╗\n";
    std::cout << "║                   Example 5: Kernel Machine                      ║\n";
    std::cout << "╚══════════════════════════════════════════════════════════════════╝\n\n";

    std::cout << "From the paper:\n";
    std::cout << "  Y[Q] = f(A[i] Y[i] K[Q, i] + B)\n";
    std::cout << "where Q is query, i ranges over support vectors, K is kernel.\n\n";

    std::cout << "Polynomial kernel: K[i, i'] = (X[i, j] X[i', j])^n\n";
    std::cout << "Gaussian kernel:   K[i, i'] = exp(-(X[i, j] - X[i', j])^2 / Var)\n\n";

    std::vector<std::vector<double>> support_vectors = {
        {1.0, 1.0},    // class +1
        {1.5, 1.2},    // class +1
        {-1.0, -1.0},  // class -1
        {-0.8, -1.2}   // class -1
    };
    std::vector<double> alphas = {0.5, 0.5, -0.5, -0.5};
    double bias = 0.0;
    // RBF
    double gamma = 0.5;

    std::vector<double> query = {0.5, 0.5};

    std::cout << "Support vectors:\n";
    for (size_t i = 0; i < support_vectors.size(); ++i) {
        std::cout << "  SV[" << i << "] = (" << support_vectors[i][0] << ", " << support_vectors[i][1]
                  << "), α = " << alphas[i] << "\n";
    }
    std::cout << "\nQuery point: (" << query[0] << ", " << query[1] << ")\n\n";

    // K[Q, i] = exp(-gamma * ||Q - X[i]||^2)
    // Y[Q] = Σ_i α[i] K[Q, i] + bias
    std::cout << "RBF Kernel computation (tensor logic style):\n";
    std::cout << "  K[Q, i] = exp(-γ * (X[Q, j] - X[i, j])²)\n\n";

    double prediction = bias;
    for (size_t i = 0; i < support_vectors.size(); ++i) {
        double dist_sq = 0.0;
        for (size_t j = 0; j < query.size(); ++j) {
            double diff = query[j] - support_vectors[i][j];
            dist_sq += diff * diff;
        }
        double kernel_val = std::exp(-gamma * dist_sq);
        double contribution = alphas[i] * kernel_val;
        prediction += contribution;

        std::cout << "  K[Q, " << i << "] = exp(-" << gamma << " * " << std::fixed << std::setprecision(2) << dist_sq
                  << ") = " << std::setprecision(4) << kernel_val << " → α*K = " << contribution << "\n";
    }

    std::cout << "\nPrediction Y[Q] = " << std::setprecision(4) << prediction;
    std::cout << " → Class: " << (prediction > 0 ? "+1" : "-1") << "\n";
}

void example_embedding_relations() {
    std::cout << "\n";
    std::cout << "╔══════════════════════════════════════════════════════════════════╗\n";
    std::cout << "║                 Example 6: Embedding Relations                   ║\n";
    std::cout << "╚══════════════════════════════════════════════════════════════════╝\n\n";
    std::cout << "From the paper:\n";
    std::cout << "  EmbR[i, j] = R(x, y) Emb[x, i] Emb[y, j]\n";
    std::cout << "  D[A, B] = EmbR[i, j] Emb[A, i] Emb[B, j]  ≈ 1 if (A,B) ∈ R\n\n";

    EmbeddingSpace space(64, 0.0, 123);

    SparseBoolTensor likes("Likes", {"x", "y"});
    likes.add_tuple({"Alice", "Pizza"});
    likes.add_tuple({"Alice", "Sushi"});
    likes.add_tuple({"Bob", "Pizza"});
    likes.add_tuple({"Carol", "Tacos"});

    std::cout << "Relation Likes:\n";
    for (const auto& t : likes.tuples) {
        std::cout << "  Likes" << t.to_string() << "\n";
    }
    std::cout << "\n";

    auto emb_likes = space.embed_relation(likes);

    std::cout << "Embedded as tensor EmbLikes[i, j] of shape (64, 64)\n\n";

    std::cout << "Querying embedded relation:\n";

    auto test_query = [&](const std::string& x, const std::string& y, bool expected) {
        double result = space.query_relation(emb_likes, x, y);
        std::cout << "  D[" << x << ", " << y << "] = " << std::fixed << std::setprecision(3) << result
                  << "  (expected: " << (expected ? "~1" : "~0") << ")"
                  << (std::abs(result - (expected ? 1.0 : 0.0)) < 0.5 ? " ✓" : " ?") << "\n";
    };

    test_query("Alice", "Pizza", true);
    test_query("Alice", "Sushi", true);
    test_query("Bob", "Pizza", true);
    test_query("Carol", "Tacos", true);
    test_query("Alice", "Tacos", false);
    test_query("Bob", "Sushi", false);

    std::cout << "\nKey insight: Dot products of random unit vectors ≈ 0,\n";
    std::cout << "so querying non-existent tuples returns ~0.\n";
    std::cout << "Error probability decreases with embedding dimension.\n";
}

void example_tucker_decomposition() {
    std::cout << "\n";
    std::cout << "╔══════════════════════════════════════════════════════════════════╗\n";
    std::cout << "║                 Example 7: Tucker Decomposition                  ║\n";
    std::cout << "╚══════════════════════════════════════════════════════════════════╝\n\n";

    std::cout << "From the paper:\n";
    std::cout << "  A[i,j,k] = M[i,p] M'[j,q] M''[k,r] C[p,q,r]\n\n";

    std::cout << "Tucker decomposition converts sparse tensors to dense operations:\n";
    std::cout << "  - Core tensor C captures the essential structure\n";
    std::cout << "  - Factor matrices M, M', M'' map indices to embedding space\n";
    std::cout << "  - Exponentially more efficient than operating on sparse tensors\n\n";

    size_t orig_dim = 100;
    size_t core_dim = 10;

    std::cout << "Example scaling:\n";
    std::cout << "  Original tensor: " << orig_dim << "³ = " << (orig_dim * orig_dim * orig_dim) << " elements\n";
    std::cout << "  Core tensor:     " << core_dim << "³ = " << (core_dim * core_dim * core_dim) << " elements\n";
    std::cout << "  Factor matrices: 3 × " << orig_dim << " × " << core_dim << " = " << (3 * orig_dim * core_dim)
              << " elements\n";
    std::cout << "  Total compressed: " << (core_dim * core_dim * core_dim + 3 * orig_dim * core_dim) << " elements\n";
    std::cout << "  Compression ratio: " << std::fixed << std::setprecision(1)
              << (static_cast<double>(orig_dim * orig_dim * orig_dim) /
                  (core_dim * core_dim * core_dim + 3 * orig_dim * core_dim))
              << "×\n\n";

    std::cout << "In tensor logic, this is written as a single equation:\n";
    std::cout << "  A[i, j, k] = M[i, p] M'[j, q] M''[k, r] C[p, q, r]\n\n";

    std::cout << "Key insight: 'Scaling up via Tucker decompositions has the\n";
    std::cout << "significant advantage that it combines seamlessly with the\n";
    std::cout << "learning and reasoning algorithms.' - Section 6\n";
}

int main(int argc, char* argv[]) {
    std::cout << "╔══════════════════════════════════════════════════════════════════╗\n";
    std::cout << "║              Tensor Logic - Examples from the Paper              ║\n";
    std::cout << "║                    Pedro Domingos (2025)                         ║\n";
    std::cout << "╚══════════════════════════════════════════════════════════════════╝\n";

    if (argc > 1) {
        std::string arg = argv[1];
        if (arg == "temp" || arg == "1")
            example_temperature_reasoning();
        else if (arg == "deductive" || arg == "2")
            example_deductive_chain();
        else if (arg == "transformer" || arg == "3")
            example_transformer();
        else if (arg == "gnn" || arg == "4")
            example_gnn();
        else if (arg == "kernel" || arg == "5")
            example_kernel_machine();
        else if (arg == "embed" || arg == "6")
            example_embedding_relations();
        else if (arg == "tucker" || arg == "7")
            example_tucker_decomposition();
        else {
            std::cout << "\nUsage: " << argv[0] << " [example]\n";
            std::cout << "Examples: temp, deductive, transformer, gnn, kernel, embed, tucker\n";
            std::cout << "Or use numbers 1-7\n";
        }
    } else {
        example_temperature_reasoning();
        example_deductive_chain();
        example_transformer();
        example_gnn();
        example_kernel_machine();
        example_embedding_relations();
        example_tucker_decomposition();
    }

    return 0;
}
