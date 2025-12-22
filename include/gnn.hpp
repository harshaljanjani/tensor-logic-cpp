#ifndef TENSOR_LOGIC_GNN_HPP
#define TENSOR_LOGIC_GNN_HPP

#include <algorithm>
#include <cmath>
#include <random>
#include <unordered_map>
#include <vector>

#include "tensor_logic.hpp"

namespace tl {

struct GNNConfig {
    size_t d_node = 64;          // node embedding dimension
    size_t d_hidden = 128;       // hidden layer dimension in MLP
    size_t n_layers = 3;         // number of message passing layers
    size_t n_classes = 2;        // number of output classes
    bool use_self_loops = true;  // add self-loops to graph
};

class GNN {
   public:
    GNNConfig config;

    // Emb[node][layer][dim]
    std::unordered_map<std::string, std::vector<std::vector<double>>> node_embeddings;

    // neighbors[node] = list of neighbor nodes
    std::unordered_map<std::string, std::vector<std::string>> adjacency;

    // WP[l]: `d_hidden × d_node`
    std::vector<DenseTensor> W_preprocess;

    // WAgg[l]: `d_node × d_hidden`
    std::vector<DenseTensor> W_aggregate;

    // WSelf[l]: `d_node × d_node`
    std::vector<DenseTensor> W_self;

    // WOut: `n_classes × d_node`
    DenseTensor W_out;

    std::vector<std::vector<double>> b_preprocess;
    std::vector<std::vector<double>> b_aggregate;
    std::vector<double> b_out;

    std::mt19937 rng;

    GNN(const GNNConfig& cfg = GNNConfig(), unsigned seed = 42) : config(cfg), rng(seed) {
        initialize_weights();
    }

    void initialize_weights() {
        W_preprocess.resize(config.n_layers);
        W_aggregate.resize(config.n_layers);
        W_self.resize(config.n_layers);
        b_preprocess.resize(config.n_layers);
        b_aggregate.resize(config.n_layers);

        for (size_t l = 0; l < config.n_layers; ++l) {
            // WP[l, d', d]
            double std_p = std::sqrt(2.0 / (config.d_node + config.d_hidden));
            W_preprocess[l] = DenseTensor("WP_" + std::to_string(l), {"d'", "d"}, {config.d_hidden, config.d_node});
            std::normal_distribution<double> dist_p(0.0, std_p);
            for (auto& v : W_preprocess[l].data) v = dist_p(rng);

            b_preprocess[l].resize(config.d_hidden, 0.0);

            // WAgg[l, d, d']
            double std_a = std::sqrt(2.0 / (config.d_hidden + config.d_node));
            W_aggregate[l] = DenseTensor("WAgg_" + std::to_string(l), {"d", "d'"}, {config.d_node, config.d_hidden});
            std::normal_distribution<double> dist_a(0.0, std_a);
            for (auto& v : W_aggregate[l].data) v = dist_a(rng);

            b_aggregate[l].resize(config.d_node, 0.0);

            // WSelf[l, d, d]
            double std_s = std::sqrt(2.0 / (config.d_node + config.d_node));
            W_self[l] = DenseTensor("WSelf_" + std::to_string(l), {"d_out", "d_in"}, {config.d_node, config.d_node});
            std::normal_distribution<double> dist_s(0.0, std_s);
            for (auto& v : W_self[l].data) v = dist_s(rng);
        }

        // WOut[c, d]
        double std_out = std::sqrt(2.0 / (config.d_node + config.n_classes));
        W_out = DenseTensor("WOut", {"c", "d"}, {config.n_classes, config.d_node});
        std::normal_distribution<double> dist_out(0.0, std_out);
        for (auto& v : W_out.data) v = dist_out(rng);

        b_out.resize(config.n_classes, 0.0);
    }

    static double relu(double x) {
        return std::max(0.0, x);
    }

    static void softmax(std::vector<double>& x) {
        double max_val = *std::max_element(x.begin(), x.end());
        double sum = 0.0;
        for (auto& v : x) {
            v = std::exp(v - max_val);
            sum += v;
        }
        for (auto& v : x) v /= sum;
    }

    void add_node(const std::string& node, const std::vector<double>& features) {
        if (features.size() != config.d_node) {
            throw std::runtime_error("Feature dimension mismatch");
        }

        node_embeddings[node].resize(config.n_layers + 1);
        node_embeddings[node][0] = features;

        for (size_t l = 1; l <= config.n_layers; ++l) {
            node_embeddings[node][l].resize(config.d_node, 0.0);
        }

        if (adjacency.find(node) == adjacency.end()) {
            adjacency[node] = {};
        }
    }

    void add_edge(const std::string& u, const std::string& v) {
        if (node_embeddings.find(u) == node_embeddings.end() || node_embeddings.find(v) == node_embeddings.end()) {
            throw std::runtime_error("Cannot add edge: node not found");
        }

        adjacency[u].push_back(v);
    }

    void load_graph_structure(const SparseBoolTensor& neig_tensor) {
        if (neig_tensor.arity() != 2) {
            throw std::runtime_error("Neig tensor must be binary");
        }

        for (const auto& tuple : neig_tensor.tuples) {
            std::string u = std::get<std::string>(tuple.values[0]);
            std::string v = std::get<std::string>(tuple.values[1]);

            if (node_embeddings.find(u) == node_embeddings.end()) {
                std::vector<double> features(config.d_node);
                std::normal_distribution<double> dist(0.0, 1.0);
                for (auto& f : features) f = dist(rng);
                add_node(u, features);
            }
            if (node_embeddings.find(v) == node_embeddings.end()) {
                std::vector<double> features(config.d_node);
                std::normal_distribution<double> dist(0.0, 1.0);
                for (auto& f : features) f = dist(rng);
                add_node(v, features);
            }

            add_edge(u, v);
        }
    }

    void forward() {
        std::vector<std::string> nodes;
        for (const auto& [node, _] : node_embeddings) {
            nodes.push_back(node);
        }

        for (size_t l = 0; l < config.n_layers; ++l) {
            // Z[n, l, d'] = relu(WP[l, d', d] Emb[n, l, d] + b)
            std::unordered_map<std::string, std::vector<double>> Z;

            for (const auto& node : nodes) {
                Z[node].resize(config.d_hidden);

                for (size_t d_out = 0; d_out < config.d_hidden; ++d_out) {
                    double sum = b_preprocess[l][d_out];
                    for (size_t d_in = 0; d_in < config.d_node; ++d_in) {
                        sum += W_preprocess[l].at({d_out, d_in}) * node_embeddings[node][l][d_in];
                    }
                    Z[node][d_out] = relu(sum);
                }
            }

            for (const auto& node : nodes) {
                // Agg[n, l, d'] = Σ{n' ∈ Neig(n)} Z[n', l, d']
                std::vector<double> agg(config.d_hidden, 0.0);

                for (const auto& neighbor : adjacency[node]) {
                    for (size_t d = 0; d < config.d_hidden; ++d) {
                        agg[d] += Z[neighbor][d];
                    }
                }

                if (config.use_self_loops) {
                    for (size_t d = 0; d < config.d_hidden; ++d) {
                        agg[d] += Z[node][d];
                    }
                }

                double degree = adjacency[node].size() + (config.use_self_loops ? 1.0 : 0.0);
                if (degree > 0) {
                    for (size_t d = 0; d < config.d_hidden; ++d) {
                        agg[d] /= std::sqrt(degree);
                    }
                }

                // Emb[n, l+1, d] = relu(WAgg * Agg + WSelf * Emb + b)
                for (size_t d_out = 0; d_out < config.d_node; ++d_out) {
                    double sum = b_aggregate[l][d_out];

                    for (size_t d_in = 0; d_in < config.d_hidden; ++d_in) {
                        sum += W_aggregate[l].at({d_out, d_in}) * agg[d_in];
                    }

                    for (size_t d_in = 0; d_in < config.d_node; ++d_in) {
                        sum += W_self[l].at({d_out, d_in}) * node_embeddings[node][l][d_in];
                    }

                    node_embeddings[node][l + 1][d_out] = relu(sum);
                }
            }
        }
    }

    // Y[n, c] = softmax(WOut[c, d] Emb[n, L, d] + b)
    std::vector<double> classify(const std::string& node) {
        if (node_embeddings.find(node) == node_embeddings.end()) {
            throw std::runtime_error("Node not found: " + node);
        }

        std::vector<double> logits(config.n_classes);

        for (size_t c = 0; c < config.n_classes; ++c) {
            double sum = b_out[c];
            for (size_t d = 0; d < config.d_node; ++d) {
                sum += W_out.at({c, d}) * node_embeddings[node][config.n_layers][d];
            }
            logits[c] = sum;
        }

        softmax(logits);
        return logits;
    }

    std::unordered_map<std::string, std::vector<double>> classify_all() {
        std::unordered_map<std::string, std::vector<double>> results;

        for (const auto& [node, _] : node_embeddings) {
            results[node] = classify(node);
        }

        return results;
    }

    const std::vector<double>& get_embedding(const std::string& node) const {
        auto it = node_embeddings.find(node);
        if (it == node_embeddings.end()) {
            throw std::runtime_error("Node not found: " + node);
        }
        return it->second[config.n_layers];
    }

    std::string to_tensor_logic() const {
        std::ostringstream oss;
        oss << "# Graph Neural Network in Tensor Logic (Table 1)\n\n";
        oss << "# Graph structure:  Neig(x, y)\n";
        oss << "# Initialization:   Emb[n, 0, d] = X[n, d]\n";
        oss << "# MLP:              Z[n, l, d'] = relu(WP[l, d', d] Emb[n, l, d] + b)\n";
        oss << "# Aggregation:      Agg[n, l, d'] = Neig(n, n') Z[n', l, d']\n";
        oss << "# Update:           Emb[n, l+1, d] = relu(WAgg[l, d, d'] Agg[n, l, d'] + WSelf[l, d, d] Emb[n, l, d] + "
               "b)\n";
        oss << "# Node class:       Y[n, c] = softmax(WOut[c, d] Emb[n, L, d])\n\n";
        oss << "Configuration:\n";
        oss << "  d_node = " << config.d_node << " (node embedding dim)\n";
        oss << "  d_hidden = " << config.d_hidden << " (hidden MLP dim)\n";
        oss << "  n_layers = " << config.n_layers << " (message passing layers)\n";
        oss << "  n_classes = " << config.n_classes << " (output classes)\n";
        return oss.str();
    }
};

}  // namespace tl

#endif  // TENSOR_LOGIC_GNN_HPP
