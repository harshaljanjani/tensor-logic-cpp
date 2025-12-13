#ifndef TENSOR_LOGIC_TRANSFORMER_HPP
#define TENSOR_LOGIC_TRANSFORMER_HPP

#include <cmath>
#include <random>

#include "tensor_logic.hpp"

namespace tl {

struct TransformerConfig {
    size_t vocab_size = 100;
    size_t max_seq_len = 32;
    size_t d_model = 64;
    size_t n_heads = 4;
    size_t d_k = 16;
    size_t d_v = 16;
    size_t n_blocks = 2;
    size_t d_ff = 128;
    double L = 10000.0;
};

class Transformer {
   public:
    TransformerConfig config;

    // Emb[t, d]
    DenseTensor token_embeddings;

    // PosEnc[p, d]
    DenseTensor pos_encoding;

    // WQ[b, h, dk, d], WK[b, h, dk, d], WV[b, h, dv, d]
    std::vector<std::vector<DenseTensor>> W_Q, W_K, W_V;

    // WO[b, d, h*dv]
    std::vector<DenseTensor> W_O;

    std::vector<DenseTensor> W_ff1, W_ff2;

    // WOut[t, d]
    DenseTensor W_out;

    std::mt19937 rng;

    Transformer(const TransformerConfig& cfg = TransformerConfig(), unsigned seed = 42) : config(cfg), rng(seed) {
        initialize_weights();
        compute_positional_encoding();
    }

    void initialize_weights() {
        std::normal_distribution<double> dist(0.0, 0.02);

        token_embeddings = DenseTensor("Emb", {"t", "d"}, {config.vocab_size, config.d_model});
        for (auto& v : token_embeddings.data) v = dist(rng);

        W_Q.resize(config.n_blocks);
        W_K.resize(config.n_blocks);
        W_V.resize(config.n_blocks);
        W_O.resize(config.n_blocks);
        W_ff1.resize(config.n_blocks);
        W_ff2.resize(config.n_blocks);

        for (size_t b = 0; b < config.n_blocks; ++b) {
            W_Q[b].resize(config.n_heads);
            W_K[b].resize(config.n_heads);
            W_V[b].resize(config.n_heads);

            for (size_t h = 0; h < config.n_heads; ++h) {
                W_Q[b][h] = DenseTensor("WQ", {"dk", "d"}, {config.d_k, config.d_model});
                W_K[b][h] = DenseTensor("WK", {"dk", "d"}, {config.d_k, config.d_model});
                W_V[b][h] = DenseTensor("WV", {"dv", "d"}, {config.d_v, config.d_model});

                for (auto& v : W_Q[b][h].data) v = dist(rng);
                for (auto& v : W_K[b][h].data) v = dist(rng);
                for (auto& v : W_V[b][h].data) v = dist(rng);
            }

            W_O[b] = DenseTensor("WO", {"d", "dv_concat"}, {config.d_model, config.n_heads * config.d_v});
            for (auto& v : W_O[b].data) v = dist(rng);

            W_ff1[b] = DenseTensor("Wff1", {"dff", "d"}, {config.d_ff, config.d_model});
            W_ff2[b] = DenseTensor("Wff2", {"d", "dff"}, {config.d_model, config.d_ff});
            for (auto& v : W_ff1[b].data) v = dist(rng);
            for (auto& v : W_ff2[b].data) v = dist(rng);
        }

        W_out = DenseTensor("WOut", {"t", "d"}, {config.vocab_size, config.d_model});
        for (auto& v : W_out.data) v = dist(rng);
    }

    // PosEnc[p, d] = Even(d) sin(p/L^(d/De)) + Odd(d) cos(p/L^((d-1)/De))
    void compute_positional_encoding() {
        pos_encoding = DenseTensor("PosEnc", {"p", "d"}, {config.max_seq_len, config.d_model});

        for (size_t p = 0; p < config.max_seq_len; ++p) {
            for (size_t d = 0; d < config.d_model; ++d) {
                double angle = p / std::pow(config.L, static_cast<double>(d) / config.d_model);
                if (d % 2 == 0) {
                    pos_encoding.at({p, d}) = std::sin(angle);
                } else {
                    pos_encoding.at({p, d}) = std::cos(angle);
                }
            }
        }
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

    static double relu(double x) {
        return std::max(0.0, x);
    }

    // TODO: Replace this with a full layer normalization implementation.
    static void layer_norm(std::vector<double>& x) {
        double mean = 0.0, var = 0.0;
        for (double v : x) mean += v;
        mean /= x.size();
        for (double v : x) var += (v - mean) * (v - mean);
        var /= x.size();
        double std_dev = std::sqrt(var + 1e-6);
        for (auto& v : x) v = (v - mean) / std_dev;
    }

    std::vector<std::vector<double>> forward(const std::vector<size_t>& tokens) {
        size_t seq_len = tokens.size();

        // EmbX[p, d] = X(p, t) Emb[t, d]
        std::vector<std::vector<double>> stream(seq_len, std::vector<double>(config.d_model));

        for (size_t p = 0; p < seq_len; ++p) {
            size_t t = tokens[p];
            for (size_t d = 0; d < config.d_model; ++d) {
                // EmbX[p, d] = Emb[t, d]
                stream[p][d] = token_embeddings.at({t, d});
                // Stream[0, p, d] = EmbX[p, d] + PosEnc[p, d]
                stream[p][d] += pos_encoding.at({p, d});
            }
        }

        for (size_t b = 0; b < config.n_blocks; ++b) {
            stream = transformer_block(stream, b);
        }

        // Y[p, t.] = softmax(WOut[t, d] Stream[B, p, d])
        std::vector<std::vector<double>> logits(seq_len, std::vector<double>(config.vocab_size));

        for (size_t p = 0; p < seq_len; ++p) {
            for (size_t t = 0; t < config.vocab_size; ++t) {
                double sum = 0.0;
                for (size_t d = 0; d < config.d_model; ++d) {
                    sum += W_out.at({t, d}) * stream[p][d];
                }
                logits[p][t] = sum;
            }
            softmax(logits[p]);
        }

        return logits;
    }

    std::vector<std::vector<double>> transformer_block(const std::vector<std::vector<double>>& input, size_t block) {
        size_t seq_len = input.size();

        std::vector<std::vector<double>> attn_output(seq_len, std::vector<double>(config.n_heads * config.d_v, 0.0));

        for (size_t h = 0; h < config.n_heads; ++h) {
            // Query[b, h, p, dk] = WQ[b, h, dk, d] Stream[b, p, d]
            std::vector<std::vector<double>> Q(seq_len, std::vector<double>(config.d_k));
            std::vector<std::vector<double>> K(seq_len, std::vector<double>(config.d_k));
            std::vector<std::vector<double>> V(seq_len, std::vector<double>(config.d_v));

            for (size_t p = 0; p < seq_len; ++p) {
                for (size_t dk = 0; dk < config.d_k; ++dk) {
                    double q_sum = 0.0, k_sum = 0.0;
                    for (size_t d = 0; d < config.d_model; ++d) {
                        q_sum += W_Q[block][h].at({dk, d}) * input[p][d];
                        k_sum += W_K[block][h].at({dk, d}) * input[p][d];
                    }
                    Q[p][dk] = q_sum;
                    K[p][dk] = k_sum;
                }
                for (size_t dv = 0; dv < config.d_v; ++dv) {
                    double v_sum = 0.0;
                    for (size_t d = 0; d < config.d_model; ++d) {
                        v_sum += W_V[block][h].at({dv, d}) * input[p][d];
                    }
                    V[p][dv] = v_sum;
                }
            }

            // Comp[b, h, p, p'.] = softmax(Query * Key^T / sqrt(dk))
            double scale = 1.0 / std::sqrt(static_cast<double>(config.d_k));

            for (size_t p = 0; p < seq_len; ++p) {
                std::vector<double> scores(seq_len);
                for (size_t p2 = 0; p2 < seq_len; ++p2) {
                    double dot = 0.0;
                    for (size_t dk = 0; dk < config.d_k; ++dk) {
                        dot += Q[p][dk] * K[p2][dk];
                    }
                    scores[p2] = dot * scale;
                }
                softmax(scores);

                // Attn[b, h, p, dv] = Comp[b, h, p, p'] Val[b, h, p', dv]
                for (size_t dv = 0; dv < config.d_v; ++dv) {
                    double attn_val = 0.0;
                    for (size_t p2 = 0; p2 < seq_len; ++p2) {
                        attn_val += scores[p2] * V[p2][dv];
                    }
                    attn_output[p][h * config.d_v + dv] = attn_val;
                }
            }
        }

        // Merge[b, p, dm] = concat(Attn[b, h, p, dv])
        std::vector<std::vector<double>> stream(seq_len, std::vector<double>(config.d_model));
        for (size_t p = 0; p < seq_len; ++p) {
            for (size_t d = 0; d < config.d_model; ++d) {
                double proj = 0.0;
                for (size_t dv = 0; dv < config.n_heads * config.d_v; ++dv) {
                    proj += W_O[block].at({d, dv}) * attn_output[p][dv];
                }
                stream[p][d] = input[p][d] + proj;
            }
            layer_norm(stream[p]);
        }

        // MLP[b, p] = relu(WP[p, d] Stream[b, p, d])
        for (size_t p = 0; p < seq_len; ++p) {
            std::vector<double> hidden(config.d_ff);
            for (size_t ff = 0; ff < config.d_ff; ++ff) {
                double sum = 0.0;
                for (size_t d = 0; d < config.d_model; ++d) {
                    sum += W_ff1[block].at({ff, d}) * stream[p][d];
                }
                hidden[ff] = relu(sum);
            }

            std::vector<double> ff_out(config.d_model);
            for (size_t d = 0; d < config.d_model; ++d) {
                double sum = 0.0;
                for (size_t ff = 0; ff < config.d_ff; ++ff) {
                    sum += W_ff2[block].at({d, ff}) * hidden[ff];
                }
                ff_out[d] = sum;
            }

            for (size_t d = 0; d < config.d_model; ++d) {
                stream[p][d] += ff_out[d];
            }
            layer_norm(stream[p]);
        }

        return stream;
    }

    std::string to_tensor_logic() const {
        std::ostringstream oss;
        oss << "# Transformer in Tensor Logic (Table 2)\n\n";
        oss << "# Input: X(p, t) - position p has token t\n";
        oss << "# Embedding: EmbX[p, d] = X(p, t) Emb[t, d]\n";
        oss << "# Pos. encoding: PosEnc[p, d] = Even(d) sin(p/L^(d/De)) + Odd(d) cos(p/L^((d-1)/De))\n";
        oss << "# Residual: Stream[0, p, d] = EmbX[p, d] + PosEnc[p, d]\n\n";
        oss << "# For each block b and head h:\n";
        oss << "#   Query[b, h, p, dk] = WQ[b, h, dk, d] Stream[b, p, d]\n";
        oss << "#   Key[b, h, p, dk] = WK[b, h, dk, d] Stream[b, p, d]\n";
        oss << "#   Val[b, h, p, dv] = WV[b, h, dv, d] Stream[b, p, d]\n";
        oss << "#   Comp[b, h, p, p'.] = softmax(Query[b, h, p, dk] Key[b, h, p', dk] / sqrt(Dk))\n";
        oss << "#   Attn[b, h, p, dv] = Comp[b, h, p, p'] Val[b, h, p', dv]\n\n";
        oss << "# Output: Y[p, t.] = softmax(WOut[t, d] Stream[B, p, d])\n";
        return oss.str();
    }
};

class Tokenizer {
   public:
    std::unordered_map<std::string, size_t> word_to_id;
    std::vector<std::string> id_to_word;
    size_t next_id = 0;

    size_t encode(const std::string& word) {
        auto it = word_to_id.find(word);
        if (it != word_to_id.end()) return it->second;

        word_to_id[word] = next_id;
        id_to_word.push_back(word);
        return next_id++;
    }

    std::string decode(size_t id) const {
        return id < id_to_word.size() ? id_to_word[id] : "<unk>";
    }

    std::vector<size_t> tokenize(const std::string& text) {
        std::vector<size_t> tokens;
        std::istringstream iss(text);
        std::string word;
        while (iss >> word) {
            tokens.push_back(encode(word));
        }
        return tokens;
    }
};

}  // namespace tl

#endif  // TENSOR_LOGIC_TRANSFORMER_HPP
