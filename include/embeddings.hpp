#ifndef TENSOR_LOGIC_EMBEDDINGS_HPP
#define TENSOR_LOGIC_EMBEDDINGS_HPP

#include <cmath>
#include <random>

#include "tensor_logic.hpp"

namespace tl {

class EmbeddingSpace {
   public:
    size_t dim;          // embedding dim
    double temperature;  // T = 0: deductive, T > 0: analogical
    std::unordered_map<std::string, std::vector<double>> embeddings;
    std::mt19937 rng;

    EmbeddingSpace(size_t dimension = 64, double temp = 0.0, unsigned seed = 42)
        : dim(dimension), temperature(temp), rng(seed) {}

    const std::vector<double>& get_embedding(const std::string& object) {
        auto it = embeddings.find(object);
        if (it != embeddings.end()) {
            return it->second;
        }

        std::normal_distribution<double> dist(0.0, 1.0);
        std::vector<double> emb(dim);
        double norm = 0.0;
        for (size_t i = 0; i < dim; ++i) {
            emb[i] = dist(rng);
            norm += emb[i] * emb[i];
        }
        norm = std::sqrt(norm);
        for (size_t i = 0; i < dim; ++i) {
            emb[i] /= norm;
        }

        embeddings[object] = std::move(emb);
        return embeddings[object];
    }

    void set_embedding(const std::string& object, const std::vector<double>& emb) {
        embeddings[object] = emb;
    }

    double similarity(const std::string& a, const std::string& b) {
        const auto& emb_a = get_embedding(a);
        const auto& emb_b = get_embedding(b);
        double dot = 0.0;
        for (size_t i = 0; i < dim; ++i) {
            dot += emb_a[i] * emb_b[i];
        }
        return dot;
    }

    // σ(x, T) = 1 / (1 + e^(-x/T))
    double sigma(double x) const {
        if (temperature <= 1e-10) {
            // heaviside step function.
            return x > 0.5 ? 1.0 : 0.0;
        }
        return 1.0 / (1.0 + std::exp(-x / temperature));
    }

    // EmbR[i, j] = Σ_{(x, y) ∈ R} Emb[x, i] * Emb[y, j]
    DenseTensor embed_relation(const SparseBoolTensor& relation) {
        if (relation.arity() != 2) {
            throw std::runtime_error("embed_relation requires binary relation");
        }

        DenseTensor result("Emb_" + relation.name, {"i", "j"}, {dim, dim});

        for (const auto& tuple : relation.tuples) {
            std::string x = std::get<std::string>(tuple.values[0]);
            std::string y = std::get<std::string>(tuple.values[1]);

            const auto& emb_x = get_embedding(x);
            const auto& emb_y = get_embedding(y);

            for (size_t i = 0; i < dim; ++i) {
                for (size_t j = 0; j < dim; ++j) {
                    result.at({i, j}) += emb_x[i] * emb_y[j];
                }
            }
        }

        return result;
    }

    // D[A, B] = EmbR[i, j] * Emb[A, i] * Emb[B, j]
    double query_relation(const DenseTensor& emb_relation, const std::string& a, const std::string& b) {
        const auto& emb_a = get_embedding(a);
        const auto& emb_b = get_embedding(b);

        double result = 0.0;
        for (size_t i = 0; i < dim; ++i) {
            for (size_t j = 0; j < dim; ++j) {
                result += emb_relation.at({i, j}) * emb_a[i] * emb_b[j];
            }
        }

        return sigma(result);
    }

    std::vector<std::pair<std::string, double>> analogical_query(const DenseTensor& emb_relation,
                                                                 const std::string& query_obj,
                                                                 const std::vector<std::string>& candidates) {
        std::vector<std::pair<std::string, double>> results;

        for (const auto& candidate : candidates) {
            double score = query_relation(emb_relation, query_obj, candidate);
            if (score > 0.01) {
                results.emplace_back(candidate, score);
            }
        }

        std::sort(results.begin(), results.end(), [](const auto& a, const auto& b) { return a.second > b.second; });

        return results;
    }
};

class EmbeddedProgram {
   public:
    Program symbolic;      // symbolic (boolean tensor) program
    EmbeddingSpace space;  // embedding space for neural reasoning
    std::unordered_map<std::string, DenseTensor> embedded_relations;

    EmbeddedProgram(size_t dim = 64, double temp = 0.0) : space(dim, temp) {}

    void set_temperature(double t) {
        space.temperature = t;
    }

    void add_fact(const std::string& relation, const std::string& arg1, const std::string& arg2) {
        symbolic.add_fact(relation, {arg1, arg2});

        auto it = embedded_relations.find(relation);
        if (it == embedded_relations.end()) {
            embedded_relations.emplace(relation, DenseTensor("Emb_" + relation, {"i", "j"}, {space.dim, space.dim}));
            it = embedded_relations.find(relation);
        }

        const auto& emb_x = space.get_embedding(arg1);
        const auto& emb_y = space.get_embedding(arg2);

        for (size_t i = 0; i < space.dim; ++i) {
            for (size_t j = 0; j < space.dim; ++j) {
                it->second.at({i, j}) += emb_x[i] * emb_y[j];
            }
        }
    }

    void add_rule(const TensorEquation& eq) {
        symbolic.add_equation(eq);
    }

    void forward_chain() {
        symbolic.forward_chain();

        for (const auto& [name, tensor] : symbolic.bool_tensors) {
            if (tensor.arity() == 2) {
                embedded_relations[name] = space.embed_relation(tensor);
            }
        }
    }

    // T = 0: exact symbolic query (deductive); T > 0: soft embedding query (analogical)
    double query(const std::string& relation, const std::string& a, const std::string& b) {
        if (space.temperature <= 1e-10) {
            auto it = symbolic.bool_tensors.find(relation);
            if (it == symbolic.bool_tensors.end()) return 0.0;
            return it->second.contains(Tuple{{a, b}}) ? 1.0 : 0.0;
        } else {
            auto it = embedded_relations.find(relation);
            if (it == embedded_relations.end()) return 0.0;
            return space.query_relation(it->second, a, b);
        }
    }

    void trace_deduction(const std::string& relation, const std::string& a, const std::string& b) {
        std::cout << "=== Deductive Trace (T = 0) ===" << std::endl;
        std::cout << "Query: " << relation << "(" << a << ", " << b << ")" << std::endl;

        for (const auto& [name, tensor] : symbolic.bool_tensors) {
            if (name == relation && tensor.contains(Tuple{{a, b}})) {
                std::cout << "  ✓ Direct fact: " << relation << "(" << a << ", " << b << ")" << std::endl;
                return;
            }
        }

        for (const auto& eq : symbolic.equations) {
            if (eq.lhs.name == relation) {
                std::cout << "  Rule: " << eq.to_string() << std::endl;

                // TODO: This is just the bare bones boo-hoo logic; replace it with a proper proof search.
                for (const auto& body_ref : eq.rhs) {
                    std::cout << "    Checking: " << body_ref.name << "(.)" << std::endl;
                }
            }
        }
    }
};

}  // namespace tl

#endif  // TENSOR_LOGIC_EMBEDDINGS_HPP
