#include <algorithm>
#include <cassert>

#include "tensor_logic.hpp"

namespace tl {

SparseBoolTensor SparseBoolTensor::join(const SparseBoolTensor& U, const SparseBoolTensor& V) {
    std::vector<std::pair<size_t, size_t>> common_positions;  // (pos in U, pos in V)
    std::vector<size_t> u_only_positions;                     // α positions
    std::vector<size_t> v_only_positions;                     // γ positions

    for (size_t i = 0; i < U.index_names.size(); ++i) {
        bool found = false;
        for (size_t j = 0; j < V.index_names.size(); ++j) {
            if (U.index_names[i] == V.index_names[j]) {
                common_positions.emplace_back(i, j);
                found = true;
                break;
            }
        }
        if (!found) {
            u_only_positions.push_back(i);
        }
    }

    for (size_t j = 0; j < V.index_names.size(); ++j) {
        bool is_common = false;
        for (const auto& cp : common_positions) {
            if (cp.second == j) {
                is_common = true;
                break;
            }
        }
        if (!is_common) {
            v_only_positions.push_back(j);
        }
    }

    std::vector<std::string> result_indices;
    for (size_t pos : u_only_positions) {
        result_indices.push_back(U.index_names[pos]);
    }
    for (const auto& cp : common_positions) {
        result_indices.push_back(U.index_names[cp.first]);
    }
    for (size_t pos : v_only_positions) {
        result_indices.push_back(V.index_names[pos]);
    }

    SparseBoolTensor result(U.name + "_join_" + V.name, result_indices);

    if (common_positions.empty()) {
        for (const auto& u_tuple : U.tuples) {
            for (const auto& v_tuple : V.tuples) {
                Tuple combined;
                combined.values.insert(combined.values.end(), u_tuple.values.begin(), u_tuple.values.end());
                combined.values.insert(combined.values.end(), v_tuple.values.begin(), v_tuple.values.end());
                result.tuples.insert(combined);
            }
        }
        return result;
    }

    std::unordered_map<Tuple, std::vector<const Tuple*>, TupleHash> v_index;
    for (const auto& v_tuple : V.tuples) {
        Tuple key;
        for (const auto& cp : common_positions) {
            key.values.push_back(v_tuple.values[cp.second]);
        }
        v_index[key].push_back(&v_tuple);
    }

    for (const auto& u_tuple : U.tuples) {
        Tuple key;
        for (const auto& cp : common_positions) {
            key.values.push_back(u_tuple.values[cp.first]);
        }

        auto it = v_index.find(key);
        if (it != v_index.end()) {
            for (const Tuple* v_tuple : it->second) {
                Tuple result_tuple;

                // α: U-only indices
                for (size_t pos : u_only_positions) {
                    result_tuple.values.push_back(u_tuple.values[pos]);
                }
                // β: common indices (from U, same as V)
                for (const auto& cp : common_positions) {
                    result_tuple.values.push_back(u_tuple.values[cp.first]);
                }
                // γ: V-only indices
                for (size_t pos : v_only_positions) {
                    result_tuple.values.push_back(v_tuple->values[pos]);
                }

                result.tuples.insert(result_tuple);
            }
        }
    }

    return result;
}

SparseBoolTensor SparseBoolTensor::project(const std::vector<std::string>& keep_indices) const {
    std::vector<size_t> keep_positions;
    for (const auto& idx : keep_indices) {
        for (size_t i = 0; i < index_names.size(); ++i) {
            if (index_names[i] == idx) {
                keep_positions.push_back(i);
                break;
            }
        }
    }

    SparseBoolTensor result(name + "_proj", keep_indices);

    for (const auto& tuple : tuples) {
        Tuple projected;
        for (size_t pos : keep_positions) {
            projected.values.push_back(tuple.values[pos]);
        }
        result.tuples.insert(projected);
    }

    return result;
}

std::vector<std::string> TensorEquation::get_projected_indices() const {
    std::unordered_set<std::string> lhs_indices;
    for (const auto& idx : lhs.indices) {
        if (idx.is_variable) {
            lhs_indices.insert(idx.name);
        }
    }

    std::unordered_set<std::string> rhs_indices;
    for (const auto& tensor : rhs) {
        for (const auto& idx : tensor.indices) {
            if (idx.is_variable) {
                rhs_indices.insert(idx.name);
            }
        }
    }

    std::vector<std::string> projected;
    for (const auto& idx : rhs_indices) {
        if (lhs_indices.find(idx) == lhs_indices.end()) {
            projected.push_back(idx);
        }
    }
    return projected;
}

std::vector<std::string> TensorEquation::get_common_indices(const TensorRef& a, const TensorRef& b) {
    std::vector<std::string> common;
    for (const auto& idx_a : a.indices) {
        if (!idx_a.is_variable) continue;
        for (const auto& idx_b : b.indices) {
            if (idx_b.is_variable && idx_a.name == idx_b.name) {
                common.push_back(idx_a.name);
                break;
            }
        }
    }
    return common;
}

}  // namespace tl
