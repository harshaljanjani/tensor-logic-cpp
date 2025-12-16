#include <algorithm>
#include <queue>

#include "tensor_logic.hpp"

namespace tl {

void Program::forward_chain(int max_iterations) {
    bool changed = true;
    int iteration = 0;

    while (changed && iteration < max_iterations) {
        changed = false;
        ++iteration;

        for (const auto& eq : equations) {
            if (!eq.lhs.is_boolean) continue;
            if (eq.rhs.empty()) continue;

            bool all_exist = true;
            for (const auto& ref : eq.rhs) {
                if (bool_tensors.find(ref.name) == bool_tensors.end()) {
                    all_exist = false;
                    break;
                }
            }
            if (!all_exist) continue;

            SparseBoolTensor result = bool_tensors.at(eq.rhs[0].name);

            auto remap_tensor = [](const SparseBoolTensor& tensor, const TensorRef& ref) {
                SparseBoolTensor remapped = tensor;
                remapped.index_names.clear();
                for (const auto& idx : ref.indices) {
                    remapped.index_names.push_back(idx.name);
                }
                return remapped;
            };

            result = remap_tensor(result, eq.rhs[0]);

            for (size_t i = 1; i < eq.rhs.size(); ++i) {
                SparseBoolTensor next = bool_tensors.at(eq.rhs[i].name);
                next = remap_tensor(next, eq.rhs[i]);
                result = SparseBoolTensor::join(result, next);
            }

            std::vector<std::string> lhs_indices;
            for (const auto& idx : eq.lhs.indices) {
                lhs_indices.push_back(idx.name);
            }
            result = result.project(lhs_indices);

            auto it = bool_tensors.find(eq.lhs.name);
            if (it == bool_tensors.end()) {
                bool_tensors.emplace(eq.lhs.name, SparseBoolTensor(eq.lhs.name, lhs_indices));
                it = bool_tensors.find(eq.lhs.name);
            }

            size_t old_size = it->second.tuples.size();
            for (const auto& tuple : result.tuples) {
                it->second.tuples.insert(tuple);
            }

            if (it->second.tuples.size() > old_size) {
                changed = true;
            }
        }
    }
}

std::vector<Tuple> Program::query(const std::string& relation) const {
    auto it = bool_tensors.find(relation);
    if (it == bool_tensors.end()) {
        return {};
    }
    return std::vector<Tuple>(it->second.tuples.begin(), it->second.tuples.end());
}

std::vector<Tuple> Program::query(const std::string& relation, const Binding& partial) const {
    auto it = bool_tensors.find(relation);
    if (it == bool_tensors.end()) {
        return {};
    }

    std::vector<Tuple> results;
    for (const auto& tuple : it->second.tuples) {
        bool matches = true;
        for (size_t i = 0; i < it->second.index_names.size() && i < tuple.size(); ++i) {
            auto bound = partial.find(it->second.index_names[i]);
            if (bound != partial.end()) {
                if (tuple.values[i] != bound->second) {
                    matches = false;
                    break;
                }
            }
        }
        if (matches) {
            results.push_back(tuple);
        }
    }
    return results;
}

bool Program::backward_chain(const TensorRef& goal, Binding& binding) {
    auto it = bool_tensors.find(goal.name);
    if (it != bool_tensors.end()) {
        for (const auto& tuple : it->second.tuples) {
            bool matches = true;
            Binding new_binding = binding;

            for (size_t i = 0; i < goal.indices.size() && i < tuple.size(); ++i) {
                const Index& idx = goal.indices[i];
                if (idx.is_variable) {
                    auto bound = new_binding.find(idx.name);
                    if (bound != new_binding.end()) {
                        if (bound->second != tuple.values[i]) {
                            matches = false;
                            break;
                        }
                    } else {
                        new_binding[idx.name] = tuple.values[i];
                    }
                } else {
                    TupleValue const_val;
                    try {
                        const_val = std::stoi(idx.name);
                    } catch (...) {
                        const_val = idx.name;
                    }
                    if (tuple.values[i] != const_val) {
                        matches = false;
                        break;
                    }
                }
            }

            if (matches) {
                binding = new_binding;
                return true;
            }
        }
    }

    for (const auto& eq : equations) {
        if (!eq.lhs.is_boolean) continue;
        if (eq.lhs.name != goal.name) continue;
        if (eq.lhs.indices.size() != goal.indices.size()) continue;

        Binding rule_binding = binding;
        bool can_unify = true;

        for (size_t i = 0; i < goal.indices.size(); ++i) {
            const Index& goal_idx = goal.indices[i];
            const Index& head_idx = eq.lhs.indices[i];

            if (goal_idx.is_variable) {
                auto bound = rule_binding.find(goal_idx.name);
                if (bound != rule_binding.end()) {
                    if (head_idx.is_variable) {
                        auto head_bound = rule_binding.find(head_idx.name);
                        if (head_bound != rule_binding.end()) {
                            if (head_bound->second != bound->second) {
                                can_unify = false;
                                break;
                            }
                        } else {
                            rule_binding[head_idx.name] = bound->second;
                        }
                    }
                } else if (head_idx.is_variable) {
                    auto head_bound = rule_binding.find(head_idx.name);
                    if (head_bound != rule_binding.end()) {
                        rule_binding[goal_idx.name] = head_bound->second;
                    }
                }
            } else {
                TupleValue const_val;
                try {
                    const_val = std::stoi(goal_idx.name);
                } catch (...) {
                    const_val = goal_idx.name;
                }
                if (head_idx.is_variable) {
                    auto head_bound = rule_binding.find(head_idx.name);
                    if (head_bound != rule_binding.end()) {
                        if (head_bound->second != const_val) {
                            can_unify = false;
                            break;
                        }
                    } else {
                        rule_binding[head_idx.name] = const_val;
                    }
                }
            }
        }

        if (!can_unify) continue;

        bool all_proved = true;
        for (const auto& body_atom : eq.rhs) {
            if (!backward_chain(body_atom, rule_binding)) {
                all_proved = false;
                break;
            }
        }

        if (all_proved) {
            for (size_t i = 0; i < goal.indices.size(); ++i) {
                if (goal.indices[i].is_variable) {
                    auto it = rule_binding.find(eq.lhs.indices[i].name);
                    if (it != rule_binding.end()) {
                        binding[goal.indices[i].name] = it->second;
                    }
                }
            }
            return true;
        }
    }

    return false;
}

std::string Program::to_string() const {
    std::ostringstream oss;

    oss << "=== Tensor Logic Program ===\n\n";

    if (!equations.empty()) {
        oss << "Equations/Rules:\n";
        for (const auto& eq : equations) {
            oss << "  " << eq.to_string() << "\n";
        }
        oss << "\n";
    }

    if (!bool_tensors.empty()) {
        oss << "Boolean Tensors (Relations):\n";
        for (const auto& [name, tensor] : bool_tensors) {
            oss << "  " << name << ": " << tensor.size() << " tuples\n";
            for (const auto& tuple : tensor.tuples) {
                oss << "    " << name << tuple.to_string() << "\n";
            }
        }
        oss << "\n";
    }

    if (!dense_tensors.empty()) {
        oss << "Dense Tensors:\n";
        for (const auto& [name, tensor] : dense_tensors) {
            oss << "  " << tensor.to_string() << "\n";
        }
    }

    return oss.str();
}

}  // namespace tl
