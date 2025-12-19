#include <algorithm>
#include <numeric>
#include <sstream>

#include "tensor_logic.hpp"

namespace tl {

DenseTensor DenseTensor::einsum(const DenseTensor& A, const DenseTensor& B,
                                const std::vector<std::string>& result_indices) {
    struct IndexInfo {
        std::string name;
        size_t size;
        int a_pos = -1;  // position in A (-1 if not present)
        int b_pos = -1;  // position in B (-1 if not present)
        int r_pos = -1;  // position in result (-1 if summed)
    };

    std::vector<IndexInfo> all_indices;
    std::unordered_map<std::string, size_t> index_map;  // name -> position in all_indices

    // A's indices
    for (size_t i = 0; i < A.index_names.size(); ++i) {
        IndexInfo info;
        info.name = A.index_names[i];
        info.size = A.shape[i];
        info.a_pos = i;
        index_map[info.name] = all_indices.size();
        all_indices.push_back(info);
    }

    // add or merge B's indices
    for (size_t i = 0; i < B.index_names.size(); ++i) {
        auto it = index_map.find(B.index_names[i]);
        if (it != index_map.end()) {
            all_indices[it->second].b_pos = i;
        } else {
            IndexInfo info;
            info.name = B.index_names[i];
            info.size = B.shape[i];
            info.b_pos = i;
            index_map[info.name] = all_indices.size();
            all_indices.push_back(info);
        }
    }

    // which indices are in result
    for (size_t i = 0; i < result_indices.size(); ++i) {
        auto it = index_map.find(result_indices[i]);
        if (it != index_map.end()) {
            all_indices[it->second].r_pos = i;
        }
    }

    std::vector<size_t> result_shape;
    std::vector<size_t> result_idx_to_all;
    for (size_t i = 0; i < result_indices.size(); ++i) {
        auto it = index_map.find(result_indices[i]);
        if (it != index_map.end()) {
            result_shape.push_back(all_indices[it->second].size);
            result_idx_to_all.push_back(it->second);
        } else {
            throw std::runtime_error("Result index '" + result_indices[i] + "' not found in inputs");
        }
    }

    DenseTensor result("result", result_indices, result_shape);
    std::vector<size_t> indices(all_indices.size(), 0);

    std::function<void(size_t)> iterate = [&](size_t depth) {
        if (depth == all_indices.size()) {
            std::vector<size_t> a_indices, b_indices, r_indices;

            for (size_t i = 0; i < all_indices.size(); ++i) {
                if (all_indices[i].a_pos >= 0) {
                    while (a_indices.size() <= static_cast<size_t>(all_indices[i].a_pos)) {
                        a_indices.push_back(0);
                    }
                    a_indices[all_indices[i].a_pos] = indices[i];
                }
                if (all_indices[i].b_pos >= 0) {
                    while (b_indices.size() <= static_cast<size_t>(all_indices[i].b_pos)) {
                        b_indices.push_back(0);
                    }
                    b_indices[all_indices[i].b_pos] = indices[i];
                }
            }

            for (size_t i = 0; i < result_idx_to_all.size(); ++i) {
                r_indices.push_back(indices[result_idx_to_all[i]]);
            }

            double a_val = a_indices.empty() ? 1.0 : A.at(a_indices);
            double b_val = b_indices.empty() ? 1.0 : B.at(b_indices);

            if (!r_indices.empty()) {
                result.at(r_indices) += a_val * b_val;
            } else if (result.data.size() == 1) {
                result.data[0] += a_val * b_val;
            }
            return;
        }

        for (size_t i = 0; i < all_indices[depth].size; ++i) {
            indices[depth] = i;
            iterate(depth + 1);
        }
    };

    iterate(0);
    return result;
}

std::string DenseTensor::to_string() const {
    std::ostringstream oss;
    oss << name << "[";
    for (size_t i = 0; i < index_names.size(); ++i) {
        if (i > 0) oss << ", ";
        oss << index_names[i];
    }
    oss << "] shape=(";
    for (size_t i = 0; i < shape.size(); ++i) {
        if (i > 0) oss << ", ";
        oss << shape[i];
    }
    oss << ")\n";

    // print data (for small tensors)
    if (data.size() <= 100) {
        if (shape.size() == 1) {
            oss << "[";
            for (size_t i = 0; i < data.size(); ++i) {
                if (i > 0) oss << ", ";
                oss << data[i];
            }
            oss << "]";
        } else if (shape.size() == 2) {
            for (size_t i = 0; i < shape[0]; ++i) {
                oss << "  [";
                for (size_t j = 0; j < shape[1]; ++j) {
                    if (j > 0) oss << ", ";
                    oss << data[i * shape[1] + j];
                }
                oss << "]\n";
            }
        } else {
            oss << "  (data: " << data.size() << " elements)";
        }
    } else {
        oss << "  (data: " << data.size() << " elements, too large to display)";
    }

    return oss.str();
}

}  // namespace tl
