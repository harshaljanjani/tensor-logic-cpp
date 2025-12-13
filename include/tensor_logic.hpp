#ifndef TENSOR_LOGIC_HPP
#define TENSOR_LOGIC_HPP

#include <cmath>
#include <functional>
#include <iostream>
#include <memory>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <variant>
#include <vector>

namespace tl {

class Tensor;
class SparseBoolTensor;
class DenseTensor;
class TensorEquation;
class Program;

enum class TensorType { Boolean, Real, Integer };

struct Index {
    std::string name;
    bool is_variable;
    int offset = 0;

    Index(const std::string& n, bool var = true, int off = 0) : name(n), is_variable(var), offset(off) {}

    bool operator==(const Index& other) const {
        return name == other.name && is_variable == other.is_variable && offset == other.offset;
    }

    std::string to_string() const {
        std::string result = name;
        if (offset > 0)
            result += "+" + std::to_string(offset);
        else if (offset < 0)
            result += std::to_string(offset);
        return result;
    }
};

struct IndexHash {
    size_t operator()(const Index& idx) const {
        return std::hash<std::string>()(idx.name) ^ std::hash<bool>()(idx.is_variable) ^ std::hash<int>()(idx.offset);
    }
};

enum class Nonlinearity { None, Step, Sigmoid, ReLU, Softmax, Tanh, Exp, Log, Sqrt };

inline double apply_nonlinearity(Nonlinearity nl, double x, double temp = 1.0) {
    switch (nl) {
        case Nonlinearity::None:
            return x;
        case Nonlinearity::Step:
            return x > 0 ? 1.0 : 0.0;
        case Nonlinearity::Sigmoid:
            return 1.0 / (1.0 + std::exp(-x / temp));
        case Nonlinearity::ReLU:
            return std::max(0.0, x);
        case Nonlinearity::Tanh:
            return std::tanh(x);
        case Nonlinearity::Exp:
            return std::exp(x);
        case Nonlinearity::Log:
            return std::log(x);
        case Nonlinearity::Sqrt:
            return std::sqrt(x);
        default:
            return x;
    }
}

using TupleValue = std::variant<int, std::string>;

struct Tuple {
    std::vector<TupleValue> values;

    Tuple() = default;
    Tuple(std::initializer_list<TupleValue> vals) : values(vals) {}
    explicit Tuple(std::vector<TupleValue> vals) : values(std::move(vals)) {}

    bool operator==(const Tuple& other) const {
        return values == other.values;
    }

    size_t size() const {
        return values.size();
    }

    std::string to_string() const {
        std::ostringstream oss;
        oss << "(";
        for (size_t i = 0; i < values.size(); ++i) {
            if (i > 0) oss << ", ";
            if (std::holds_alternative<int>(values[i])) {
                oss << std::get<int>(values[i]);
            } else {
                oss << std::get<std::string>(values[i]);
            }
        }
        oss << ")";
        return oss.str();
    }
};

struct TupleHash {
    size_t operator()(const Tuple& t) const {
        size_t h = 0;
        for (const auto& v : t.values) {
            if (std::holds_alternative<int>(v)) {
                h ^= std::hash<int>()(std::get<int>(v)) + 0x9e3779b9 + (h << 6) + (h >> 2);
            } else {
                h ^= std::hash<std::string>()(std::get<std::string>(v)) + 0x9e3779b9 + (h << 6) + (h >> 2);
            }
        }
        return h;
    }
};

using Binding = std::unordered_map<std::string, TupleValue>;

class SparseBoolTensor {
   public:
    std::string name;
    std::vector<std::string> index_names;
    std::unordered_set<Tuple, TupleHash> tuples;

    SparseBoolTensor() = default;

    SparseBoolTensor(const std::string& n, std::vector<std::string> indices = {})
        : name(n), index_names(std::move(indices)) {}

    void add_tuple(const Tuple& t) {
        if (index_names.empty()) {
            index_names.resize(t.size());
            for (size_t i = 0; i < t.size(); ++i) {
                index_names[i] = "arg" + std::to_string(i);
            }
        }
        tuples.insert(t);
    }

    void add_tuple(std::initializer_list<TupleValue> vals) {
        add_tuple(Tuple(vals));
    }

    bool contains(const Tuple& t) const {
        return tuples.find(t) != tuples.end();
    }

    size_t arity() const {
        return index_names.size();
    }
    size_t size() const {
        return tuples.size();
    }

    // tensor join: (U ⋈ V)[α, β, γ] = U[α, β] · V[β, γ]
    static SparseBoolTensor join(const SparseBoolTensor& U, const SparseBoolTensor& V);

    // tensor projection: π_α(T) = sum_β T[α, β]
    SparseBoolTensor project(const std::vector<std::string>& keep_indices) const;

    std::string to_string() const {
        std::ostringstream oss;
        oss << name << "(";
        for (size_t i = 0; i < index_names.size(); ++i) {
            if (i > 0) oss << ", ";
            oss << index_names[i];
        }
        oss << ") = {\n";
        for (const auto& t : tuples) {
            oss << "  " << name << t.to_string() << "\n";
        }
        oss << "}";
        return oss.str();
    }
};

class DenseTensor {
   public:
    std::string name;
    std::vector<std::string> index_names;
    std::vector<size_t> shape;
    std::vector<double> data;

    DenseTensor() = default;

    DenseTensor(const std::string& n, std::vector<std::string> indices, std::vector<size_t> sh)
        : name(n), index_names(std::move(indices)), shape(std::move(sh)) {
        size_t total = 1;
        for (size_t s : shape) total *= s;
        data.resize(total, 0.0);
    }

    DenseTensor(const std::string& n, std::vector<double> values) : name(n), data(std::move(values)) {
        shape = {data.size()};
        index_names = {"i"};
    }

    size_t rank() const {
        return shape.size();
    }

    size_t flat_index(const std::vector<size_t>& indices) const {
        size_t idx = 0;
        size_t multiplier = 1;
        for (int i = shape.size() - 1; i >= 0; --i) {
            idx += indices[i] * multiplier;
            multiplier *= shape[i];
        }
        return idx;
    }

    double& at(const std::vector<size_t>& indices) {
        return data[flat_index(indices)];
    }

    double at(const std::vector<size_t>& indices) const {
        return data[flat_index(indices)];
    }

    static DenseTensor einsum(const DenseTensor& A, const DenseTensor& B,
                              const std::vector<std::string>& result_indices);

    void apply(Nonlinearity nl, double temp = 1.0) {
        for (auto& v : data) {
            v = apply_nonlinearity(nl, v, temp);
        }
    }

    std::string to_string() const;
};

struct TensorRef {
    std::string name;
    std::vector<Index> indices;
    bool is_boolean;

    TensorRef() : is_boolean(false) {}
    TensorRef(const std::string& n, std::vector<Index> idx = {}, bool boolean = false)
        : name(n), indices(std::move(idx)), is_boolean(boolean) {}

    std::string to_string() const {
        std::ostringstream oss;
        oss << name;
        if (!indices.empty()) {
            oss << (is_boolean ? "(" : "[");
            for (size_t i = 0; i < indices.size(); ++i) {
                if (i > 0) oss << ", ";
                oss << indices[i].to_string();
            }
            oss << (is_boolean ? ")" : "]");
        }
        return oss.str();
    }
};

struct TensorEquation {
    TensorRef lhs;
    std::vector<TensorRef> rhs;
    Nonlinearity nonlinearity = Nonlinearity::None;
    double temperature = 1.0;

    std::vector<std::string> get_projected_indices() const;

    static std::vector<std::string> get_common_indices(const TensorRef& a, const TensorRef& b);

    std::string to_string() const {
        std::ostringstream oss;
        oss << lhs.to_string();
        if (lhs.is_boolean) {
            oss << " <- ";
        } else {
            oss << " = ";
            if (nonlinearity != Nonlinearity::None) {
                switch (nonlinearity) {
                    case Nonlinearity::Step:
                        oss << "step(";
                        break;
                    case Nonlinearity::Sigmoid:
                        oss << "sig(";
                        break;
                    case Nonlinearity::ReLU:
                        oss << "relu(";
                        break;
                    case Nonlinearity::Softmax:
                        oss << "softmax(";
                        break;
                    case Nonlinearity::Tanh:
                        oss << "tanh(";
                        break;
                    default:
                        break;
                }
            }
        }
        for (size_t i = 0; i < rhs.size(); ++i) {
            if (i > 0) oss << (lhs.is_boolean ? ", " : " ");
            oss << rhs[i].to_string();
        }
        if (nonlinearity != Nonlinearity::None && !lhs.is_boolean) {
            oss << ")";
        }
        return oss.str();
    }
};

class Program {
   public:
    std::vector<TensorEquation> equations;
    std::unordered_map<std::string, SparseBoolTensor> bool_tensors;
    std::unordered_map<std::string, DenseTensor> dense_tensors;

    void add_relation(const std::string& name, const std::vector<std::string>& indices) {
        bool_tensors.emplace(name, SparseBoolTensor(name, indices));
    }

    void add_fact(const std::string& relation, const Tuple& tuple) {
        auto it = bool_tensors.find(relation);
        if (it == bool_tensors.end()) {
            bool_tensors.emplace(relation, SparseBoolTensor(relation));
            it = bool_tensors.find(relation);
        }
        it->second.add_tuple(tuple);
    }

    void add_fact(const std::string& relation, std::initializer_list<TupleValue> values) {
        add_fact(relation, Tuple(values));
    }

    void add_tensor(const DenseTensor& t) {
        dense_tensors.emplace(t.name, t);
    }

    void add_equation(const TensorEquation& eq) {
        equations.push_back(eq);
    }

    void forward_chain(int max_iterations = 100);

    std::vector<Tuple> query(const std::string& relation) const;
    std::vector<Tuple> query(const std::string& relation, const Binding& partial) const;

    bool backward_chain(const TensorRef& goal, Binding& binding);

    std::string to_string() const;
};

class Parser {
   public:
    explicit Parser(const std::string& source) : source_(source), pos_(0) {}

    Program parse();

    std::variant<TensorEquation, std::pair<std::string, Tuple>> parse_statement();

    TensorRef parse_tensor_ref();

   private:
    std::string source_;
    size_t pos_;

    char peek() const {
        return pos_ < source_.size() ? source_[pos_] : '\0';
    }
    char advance() {
        return pos_ < source_.size() ? source_[pos_++] : '\0';
    }
    void skip_whitespace();
    void skip_comment();
    bool match(char c);
    bool match(const std::string& s);
    void expect(char c);

    std::string parse_identifier();
    Index parse_index();
    TensorEquation parse_equation();
    Nonlinearity parse_nonlinearity();
    TupleValue parse_value();
};

class REPL {
   public:
    REPL() = default;

    void run();
    void execute(const std::string& input);
    void print_help();

   private:
    Program program_;
};

}  // namespace tl

#endif  // TENSOR_LOGIC_HPP
