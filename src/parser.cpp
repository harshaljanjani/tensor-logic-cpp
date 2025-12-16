#include <cctype>
#include <stdexcept>

#include "tensor_logic.hpp"

namespace tl {

void Parser::skip_whitespace() {
    while (pos_ < source_.size() && (std::isspace(source_[pos_]) || source_[pos_] == '\n' || source_[pos_] == '\r')) {
        ++pos_;
    }
    skip_comment();
}

void Parser::skip_comment() {
    // "//" comments
    if (pos_ < source_.size() - 1 && source_[pos_] == '/' && source_[pos_ + 1] == '/') {
        while (pos_ < source_.size() && source_[pos_] != '\n') {
            ++pos_;
        }
        skip_whitespace();
    }
    // "#" comments
    if (pos_ < source_.size() && source_[pos_] == '#') {
        while (pos_ < source_.size() && source_[pos_] != '\n') {
            ++pos_;
        }
        skip_whitespace();
    }
}

bool Parser::match(char c) {
    skip_whitespace();
    if (peek() == c) {
        advance();
        return true;
    }
    return false;
}

bool Parser::match(const std::string& s) {
    skip_whitespace();
    if (pos_ + s.size() <= source_.size()) {
        for (size_t i = 0; i < s.size(); ++i) {
            if (source_[pos_ + i] != s[i]) {
                return false;
            }
        }
        pos_ += s.size();
        return true;
    }
    return false;
}

void Parser::expect(char c) {
    skip_whitespace();
    if (peek() != c) {
        throw std::runtime_error(std::string("Expected '") + c + "' but got '" + peek() + "'");
    }
    advance();
}

std::string Parser::parse_identifier() {
    skip_whitespace();
    std::string result;

    if (!std::isalpha(peek()) && peek() != '_') {
        throw std::runtime_error("Expected identifier");
    }

    while (std::isalnum(peek()) || peek() == '_') {
        result += advance();
    }

    return result;
}

Index Parser::parse_index() {
    skip_whitespace();
    std::string name;
    int offset = 0;

    bool negative = false;
    if (peek() == '-') {
        advance();
        negative = true;
    }

    if (std::isdigit(peek())) {
        while (std::isdigit(peek())) {
            name += advance();
        }
        int val = std::stoi(name);
        if (negative) val = -val;
        return Index(std::to_string(val), false);
    }

    name = parse_identifier();

    skip_whitespace();
    if (peek() == '+') {
        advance();
        skip_whitespace();
        std::string offset_str;
        while (std::isdigit(peek())) {
            offset_str += advance();
        }
        offset = std::stoi(offset_str);
    } else if (peek() == '-') {
        advance();
        skip_whitespace();
        std::string offset_str;
        while (std::isdigit(peek())) {
            offset_str += advance();
        }
        offset = -std::stoi(offset_str);
    }

    bool is_var = !name.empty() && std::islower(name[0]);

    return Index(name, is_var, offset);
}

Nonlinearity Parser::parse_nonlinearity() {
    skip_whitespace();
    size_t start = pos_;

    if (match("step")) return Nonlinearity::Step;
    if (match("sig")) return Nonlinearity::Sigmoid;
    if (match("sigmoid")) return Nonlinearity::Sigmoid;
    if (match("relu")) return Nonlinearity::ReLU;
    if (match("softmax")) return Nonlinearity::Softmax;
    if (match("tanh")) return Nonlinearity::Tanh;
    if (match("exp")) return Nonlinearity::Exp;
    if (match("log")) return Nonlinearity::Log;
    if (match("sqrt")) return Nonlinearity::Sqrt;
    if (match("H")) return Nonlinearity::Step;

    pos_ = start;
    return Nonlinearity::None;
}

TensorRef Parser::parse_tensor_ref() {
    skip_whitespace();

    std::string name = parse_identifier();
    std::vector<Index> indices;
    bool is_boolean = false;

    skip_whitespace();

    if (peek() == '(') {
        is_boolean = true;
        advance();

        while (peek() != ')') {
            indices.push_back(parse_index());
            skip_whitespace();
            if (peek() == ',') {
                advance();
            }
        }
        expect(')');
    } else if (peek() == '[') {
        advance();

        while (peek() != ']') {
            indices.push_back(parse_index());
            skip_whitespace();
            if (peek() == ',') {
                advance();
            }
        }
        expect(']');
    }

    return TensorRef(name, indices, is_boolean);
}

TupleValue Parser::parse_value() {
    skip_whitespace();

    bool negative = false;
    if (peek() == '-') {
        advance();
        negative = true;
    }

    if (std::isdigit(peek())) {
        std::string num;
        while (std::isdigit(peek())) {
            num += advance();
        }
        int val = std::stoi(num);
        return negative ? -val : val;
    }

    std::string str = parse_identifier();
    return str;
}

TensorEquation Parser::parse_equation() {
    TensorEquation eq;

    eq.lhs = parse_tensor_ref();

    skip_whitespace();

    if (match("<-") || match("←") || match(":-")) {
        eq.lhs.is_boolean = true;
        eq.nonlinearity = Nonlinearity::Step;

        do {
            skip_whitespace();
            eq.rhs.push_back(parse_tensor_ref());
            eq.rhs.back().is_boolean = true;
            skip_whitespace();
        } while (match(','));

    } else if (match("=")) {
        skip_whitespace();

        Nonlinearity nl = parse_nonlinearity();
        if (nl != Nonlinearity::None) {
            eq.nonlinearity = nl;
            expect('(');
        }

        while (peek() != ')' && peek() != '\0' && peek() != '\n' && peek() != ';') {
            skip_whitespace();
            if (peek() == ')' || peek() == '\0' || peek() == '\n' || peek() == ';') break;
            eq.rhs.push_back(parse_tensor_ref());
            skip_whitespace();

            if (peek() == '*') advance();
        }

        if (nl != Nonlinearity::None) {
            expect(')');
        }
    }

    return eq;
}

std::variant<TensorEquation, std::pair<std::string, Tuple>> Parser::parse_statement() {
    skip_whitespace();

    TensorRef first = parse_tensor_ref();

    skip_whitespace();

    if (peek() == '=' || peek() == '<' || peek() == ':' || (pos_ + 1 < source_.size() && source_[pos_] == '\xe2')) {
        pos_ = 0;
        return parse_equation();
    } else if (peek() == '\0' || peek() == '\n' || peek() == '.' || peek() == ';' || peek() == ',') {
        Tuple tuple;
        for (const auto& idx : first.indices) {
            if (idx.is_variable) {
                throw std::runtime_error("Facts cannot contain variables: " + idx.name);
            }
            try {
                tuple.values.push_back(std::stoi(idx.name));
            } catch (...) {
                tuple.values.push_back(idx.name);
            }
        }
        return std::make_pair(first.name, tuple);
    }

    throw std::runtime_error("Unexpected token in statement");
}

Program Parser::parse() {
    Program prog;

    while (pos_ < source_.size()) {
        skip_whitespace();
        if (pos_ >= source_.size()) break;

        size_t start = pos_;

        while (pos_ < source_.size() && peek() != '\n' && peek() != '.' && peek() != ';') {
            ++pos_;
        }

        std::string stmt_source = source_.substr(start, pos_ - start);
        if (peek() == '.' || peek() == ';') advance();

        bool all_whitespace = true;
        for (char c : stmt_source) {
            if (!std::isspace(c)) {
                all_whitespace = false;
                break;
            }
        }
        if (all_whitespace) continue;

        Parser stmt_parser(stmt_source);
        auto result = stmt_parser.parse_statement();

        if (std::holds_alternative<TensorEquation>(result)) {
            auto eq = std::get<TensorEquation>(result);
            if (eq.rhs.empty() && eq.lhs.is_boolean) {
                Tuple tuple;
                for (const auto& idx : eq.lhs.indices) {
                    try {
                        tuple.values.push_back(std::stoi(idx.name));
                    } catch (...) {
                        tuple.values.push_back(idx.name);
                    }
                }
                prog.add_fact(eq.lhs.name, tuple);
            } else {
                prog.add_equation(eq);
            }
        } else {
            auto [name, tuple] = std::get<std::pair<std::string, Tuple>>(result);
            prog.add_fact(name, tuple);
        }
    }

    return prog;
}

}  // namespace tl
