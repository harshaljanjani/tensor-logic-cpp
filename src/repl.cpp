/*
 * Tensor Logic - REPL
 * Commands:
 * > Type facts and rules to add them
 * > ?Relation(x, y) to query
 * > :forward to run forward chaining
 * > :show to display the program
 * > :clear to reset
 * > :help for help
 */

#include <iostream>
#include <sstream>

#include "tensor_logic.hpp"

namespace tl {

void REPL::print_help() {
    std::cout << R"(
Tensor Logic REPL
=================

Syntax:
  Facts:      Parent(Alice, Bob)          # Boolean tensor entry
  Rules:      Ancestor(x,z) <- Ancestor(x,y), Parent(y,z)
  Equations:  Y = step(W[i] X[i])         # Neural-style tensor equation

Queries:
  ?Relation(x, y)           Query all tuples matching pattern
  ?Relation(Alice, x)       Query with partial binding

Commands:
  :forward [n]              Run forward chaining (n iterations, default 100)
  :backward Goal(args)      Run backward chaining on goal
  :show                     Display current program state
  :tensors                  Show all tensors
  :clear                    Clear program
  :load <file>              Load program from file
  :example <name>           Load built-in example (ancestor, gnn, perceptron)
  :help                     Show this help
  :quit                     Exit

Examples:
  > Parent(Alice, Bob)
  > Parent(Bob, Charlie)
  > Ancestor(x, y) <- Parent(x, y)
  > Ancestor(x, z) <- Ancestor(x, y), Parent(y, z)
  > :forward
  > ?Ancestor(Alice, x)

)" << std::endl;
}

void REPL::execute(const std::string& input) {
    std::string trimmed = input;
    size_t start = trimmed.find_first_not_of(" \t\n\r");
    if (start == std::string::npos) return;
    size_t end = trimmed.find_last_not_of(" \t\n\r");
    trimmed = trimmed.substr(start, end - start + 1);

    if (trimmed.empty()) return;

    if (trimmed[0] == ':') {
        std::istringstream iss(trimmed);
        std::string cmd;
        iss >> cmd;

        if (cmd == ":help" || cmd == ":h") {
            print_help();
        } else if (cmd == ":quit" || cmd == ":q" || cmd == ":exit") {
            std::cout << "Goodbye!" << std::endl;
            exit(0);
        } else if (cmd == ":show" || cmd == ":s") {
            std::cout << program_.to_string() << std::endl;
        } else if (cmd == ":clear") {
            program_ = Program();
            std::cout << "Program cleared." << std::endl;
        } else if (cmd == ":forward" || cmd == ":f") {
            int n = 100;
            iss >> n;
            std::cout << "Running forward chaining (" << n << " max iterations)..." << std::endl;
            program_.forward_chain(n);
            std::cout << "Done." << std::endl;
        } else if (cmd == ":tensors" || cmd == ":t") {
            std::cout << "Boolean Tensors:" << std::endl;
            for (const auto& [name, tensor] : program_.bool_tensors) {
                std::cout << "  " << tensor.to_string() << std::endl;
            }
            std::cout << "Dense Tensors:" << std::endl;
            for (const auto& [name, tensor] : program_.dense_tensors) {
                std::cout << "  " << tensor.to_string() << std::endl;
            }
        } else if (cmd == ":example") {
            std::string name;
            iss >> name;

            if (name == "ancestor") {
                program_ = Program();
                // Ancestor example from paper
                program_.add_fact("Parent", {"Alice", "Bob"});
                program_.add_fact("Parent", {"Bob", "Charlie"});
                program_.add_fact("Parent", {"Charlie", "David"});

                TensorEquation rule1;
                rule1.lhs = TensorRef("Ancestor", {Index("x", true), Index("y", true)}, true);
                rule1.rhs.push_back(TensorRef("Parent", {Index("x", true), Index("y", true)}, true));
                rule1.nonlinearity = Nonlinearity::Step;
                program_.add_equation(rule1);

                TensorEquation rule2;
                rule2.lhs = TensorRef("Ancestor", {Index("x", true), Index("z", true)}, true);
                rule2.rhs.push_back(TensorRef("Ancestor", {Index("x", true), Index("y", true)}, true));
                rule2.rhs.push_back(TensorRef("Parent", {Index("y", true), Index("z", true)}, true));
                rule2.nonlinearity = Nonlinearity::Step;
                program_.add_equation(rule2);

                std::cout << "Loaded ancestor example. Use :forward then ?Ancestor(x, y)" << std::endl;
            } else if (name == "aunt") {
                program_ = Program();
                // Aunt example (Axz = H(Sxy Pyz))
                program_.add_fact("Sister", {"Alice", "Bob"});
                program_.add_fact("Parent", {"Bob", "Charlie"});

                TensorEquation rule;
                rule.lhs = TensorRef("Aunt", {Index("x", true), Index("z", true)}, true);
                rule.rhs.push_back(TensorRef("Sister", {Index("x", true), Index("y", true)}, true));
                rule.rhs.push_back(TensorRef("Parent", {Index("y", true), Index("z", true)}, true));
                rule.nonlinearity = Nonlinearity::Step;
                program_.add_equation(rule);

                std::cout << "Loaded aunt example. Use :forward then ?Aunt(x, y)" << std::endl;
            } else if (name == "perceptron") {
                program_ = Program();
                // Y = step(W[i] X[i])
                DenseTensor W("W", {0.5, 1.0, -0.5, 2.0});
                DenseTensor X("X", {1.0, 0.0, 1.0, 1.0});
                program_.add_tensor(W);
                program_.add_tensor(X);

                std::cout << "Loaded perceptron example with W and X tensors." << std::endl;
                std::cout << "  W = [0.5, 1.0, -0.5, 2.0]" << std::endl;
                std::cout << "  X = [1.0, 0.0, 1.0, 1.0]" << std::endl;
                std::cout << "  W·X = " << (0.5 * 1 + 1.0 * 0 + -0.5 * 1 + 2.0 * 1) << std::endl;
            } else {
                std::cout << "Unknown example: " << name << std::endl;
                std::cout << "Available: ancestor, aunt, perceptron" << std::endl;
            }
        } else {
            std::cout << "Unknown command: " << cmd << std::endl;
        }
        return;
    }

    if (trimmed[0] == '?') {
        std::string query_str = trimmed.substr(1);
        Parser parser(query_str);
        try {
            TensorRef query = parser.parse_tensor_ref();

            Binding partial;
            std::vector<std::string> var_names;
            for (const auto& idx : query.indices) {
                if (idx.is_variable) {
                    var_names.push_back(idx.name);
                } else {
                }
            }

            auto results = program_.query(query.name);

            std::vector<Tuple> filtered;
            for (const auto& tuple : results) {
                bool matches = true;
                for (size_t i = 0; i < query.indices.size() && i < tuple.size(); ++i) {
                    if (!query.indices[i].is_variable) {
                        TupleValue const_val;
                        try {
                            const_val = std::stoi(query.indices[i].name);
                        } catch (...) {
                            const_val = query.indices[i].name;
                        }
                        if (tuple.values[i] != const_val) {
                            matches = false;
                            break;
                        }
                    }
                }
                if (matches) {
                    filtered.push_back(tuple);
                }
            }

            if (filtered.empty()) {
                std::cout << "No results found." << std::endl;
            } else {
                std::cout << "Results (" << filtered.size() << "):" << std::endl;
                for (const auto& tuple : filtered) {
                    std::cout << "  " << query.name << tuple.to_string() << std::endl;
                }
            }
        } catch (const std::exception& e) {
            std::cout << "Query error: " << e.what() << std::endl;
        }
        return;
    }

    try {
        Parser parser(trimmed);
        auto result = parser.parse_statement();

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
                program_.add_fact(eq.lhs.name, tuple);
                std::cout << "Added fact: " << eq.lhs.name << tuple.to_string() << std::endl;
            } else {
                program_.add_equation(eq);
                std::cout << "Added equation: " << eq.to_string() << std::endl;
            }
        } else {
            auto [name, tuple] = std::get<std::pair<std::string, Tuple>>(result);
            program_.add_fact(name, tuple);
            std::cout << "Added fact: " << name << tuple.to_string() << std::endl;
        }
    } catch (const std::exception& e) {
        std::cout << "Parse error: " << e.what() << std::endl;
    }
}

void REPL::run() {
    std::cout << "Tensor Logic - The Language of AI" << std::endl;
    std::cout << "Type :help for help, :quit to exit" << std::endl;
    std::cout << std::endl;

    std::string line;
    while (true) {
        std::cout << "> ";
        if (!std::getline(std::cin, line)) {
            break;
        }
        execute(line);
    }
}

}  // namespace tl
