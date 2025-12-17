/**
>   tl              >> start REPL
>   tl <file>       >> execute file
>   tl -e "code"    >> execute code string
>   tl --test       >> run tests
*/
#include <cstring>
#include <fstream>
#include <sstream>

#include "tensor_logic.hpp"

void run_tests();

int main(int argc, char* argv[]) {
    if (argc == 1) {
        tl::REPL repl;
        repl.run();
        return 0;
    }

    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "--test") == 0 || strcmp(argv[i], "-t") == 0) {
            run_tests();
            return 0;
        } else if (strcmp(argv[i], "-e") == 0) {
            if (i + 1 < argc) {
                tl::Parser parser(argv[i + 1]);
                auto prog = parser.parse();
                prog.forward_chain();
                std::cout << prog.to_string() << std::endl;
                ++i;
            } else {
                std::cerr << "Error: -e requires an argument" << std::endl;
                return 1;
            }
        } else if (strcmp(argv[i], "--help") == 0 || strcmp(argv[i], "-h") == 0) {
            std::cout << "Tensor Logic - The Language of AI\n\n";
            std::cout << "Usage:\n";
            std::cout << ">  tl              Start interactive REPL\n";
            std::cout << ">  tl <file>       Execute tensor logic file\n";
            std::cout << ">  tl -e \"code\"  Execute code string\n";
            std::cout << ">  tl --test       Run built-in tests\n";
            std::cout << ">  tl --help       Show this help\n";
            return 0;
        } else {
            std::ifstream file(argv[i]);
            if (!file) {
                std::cerr << "Error: Could not open file: " << argv[i] << std::endl;
                return 1;
            }
            std::stringstream buffer;
            buffer << file.rdbuf();

            tl::Parser parser(buffer.str());
            auto prog = parser.parse();
            prog.forward_chain();
            std::cout << prog.to_string() << std::endl;
        }
    }

    return 0;
}

void test_ancestor() {
    std::cout << "Test: Ancestor Example from Paper\n";
    std::cout << "=================================\n";

    tl::Program prog;

    prog.add_fact("Parent", {"Alice", "Bob"});
    prog.add_fact("Parent", {"Bob", "Charlie"});
    prog.add_fact("Parent", {"Charlie", "David"});

    // Ancestor(x, y) <- Parent(x, y)
    // Ancestor(x, z) <- Ancestor(x, y), Parent(y, z)
    tl::TensorEquation rule1;
    rule1.lhs = tl::TensorRef("Ancestor", {tl::Index("x", true), tl::Index("y", true)}, true);
    rule1.rhs.push_back(tl::TensorRef("Parent", {tl::Index("x", true), tl::Index("y", true)}, true));
    rule1.nonlinearity = tl::Nonlinearity::Step;
    prog.add_equation(rule1);

    tl::TensorEquation rule2;
    rule2.lhs = tl::TensorRef("Ancestor", {tl::Index("x", true), tl::Index("z", true)}, true);
    rule2.rhs.push_back(tl::TensorRef("Ancestor", {tl::Index("x", true), tl::Index("y", true)}, true));
    rule2.rhs.push_back(tl::TensorRef("Parent", {tl::Index("y", true), tl::Index("z", true)}, true));
    rule2.nonlinearity = tl::Nonlinearity::Step;
    prog.add_equation(rule2);

    std::cout << "Before forward chaining:\n" << prog.to_string() << "\n";

    prog.forward_chain();

    std::cout << "After forward chaining:\n" << prog.to_string() << "\n";

    auto results = prog.query("Ancestor");
    std::cout << "Query: Ancestor(x, y)\n";
    for (const auto& t : results) {
        std::cout << "  Ancestor" << t.to_string() << "\n";
    }

    bool passed = true;
    auto& tensor = prog.bool_tensors["Ancestor"];

    if (!tensor.contains(tl::Tuple{{"Alice", "Bob"}})) {
        passed = false;
        std::cout << "FAIL: Alice->Bob\n";
    }
    if (!tensor.contains(tl::Tuple{{"Alice", "Charlie"}})) {
        passed = false;
        std::cout << "FAIL: Alice->Charlie\n";
    }
    if (!tensor.contains(tl::Tuple{{"Alice", "David"}})) {
        passed = false;
        std::cout << "FAIL: Alice->David\n";
    }
    if (!tensor.contains(tl::Tuple{{"Bob", "Charlie"}})) {
        passed = false;
        std::cout << "FAIL: Bob->Charlie\n";
    }
    if (!tensor.contains(tl::Tuple{{"Bob", "David"}})) {
        passed = false;
        std::cout << "FAIL: Bob->David\n";
    }
    if (!tensor.contains(tl::Tuple{{"Charlie", "David"}})) {
        passed = false;
        std::cout << "FAIL: Charlie->David\n";
    }

    std::cout << (passed ? "PASSED" : "FAILED") << "\n\n";
}

void test_aunt() {
    std::cout << "Test: Aunt Example from Paper\n";
    std::cout << "=============================\n";
    std::cout << "Axz = H(Sxy * Pyz)  [Aunt is sister of parent]\n\n";

    tl::Program prog;

    prog.add_fact("Sister", {"Alice", "Bob"});
    prog.add_fact("Sister", {"Carol", "Bob"});
    prog.add_fact("Parent", {"Bob", "Charlie"});
    prog.add_fact("Parent", {"Bob", "Diana"});

    tl::TensorEquation rule;
    rule.lhs = tl::TensorRef("Aunt", {tl::Index("x", true), tl::Index("z", true)}, true);
    rule.rhs.push_back(tl::TensorRef("Sister", {tl::Index("x", true), tl::Index("y", true)}, true));
    rule.rhs.push_back(tl::TensorRef("Parent", {tl::Index("y", true), tl::Index("z", true)}, true));
    rule.nonlinearity = tl::Nonlinearity::Step;
    prog.add_equation(rule);

    prog.forward_chain();

    std::cout << "After forward chaining:\n" << prog.to_string() << "\n";

    auto& tensor = prog.bool_tensors["Aunt"];
    bool passed = tensor.size() == 4;
    passed = passed && tensor.contains(tl::Tuple{{"Alice", "Charlie"}});
    passed = passed && tensor.contains(tl::Tuple{{"Alice", "Diana"}});
    passed = passed && tensor.contains(tl::Tuple{{"Carol", "Charlie"}});
    passed = passed && tensor.contains(tl::Tuple{{"Carol", "Diana"}});

    std::cout << (passed ? "PASSED" : "FAILED") << "\n\n";
}

void test_join() {
    std::cout << "Test: Tensor Join Operation\n";
    std::cout << "===========================\n";

    tl::SparseBoolTensor U("U", {"a", "b"});
    U.add_tuple({1, "x"});
    U.add_tuple({2, "x"});
    U.add_tuple({2, "y"});

    tl::SparseBoolTensor V("V", {"b", "c"});
    V.add_tuple({"x", 10});
    V.add_tuple({"y", 20});
    V.add_tuple({"z", 30});

    std::cout << "U:\n" << U.to_string() << "\n";
    std::cout << "V:\n" << V.to_string() << "\n";

    auto result = tl::SparseBoolTensor::join(U, V);
    std::cout << "U ⋊⋉ V (join on b):\n" << result.to_string() << "\n";

    // (1, x, 10), (2, x, 10), (2, y, 20)
    bool passed = result.size() == 3;
    std::cout << (passed ? "PASSED" : "FAILED") << "\n\n";
}

void test_projection() {
    std::cout << "Test: Tensor Projection Operation\n";
    std::cout << "=================================\n";

    tl::SparseBoolTensor T("T", {"x", "y", "z"});
    T.add_tuple({1, "a", 100});
    T.add_tuple({1, "b", 100});
    T.add_tuple({2, "a", 200});

    std::cout << "T:\n" << T.to_string() << "\n";

    auto proj_xz = T.project({"x", "z"});
    std::cout << "π_{x,z}(T):\n" << proj_xz.to_string() << "\n";

    // (1, 100), (2, 200)
    bool passed = proj_xz.size() == 2;

    auto proj_x = T.project({"x"});
    std::cout << "π_{x}(T):\n" << proj_x.to_string() << "\n";

    // (1), (2)
    passed = passed && proj_x.size() == 2;

    std::cout << (passed ? "PASSED" : "FAILED") << "\n\n";
}

void test_parser() {
    std::cout << "Test: Parser\n";
    std::cout << "============\n";

    std::string source = R"(
        Parent(Alice, Bob)
        Parent(Bob, Charlie)
        Ancestor(x, y) <- Parent(x, y)
        Ancestor(x, z) <- Ancestor(x, y), Parent(y, z)
    )";

    tl::Parser parser(source);
    auto prog = parser.parse();

    std::cout << "Parsed program:\n" << prog.to_string() << "\n";

    prog.forward_chain();
    std::cout << "After forward chaining:\n" << prog.to_string() << "\n";

    bool passed = prog.bool_tensors["Ancestor"].size() == 3;
    std::cout << (passed ? "PASSED" : "FAILED") << "\n\n";
}

void test_einsum() {
    std::cout << "Test: Einsum (Dense Tensor)\n";
    std::cout << "===========================\n";
    std::cout << "Testing: Y = W[i] X[i] (dot product)\n\n";

    // Y = W[i] X[i]
    tl::DenseTensor W("W", {"i"}, {4});
    W.data = {1.0, 2.0, 3.0, 4.0};

    tl::DenseTensor X("X", {"i"}, {4});
    X.data = {1.0, 0.0, 1.0, 0.5};

    std::cout << "W: " << W.to_string() << "\n";
    std::cout << "X: " << X.to_string() << "\n";
    std::cout << "\nTesting: Y[i] = M[i,j] X[j] (matrix-vector product)\n\n";

    tl::DenseTensor M("M", {"i", "j"}, {2, 3});
    M.data = {1.0, 2.0, 3.0,   // row 0
              4.0, 5.0, 6.0};  // row 1

    tl::DenseTensor X2("X", {"j"}, {3});
    X2.data = {1.0, 0.0, 2.0};

    std::cout << "M:\n" << M.to_string() << "\n";
    std::cout << "X: " << X2.to_string() << "\n";

    auto Y = tl::DenseTensor::einsum(M, X2, {"i"});
    std::cout << "Y = M[i,j] X[j]:\n" << Y.to_string() << "\n";

    // [1*1 + 2*0 + 3*2, 4*1 + 5*0 + 6*2] = [7, 16]
    bool passed = std::abs(Y.data[0] - 7.0) < 0.001 && std::abs(Y.data[1] - 16.0) < 0.001;
    std::cout << (passed ? "PASSED" : "FAILED") << "\n\n";
}

void run_tests() {
    std::cout << "=======================\n";
    std::cout << "Tensor Logic Test Suite\n";
    std::cout << "=======================\n\n";

    test_join();
    test_projection();
    test_ancestor();
    test_aunt();
    test_parser();
    test_einsum();

    std::cout << "==================\n";
    std::cout << "All tests complete\n";
    std::cout << "==================\n";
}
