### Tensor Logic: The Language of AI

A production-ready C++ implementation of Tensor Logic, based on the research of Pedro Domingos ([arXiv:2510.12269v3](https://arxiv.org/abs/2510.12269)). This project provides a unified framework for neural and symbolic AI by representing logical relations as sparse Boolean tensors and expressing rules as tensor operations.

Tensor Logic integrates deductive and analogical reasoning through the following core mechanisms:<br>
**Sparse Boolean Tensors:** Relations are represented as sparse Boolean tensors where a relation `R(x, y)` acts as a Boolean matrix with `R[x,y] = 1` if the tuple `(x,y)` is in the relation. This allows logical joins and projections to be executed as tensor contractions and summations.<br>
**Einsum-Based Rules:** Datalog rules are implemented as Einstein summation (einsum) operations combined with step functions. For example, the rule `Aunt(x,z) <- Sister(x,y), Parent(y,z)` is equivalent to `A[x,z] = H(S[x,y] * P[y,z])` where H is the Heaviside step function, effectively performing a matrix multiplication followed by step function activation.<br>
**Join and Projection Operations:** Core tensor operations follow database semantics. Join is defined as `(U ⋊⋉ V)[α,β,γ] = U[α,β] * V[β,γ]` using einsum with common indices. Projection is `π_α(T) = Σ_β T[α,β]`, summing over non-retained indices. For Boolean tensors, these exactly match database natural join and projection operations.<br>
**Temperature-Controlled Reasoning:** The system implements a variable temperature parameter (T) to control the nature of inference using `σ(x, T) = 1 / (1 + e^(-x/T))`:<br>
    **→ Deductive (T = 0):** At zero temperature, `σ(x,0)` becomes a step function, ensuring purely deductive reasoning with zero hallucination. As stated in the paper: "Setting temperature T to 0 effectively reduces the Gram matrix to the identity matrix, making reasoning purely deductive. This contrasts with LLMs, which may hallucinate even at T = 0."<br>
    **→ Analogical (T > 0):** At positive temperatures, `σ(x,T)` is soft, allowing objects with similar embeddings to share inferences through analogical reasoning.<br>
**Tensor Decomposition:** Supports Tucker decomposition `A[i, j, k] = M[i, p] M'[j, q] M''[k, r] C[p, q, r]` to convert sparse tensors into efficient dense operations, enabling scalability on hardware accelerators like GPUs.

### Setup

**Clone the Repository:** Clone the source code to your local machine.<br>
**Build the Project:** The project includes a Makefile for compilation. Run the `make` command in the root directory. This will generate two binaries:<br>
**→ tl:** The interactive interpreter (REPL).<br>
**→ tl-examples:** A suite of examples taken directly from the paper, demonstrating the concepts presented.

### Quick Start

**Run All Tests:** Execute `./tl --test` to run the complete test suite.<br>
**Run Extended Examples:** The `tl-examples` binary demonstrates key concepts from the paper. Run all examples with `./tl-examples`, or specific examples: `./tl-examples temp` for temperature-controlled reasoning (T = 0 vs T > 0), `./tl-examples deductive` for deductive chain tracing, `./tl-examples transformer` for transformer architecture, `./tl-examples gnn` for graph neural network message passing, `./tl-examples kernel` for kernel machines, `./tl-examples embed` for embedding relations, or `./tl-examples tucker` for Tucker decomposition.<br>
**Interactive REPL:** Launch the interactive interpreter with `./tl` for direct experimentation with tensor logic programs.

### Key Examples from the Paper

**Temperature-Controlled Reasoning (Section 5):** At T = 0, reasoning is purely deductive with no hallucinations. Queries like `CanFly(Eagle, True)?` return 1.0 for facts in the knowledge base, while `CanFly(Ostrich, True)?` returns 0.0 with no hallucination. At T > 0, similar objects borrow inferences through analogical reasoning. With `similarity(Eagle, Ostrich) = 0.97`, the query `CanFly(Ostrich, True)?` returns 0.74, borrowed from the similar Eagle.<br>
**Transformer Architecture (Table 2):** The entire transformer is expressed as tensor equations including embedding `EmbX[p, d] = X(p, t) Emb[t, d]`, attention operations with query/key/value projections and softmax compatibility scores, and output projection `Y[p, t.] = softmax(WOut[t, d] Stream[B, p, d])`.<br>
**Graph Neural Networks (Table 1):** Message passing is implemented as tensor join: `Agg[n, l, d] = Neig(n, n') Z[n', l, d]`, which joins the neighborhood relation with node features, summing neighbor features for each node.<br>
**Embedding Relations (Section 5):** Relations are embedded as tensor products: `EmbR[i, j] = R(x, y) Emb[x, i] Emb[y, j]`, with distance computation `D[A, B] = EmbR[i, j] Emb[A, i] Emb[B, j]` approximating 1 if `(A,B) ∈ R`.

### REPL Usage

The interactive interpreter supports direct entry of facts, rules, and queries. Example session demonstrating transitive closure:<br>
Define facts: `Parent(Alice, Bob)` and `Parent(Bob, Charlie)`.<br>
Add rules: `Ancestor(x, y) <- Parent(x, y)` for base case and `Ancestor(x, z) <- Ancestor(x, y), Parent(y, z)` for recursive case.<br>
Execute forward chaining: `:forward` to derive all consequences.<br>
Query results: `?Ancestor(Alice, x)` returns both `Ancestor(Alice, Bob)` and `Ancestor(Alice, Charlie)`.

**Commands:** `:forward [n]` runs forward chaining with optional iteration limit (default 100), `:show` displays program state, `:tensors` shows all tensors, `:clear` resets the program, `:example name` loads predefined examples (ancestor, aunt, perceptron), `:help` shows help information, and `:quit` exits the REPL.

**License:** Apache-2.0

**Citation:** If you use this software, please cite it using the metadata provided in `CITATION.cff`.
