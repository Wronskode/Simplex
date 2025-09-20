# Simplex and Branch & Bound Solver

A lightweight linear and integer programming solver written in **Rust**.
It implements the **Simplex algorithm** (Big‑M and Two‑Phase methods) and a **Branch‑and‑Bound** procedure for Integer Linear Programming (ILP).

This project was built as an educational solver to understand how commercial solvers (e.g. Hexaly, Gurobi, CPLEX) work under the hood and to practise numerical/algorithmic engineering.

---

## 🚀 Features
- Linear programming solver using the **Simplex method**
  - Big‑M method
  - Two‑Phase method
- Integer programming solver using **Branch‑and‑Bound**
- Command‑line interface (CLI) to solve `.lp` files
- HTTP server mode with simple REST API endpoints

---

## 📦 Installation
Clone the repository and build with Cargo:

```bash
git clone https://github.com/Wronskode/Simplex.git
cd Simplex
cargo build --release
```

---

## 🖥️ Usage

### Command‑line mode
Run the solver on an `.lp` file:

```bash
cargo run --release instances/35queens.lp
```

- Solves the linear program in `35queens.lp`.
- If integer variables are detected, **Branch‑and‑Bound** is automatically applied.

<!-- > Note: the `--` separates cargo arguments from your program arguments; adjust if you use a prebuilt binary. -->

### Server mode
Start the solver in server mode:

```bash
cargo run --release server
```

- Server listens on `0.0.0.0:8888`.
- Provides these POST routes (multipart/form-data or raw body):
  - `/simplex` → solve using Simplex
  - `/branch_and_bound` → solve using Branch‑and‑Bound

Each route accepts an `.lp` file upload and returns a JSON with the solution (objective, variable values, status, runtime). In each cases, the program will use Two Phases algorithm instead Big M method due to a highest execution speed in Two Phases algorithm.

---

## 📊 Example: N‑Queens benchmark
Execution time of Branch‑and‑Bound on the **N‑Queens problem** (8 ≤ n ≤ 40, timeout = 70s):

![nqueens benchmark](https://github.com/Wronskode/Simplex/blob/main/nqueens.png)

---

## ⚠️ Known limitations
- Works well on small and medium instances.
- On larger or badly conditioned instances, numerical instability can appear due to floating‑point (`f64`) errors and matrix condition numbers.
- Memory: current implementation uses a dense matrix representation; for big sparse instances the memory footprint can explode and cause allocation (malloc) errors.
- No constraint scaling, advanced numerical pivoting, or exact arithmetic implemented yet.
- Branch‑and‑Bound currently uses simple branching and no advanced heuristics (no strong branching, no sophisticated node selection).

---

## 🔧 Implementation notes
- Language: **Rust** — chosen for safety and to practice robust, modular code.
- Parser: Pest.rs used for accessibility, correctness and performance `https://pest.rs/`.
- Benchmarks: simple script in `nqueens_curve.py` to reproduce N‑Queens timings. (and `nqueens_lp_generator.py` to create instances).

---

## 🔮 Future work
- Change the system representation to avoid sparse matrix.
- Add **constraint scaling** and improved numerical pivot selection.
- Integrate **exact/rational arithmetic** (e.g. via GMP/MPFR) for numerically challenging instances.
- Rewrite B&B to avoid parsing at every node.
- Implement advanced B&B heuristics: strong branching, best‑bound node selection, primal heuristics.
- Add cutting‑plane methods (Gomory cuts) and internal presolve routines.
- Optionally reimplement performance‑critical parts in **C++** or provide bindings to mature libraries.

---

## 🧾 Example output (format)
```
Actual values of the variables:
x1 = 0
x2 = 12
Value of objective function: 24
Time taken: 3.9745ms
```

---

## 📚 References
- G. Dantzig, *Linear Programming and Extensions*
- M. Grötschel, L. Lovász, A. Schrijver, *Geometric Algorithms and Combinatorial Optimization*
- Resources on numerical stability and pivoting strategies.
