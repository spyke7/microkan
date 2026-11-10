# microkan

A **Kolmogorov–Arnold Network (KAN)** written from scratch in C++, featuring a tiny reverse-mode automatic differentiation engine inspired by Andrej Karpathy's
[micrograd](https://github.com/karpathy/micrograd) and hand-written B-splines.

No ML libraries, no Eigen, no PyTorch: just the standard library.

Built to explore the mechanics of KANs, computational graphs, automatic differentiation, and spline-based function approximation at a low level.

---

## Table of contents

1. [The idea](#1-the-idea)
2. [Mathematics](#2-mathematics)
3. [Design: Neuron, NeuronPool, local_grad](#3-design-neuron-neuronpool-local_grad)
4. [Docs](#5-docs)
5. [Visualization and Flow](#5-visualization-and-flow)
6. [Benchmark vs micrograd](#6-benchmark-vs-micrograd)
7. [How to use?](#7-how-to-use?)
8. [Known limitations and bugs](#8-known-limitations-and-bugs)
9. [References](#9-references)

---


## 1. The idea 

A normal neural network (MLP) puts **fixed** activation functions on the **nodes** and
**learnable numbers** (weights) on the **edges**:

```
MLP:   x --w--> [ReLU] --w--> [ReLU] --w--> y        weights = numbers, activations = fixed
```

A KAN flips this. Every **edge** carries a **learnable function** φ, and the nodes just **add**:

```
KAN:   x₁ ──φ₁₀──▶ ┐
                    (+) = h₀ ──Φ₀──▶ ┐
       x₂ ──φ₂₀──▶ ┘                  │
        ⋮                             (+) ──▶ y
       x₁ ──φ₁₄──▶ ┐                  │
                    (+) = h₄ ──Φ₄──▶ ┘
       x₂ ──φ₂₄──▶ ┘
```

Each learnable function is a **B-spline** (a smooth piecewise polynomial whose shape is set by a
handful of coefficients) plus a SiLU "residual" term. Learning means moving those coefficients.

The architecture is the one suggested by the **Kolmogorov–Arnold representation theorem**:
any continuous function of `n` inputs can be written using only sums and single-variable functions.
This repo implements the 2-layer shape **[2, 5, 1]**: 2 inputs, `2n+1 = 5` hidden nodes, 1 output.

---

## 2. Mathematics

Notation used below (and in the code):

| Symbol | Meaning |
|---|---|
| `n` | number of inputs |
| `m = 2n+1` | number of hidden nodes |
| `G` | number of grid intervals |
| `k` | spline degree (order) |
| `p = G + k` | number of basis functions per spline |
| `[a, b]` | grid range |
| `h = (b-a)/G` | grid spacing |

### Kolmogorov–Arnold representation theorem

Every continuous `f : [0,1]ⁿ → ℝ` can be written as

$$
f(\mathbf{x}) = f(x_1,\dots,x_n) = \sum_{q=1}^{2n+1} \Phi_q\!\left(\sum_{p=1}^{n} \varphi_{q,p}(x_p)\right)
$$,

with continuous one-variable functions `φ_{q,p}` and `Φ_q`. A KAN **learns** those functions.
(In the code, indices are 0-based: `p ∈ {0,1}`, `q ∈ {0,…,4}`.)

### Knot vector

For grid `[a, b]` with `G` intervals and degree `k`, spacing `h = (b-a)/G`, the code builds
`G + 2k + 1` equally spaced knots, extended `k` steps beyond both ends:

$$
t_i = a + (i-k)\,h, \qquad i = 0,1,\dots,G+2k
$$

*Here,'a' is basically taken as the minimum of the params and 'b' is the maximum of the params.*

(`Spline::Spline` in `spline.hpp`, vector `t`.)

### B-spline basis: Cox–de Boor recursion

Degree 0 (indicator of one knot interval):

$$
B_{j,0}(x)=\begin{cases}1 & t_j \le x < t_{j+1}\\ 0 & \text{otherwise}\end{cases}
$$

Degree `p ≥ 1`:

$$
B_{j,p}(x)=\frac{x-t_j}{t_{j+p}-t_j}\,B_{j,p-1}(x)\;+\;\frac{t_{j+p+1}-x}{t_{j+p+1}-t_{j+1}}\,B_{j+1,p-1}(x)
$$

(`Spline::coxDeBoor(j, p, x)`.) We use `j = 0,…,G+k-1`, that is `K` basis functions of degree `k`.
Properties used implicitly: each `B_{j,k}` is non-negative, supported on `[t_j, t_{j+k+1})`, and on
`[a, b)` the `K` functions sum to 1 (partition of unity).

### Derivative of a basis function

$$
B_{j,p}'(x)=p\left(\frac{B_{j,p-1}(x)}{t_{j+p}-t_j}-\frac{B_{j+1,p-1}(x)}{t_{j+p+1}-t_{j+1}}\right)
$$

(`dB(spline, j, x)` in `kan.cpp`.)

### SiLU and its derivative

$$
\operatorname{silu}(x)=x\,\sigma(x)=\frac{x}{1+e^{-x}},\qquad
\operatorname{silu}'(x)=\sigma(x)\bigl(1+x\,(1-\sigma(x))\bigr)
$$

(`silu(Neuron&)` in `engine.cpp`, `silu_prime` in `kan.cpp`.)

### The learnable edge function

Every edge function (inner `φ` and outer `Φ`) has the same form:

$$
\varphi(x)=c\cdot\operatorname{silu}(x)+\sum_{j=0}^{K-1} w_j\,B_{j,k}(x)
$$

* `c` is one scalar (the "base"/residual weight),
* `w_0 … w_{K-1}` are the spline coefficients.

This is the KAN paper's `φ(x) = w_b·silu(x) + w_s·spline(x)` with `w_s` absorbed into the `w_j`.

### The whole network

Inner (first) layer, one function per (input `p`, hidden node `q`) pair, `n·m = 10` in total:

$$
\varphi_{p,q}(x_p)=c1_{p,q}\operatorname{silu}(x_p)+\sum_{j=0}^{K-1} w^{\text{in}}_{p,q,j}\,B_{j,k}(x_p)
$$

Hidden nodes just add:

$$
h_q=\sum_{p=0}^{n-1}\varphi_{p,q}(x_p) = \varphi_{0,q}(x_0)+\varphi_{1,q}(x_1)
$$

Outer (second) layer, one function per hidden node (5 in total):

$$
\Phi_q(h_q)=c2_q\operatorname{silu}(h_q)+\sum_{j=0}^{K-1} w^{\text{out}}_{j}\,B^{\text{out}}_{j,k}(h_q)
$$

The output is the sum:

$$
\hat y=f(x)=\sum_{q=0}^{m-1}\Phi_q(h_q)
$$

Two implementation choices to know about:

1. **`w_out` is shared** by all 5 outer functions (only `c2_q` differs per `q`). The paper gives each
   `Φ_q` its own coefficients; sharing here keeps the parameter count small.
2. **The outer grid is data-dependent.** The spline `B^out` is rebuilt on every forward pass on the
   range `[min_q h_q, max_q h_q]`, because hidden values have no fixed range. The inner grid is fixed
   at `[a, b]`.

### Loss

Per-sample squared error, minimised by plain SGD:

$$
L=(\hat y - y)^2,\qquad \frac{\partial L}{\partial \hat y}=2(\hat y-y)
$$

---

## 3. Design: Neuron, NeuronPool, local_grad

### `Neuron`: one node of the computation graph

```cpp
class Neuron {
public:
    double data;                  // value computed in the forward pass
    double grad;                  // dL/d(this neuron), filled by backward()
    std::string op;               // "", "+", "*", "tanh", "silu", "spline"
    std::string label;            // text for Graphviz ("Φ0,3", ...)
    std::vector<Neuron*> prev;    // the neurons this one was computed from (graph edges)
    std::vector<double> local_grad; // stored partial derivatives (only used by "spline" nodes)
    NeuronPool* npool;            // the pool this neuron lives in
};
```

This is micrograd's `Value`: a number, its gradient, and a record of how it was made.

### `NeuronPool`: why it exists

```cpp
class NeuronPool {
public:
    Neuron& append(Neuron n);        // add a node, return a stable reference to it
    void backward(double seed);      // reverse-mode autodiff over the whole pool
    std::deque<Neuron> pool;         // all neurons, in creation order
};
```

A `NeuronPool` is a **tape** (a "Wengert list"). It solves:

1. **Stable addresses.** `Neuron::prev` holds raw pointers. `std::deque::push_back` never moves existing elements, so
   `Neuron&` and `Neuron*` stay valid while the graph is being built.
2. **Free topological order.** A neuron is appended *after* the neurons it depends on, so creation
   order is already a valid topological order. `backward` is a single loop from the last element to
   the first. micrograd needs a DFS with a `visited` set to compute this order; we don't.
3. **Operators need no global state.** `a + b` has to create a new neuron somewhere. Each neuron
   remembers its pool (`n.npool = this` inside `append`), so `operator+` just calls
   `a.npool->append(...)`. No global tape, and several pools can coexist.


### How derivatives are stored: three places

| Stored in | Filled by|
|---|---|
| `Neuron::grad` | `NeuronPool::backward` |
| `Neuron::prev` | operators / KAN |
| `Neuron::local_grad` | `KAN::compute_layers` (forward pass) |

For the simple operations the local derivative is trivial and is recomputed inside `backward`
from `prev[i]->data` (`+` has local derivative 1, `*` has the other operand, `tanh` has `1 - t²`).
For a **spline node** it is not trivial, so it is computed once in the forward pass and stored.

### About `local_grad` - 

A spline edge `φ(x) = c·silu(x) + Σ w_j B_j(x)` is a **single** neuron with `op = "spline"` and one
parent (`prev = {&x}`). Instead of expanding it into a bigger pool of neurons, the forward pass computes everything it will ever need and stores
it:

```
local_grad = [ dφ/dx , silu(x) , B_0(x) , B_1(x) , ... , B_{K-1}(x) ]
                 [0]     [1]       [2]      [3]            [K+1]
```

| Index | Value | Equals | Used by |
|---|---|---|---|
| `[0]` | `d = c·silu'(x) + Σ w_j·B_j'(x)` | ∂φ/∂x | **`backward`**, to push the gradient to the parent |
| `[1]` | `s = silu(x)` | ∂φ/∂c | **`train`**, to update `c` |
| `[2+j]` | `B_j(x)` | ∂φ/∂w_j | **`train`**, to update `w_j` |

So `local_grad` holds **both** kinds of derivative that a parameterised node has:

* the derivative w.r.t. its **input** (index 0), needed to continue backpropagation, and
* the derivatives w.r.t. its **parameters** (indices 1…), needed by the optimiser.

The generic backward rule for `"spline"` is

```cpp
for (size_t j = 0; j < prev.size(); j++)
    prev[j]->grad += local_grad[j] * grad;     // chain rule: parent.grad += local * child.grad
```

Because `prev.size() == 1`, only `local_grad[0]` is consumed by `backward`. Everything after it is
left for the training loop, which reads it **after** `backward` has filled `phi.grad`:

$$
\frac{\partial L}{\partial c}=\texttt{phi.grad}\cdot\texttt{local\_grad[1]},\qquad
\frac{\partial L}{\partial w_j}=\texttt{phi.grad}\cdot\texttt{local\_grad[2+j]}
$$

This avoids re-evaluating the splines during the backward pass: they were already evaluated in the
forward pass.

---

## 4. Docs

### `headers/spline.hpp` + `src/spline.cpp`: class `Spline`

| Member | What it does |
|---|---|
| `int G, k; double a, b, h;` | number of intervals, degree, range, spacing `h = (b-a)/G` |
| `std::vector<double> t` | knot vector, `t[i] = a + (i-k)·h` for `i = 0..G+2k` |
| `Spline(G, k, a, b)` | stores the parameters and builds `t` (formula 3.2) |
| `double coxDeBoor(int j, int p, double x)` | evaluates `B_{j,p}(x)` by the recursion in 3.3 (recursive, no caching) |
| `double spline_value(const vector<double>& c, double x)` | evaluates `Σ_{i<G+k} c[i]·B_{i,k}(x)`, a plain spline with coefficients `c` (a helper; the KAN itself inlines this sum) |

### `headers/engine.hpp` + `src/engine.cpp`: the autograd engine

| Function | What it does |
|---|---|
| `Neuron(data, prev, op)` | creates a node; `grad = 0`, `npool = nullptr` until appended |
| `NeuronPool::append(Neuron n)` | sets `n.npool = this`, stores it in the deque, returns a reference to the stored copy |
| `operator+(Neuron&, Neuron&)` | creates a `"+"` neuron with `prev = {&a, &b}` in `a`'s pool |
| `operator*(Neuron&, Neuron&)` | same for `"*"` |
| `tanh(Neuron&)` | creates a `"tanh"` neuron |
| `silu(Neuron&)` | creates a `"silu"` neuron with a numerically stable formula (separate branches for `x ≥ 0` and `x < 0` so `exp` never overflows). The KAN reuses this neuron and re-labels it as a spline edge |
| `NeuronPool::backward(double seed)` | reverse sweep described in 4.5 |

### `headers/graph.hpp` + `src/graph.cpp`

| Function | What it does |
|---|---|
| `save_dot(const NeuronPool&, path)` | writes the whole pool as a Graphviz `.dot` file (record node with `label | data | grad`, one small operator node per non-leaf, arrows from `prev`). Render with `dot -Tpng graph.dot -o graph.png` |

### `headers/kan.hpp` + `src/kan.cpp`: class `KAN`

Members: `Spline spline` (the **inner** grid), `NeuronPool& npool` (a reference; the caller owns the
pool), `size_t nparams` (= number of inputs, 2), `vector<Neuron*> hidden_layer`, `vector<Neuron*> outer_layer`.

| Function | What it does |
|---|---|
| `static double dB(Spline&, j, x)` | derivative of `B_{j,k}` (formula 3.4) |
| `static double silu_prime(x)` | `σ(x)(1 + x(1−σ(x)))` (formula 3.5) |
| `KAN(G, k, a, b, npool)` | builds the inner `Spline` and stores the pool reference |
| `compute_first_layer(w_in, c1)` | For each input `p` it evaluates all `K` basis values `B_j(x_p)` once. Then for each of the 5 hidden targets `q` it creates one spline neuron `φ_{p,q}`: starts from `silu(x_p)` (kept as `s`), multiplies by `c1[idx]`, adds `Σ w_in[idx][j]·B_j`, accumulates the derivative `d`, and fills `local_grad = {d, s, B_0..B_{K-1}}`, `op = "spline"`, `label = "Φp,q"` |
| `compute_hidden_neurons()` | creates the 5 hidden neurons `h_q = φ_{0,q} + φ_{1,q}` with the `+` operator (its two parents are `pool[nparams+q]` and `pool[nparams+q+2n+1]`) and stores pointers in `hidden_layer` |
| `compute_outer_layer(w_out, c2)` | finds `min`/`max` of the hidden values, builds a **new** `Spline` on `[min, max]`, evaluates its basis at every `h_q`, then creates the 5 outer spline neurons `Φ_q` exactly like the inner ones (`c2[q]`, shared `w_out`) and stores pointers in `outer_layer` |
| `output()` | sums the 5 outer neurons with a right-nested chain of `+` (`Φ0 + (Φ1 + (Φ2 + (Φ3 + Φ4)))`, 4 `+` neurons). The last neuron of the pool is the network output |
| `double forward(x, w_in, c1, w_out, c2)` | `pool.clear()`, re-creates the input neurons, runs the four steps above, returns `pool.back().data` |
| `train(X, Y, lr, steps, w_in, c1, w_out, c2, verbose=true)` | per-sample SGD, see [section 7](#7-how-training-works). Returns the mean loss of each epoch; prints `step i loss L` only if `verbose` |

### `src/kan_bench.cpp`

Command-line program: `kan_bench <lr> <epochs> <seed>`. Initialises all 70 parameters uniformly in
`[-1, 1]` with `std::mt19937(seed)` (same stream on every platform), trains on `y = 3x²+7x+2` using
9 points in `[-1, 1]` (the same `x` fed to both inputs), prints `step i loss L` per epoch and then
`TIME`, `TEST` (MSE on 20 unseen points), `NPARAMS`. `compare.py` parses this output.

### `compare.py`

Runs `kan_bench` and a micrograd `MLP(2, [6, 6, 1])` (67 parameters) with the **same** data, the same
per-sample SGD, the same seeds, and prints the table in [section 8](#8-benchmark-vs-karpathys-micrograd).

---

## 5. Visualization and Flow

### Forward pass (`KAN::forward`) and the pool layout

For a sample (two starting params) `x = (x₀, x₁)`, the pool looks like this - (`n = 2`, `G+k = 5, G=2 & k=3`):

```
index        neuron                         op        parents (prev)        
─────────────────────────────────────────────────────────────────────────────
 0           x₀                             (leaf)    -
 1           x₁                             (leaf)    -
 2 … 6       φ₀,₀ … φ₀,₄  (edges of x₀)     spline    {x₀}                  
 7 … 11      φ₁,₀ … φ₁,₄  (edges of x₁)     spline    {x₁}                  
12 … 16      h₀ … h₄  = φ₀,q + φ₁,q         +         {φ₀,q , φ₁,q}         
17 … 21      Φ₀ … Φ₄                        spline    {h_q}                 
22 … 25      running sum of the Φ's         +         ...                   
                                                      └─ pool[25] = output ŷ
```

(26 neurons per sample; `local_grad` has `2 + (G+k) = 7` entries.) Inner edge `φ_{p,q}` sits at
`pool[nparams + p·5 + q]`, which is exactly the address the training loop uses.


### Backward pass (`NeuronPool::backward(2e)`)

```
pool[25].grad = 2e                       (seed = dL/dŷ)
for i = 25 … 1:                          (reverse creation order = reverse topological order)
    "+"      : pass grad to both parents unchanged
    "spline" : parent.grad += local_grad[0] * grad
```

Flow of gradients: output sum → each `Φ_q.grad = 2e` → `h_q.grad = Φ_q'(h_q)·2e` (via `local_grad[0]`)
→ both `φ_{0,q}.grad` and `φ_{1,q}.grad` equal `h_q.grad` (because `h_q` is a plain sum) →
`x_p.grad` (the input gradients; computed but not needed for training).

### Graphs and workflow - 

To sum it up this is how the layers are made - 
-  From the example only, two params are at the starting point. So now each one will give rise to 2*n+1 phi functions. So in total - n*(2n+1) phi functions. Here, n=2, so each one will have 5 phi functions. 
- To make a phi function we need to calculate the B-splines. We know that phi is the sum of B-splines. So from 0 to G+k-1 splines are created - B_0,k; B_1,k; ... B_G+k-1,k. In this case, it's only B0,3; B1,3; B2,3; B3,3 - so 4 b-splines. 
- These B-splines are multiplied with respective weights and added, and then the resultant is added with (silu of x)*c. This gives the phis of inner layer.
- We add those phis which gives hidden neurons. 
- Again B-splines of hidden neurons are taken, and similarly again added to get the last - outer layer of phis. The number of hidden neurons = number of phis in the outer layer = 2n+1 = 5.
- Finally, the outer layer phis are added to give the predicted y.

*Below is a sample graph, here hidden neurons are also shown*

---

## 6. Benchmark vs micrograd

### Setup

| | microkan | micrograd MLP |
|---|---|---|
| Model | `[2, 5, 1]`, `G=2`, `k=3`, grid `[-2,2]` | `MLP(2, [6, 6, 1])` |
| Parameters | 70 | 67 |
| Language | C++ (`-O2`) | pure Python |
| Task | `y = 3x² + 7x + 2` (same `x` fed to both inputs) | same |
| Train data | 9 points: `-1, -0.75, …, 1` | same |
| Optimiser | per-sample SGD, 200 epochs | same |
| Init | uniform `[-1, 1]`, seeds 0–4 (both models) | same |
| Test | 20 unseen points `-0.97, -0.87, …, 0.93` | same |

The reported numbers are **medians over 5 seeds**. Reproduce with:

```bash
g++ -std=c++20 -O2 -o kan_bench src/kan.cpp src/engine.cpp src/spline.cpp src/kan_bench.cpp
git clone https://github.com/karpathy/micrograd
python compare.py
```

### Results

```
task: y=3x^2+7x+2, 9 train pts, 200 epochs of per-sample SGD, random init, median of 5 seeds, test = 20 unseen points in [-1,1]

model              lr params   loss@10   loss@50  loss@end  ep<0.1   test mse    worst     sec  div
KAN             0.001     70   15.9660    1.6716   0.02778     116    0.08678   0.3678   0.154  0/5
micrograd MLP   0.001     67    6.6746    0.7063   0.09072     155    0.08939   0.3231   1.326  0/5
KAN             0.003     70    8.6255    0.1109   0.01510      55    0.02601   0.0768   0.175  0/5
micrograd MLP   0.003     67    1.2218    0.1716   0.08308      78    0.06765   0.1810   3.070  0/5
KAN              0.01     70    0.2998    0.0454   0.01226      27    0.01070   0.1577   0.149  0/5
micrograd MLP    0.01     67    0.3747    0.0834   0.06189      41    0.07265   0.1281   1.642  0/5
KAN              0.02     70    0.4624    0.0291   0.02556      18    0.01247   0.0648   0.161  0/5
micrograd MLP    0.02     67    0.6313    0.2355   0.07537      21    0.03673  17.2888   1.233  0/5

best lr per model (lowest final train loss):
  KAN            lr=0.01  loss=0.01226  test mse=0.01070  time=0.149s
  micrograd MLP  lr=0.01  loss=0.06189  test mse=0.07265  time=1.642s
```

Columns: `loss@N` = median mean-squared training loss after N epochs, `ep<0.1` = median first epoch with
loss below 0.1, `test mse` = median test error, `worst` = worst test error over the 5 seeds,
`sec` = median training time, `div` = diverged runs.

### Reading the results

* **Accuracy.** At each model's best learning rate (0.01) the KAN reaches a final training loss of
  0.0123 vs 0.0619 (about 5× lower) and a test MSE of 0.0107 vs 0.0727 (about 7× lower). It also
  gets below loss 0.1 sooner (epoch 27 vs 41). At the other three learning rates the KAN also ends
  with the lower training loss.
* **Why it suits this task.** The target is a smooth low-degree polynomial, which cubic B-splines
  represent very naturally, whereas a ReLU MLP (micrograd's default activation) must stitch it
  together from straight pieces.
* **Stability.** No run diverged for either model. micrograd at `lr=0.02` had one bad seed with a
  test error of 17.3; the KAN's worst case at that rate was 0.065. The KAN's worst case at
  `lr=0.01` (0.158) shows the seed-to-seed variance is real, so look at the medians *and* the
  `worst` column.
* **Speed (read with care).** The KAN trains about 11× faster (0.149 s vs 1.642 s), but this
  compares compiled C++ against interpreted Python, so most of the gap is language overhead, not
  architecture. The accuracy comparison is the meaningful one.
* **Scope of the claim.** This is a tiny 1-D regression with 9 training points. It shows that the
  implementation learns correctly and competitively here; it is not evidence about large-scale
  tasks.

---

## 7. How to use?

- Clone the repo
- Make a main.cpp in the root
- Make the main.cpp by the following the sample file `test.cpp` given inside testsuite
- Just follow the KAN setup, as changing the G, k, or even the number of params will result in increasing the number of weights.
- Keep the flow as it is, but you can also change. 
- At last call, `forward` to test on another value.
- Edit the paths as required
- Source compile all the files in the `headers` and `src`. Here is a sample compile command for `main.cpp` that is in the root folder, edit it as you see fit - 
    ```
    g++ -Wall -Wextra main.cpp src/engine.cpp src/graph.cpp src/kan.cpp src/spline.cpp -o main
    ```
- You have to download [graphviz](https://graphviz.org/download/) to change the `dot` files generated using the following command for each one - 
    ```
    dot -Tsvg out/graph_first.dot -o out/graph_first.svg
    ```

**How to run the benchmark test?**
*This is solely done against micrograd. The maximum benchmarking code is given by AI.*
- Go inside testsuite and clone micrograd by andrej karpathy - 
    ```
    git clone --depth 1 https://github.com/karpathy/micrograd.git
    ```
- run this command to make a `kan_bench` executable - 
    ```
    g++ -O2 kan_bench.cpp ../src/engine.cpp ../src/kan.cpp ../src/spline.cpp -o kan_bench
    ```
- then run the `compare.py` -
    ```
    python compare.py
    ```

---

## 8. Known limitations and bugs

1. **`w_out` is shared** across the 5 outer edges (the paper uses separate coefficients per `Φ_q`).
2. **Inputs outside `[a, b]`** get no spline contribution (only the SiLU term) on the inner layer. (as expected)
3. **Not implemented** from the paper: grid refinement/extension, sparsification and pruning,
   symbolic regression, multiple outputs.
4. **Performance.** `coxDeBoor` is a plain recursion without caching and `dB` re-evaluates it; fine for
   this size, not for large grids.

---

## 9. References

* **KAN paper:** Z. Liu, Y. Wang, S. Vaidya, F. Ruehle, J. Halverson, M. Soljačić, T. Y. Hou,
  M. Tegmark. *KAN: Kolmogorov–Arnold Networks.* arXiv:2404.19756 (2024), ICLR 2025.
  https://arxiv.org/abs/2404.19756
* **Representation theorem:** A. N. Kolmogorov (1957), *On the representation of continuous functions
  of many variables by superposition of continuous functions of one variable and addition*;
  V. I. Arnold (1957), *On functions of three variables*.
* **B-splines:** C. de Boor (1972), *On calculating with B-splines*, J. Approx. Theory 6, 50–62;
  M. G. Cox (1972), *The numerical evaluation of B-splines*, IMA J. Appl. Math. 10, 134–149.
* **SiLU:** S. Elfwing, E. Uchibe, K. Doya (2018), *Sigmoid-weighted linear units for neural network
  function approximation in reinforcement learning*, Neural Networks 107.
* **Autograd design:** Andrej Karpathy, *micrograd*. https://github.com/karpathy/micrograd