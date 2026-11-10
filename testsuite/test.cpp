#include <iostream>
#include <vector>
#include "../headers/engine.hpp"
#include "../headers/graph.hpp"
#include "../headers/kan.hpp"

int main()
{
    NeuronPool npool;

    // n*(2n+1) = 10 rows, each of length G+k = 5
    std::vector<std::vector<double>> w_in = {
        {0.1, 0.2, 0.3, 0.4, 0.5}, {0.4, 0.5, 0.6, 0.7, 0.8}, {0.7, 0.8, 0.9, 0.1, 0.2}, {0.2, 0.3, 0.4, 0.5, 0.6}, {0.5, 0.6, 0.7, 0.8, 0.9}, {0.3, 0.4, 0.5, 0.6, 0.7}, {0.6, 0.7, 0.8, 0.9, 0.1}, {0.9, 0.1, 0.2, 0.3, 0.4}, {0.1, 0.3, 0.5, 0.7, 0.9}, {0.2, 0.4, 0.6, 0.8, 0.1}};
    std::vector<double> c1 = {0.5, 0.6, 0.7, 0.8, 0.9, 0.4, 0.3, 0.2, 0.1, 0.5};

    // shared across the outer phis, so G+k = 5 entries
    std::vector<double> w_out = {0.1, 0.2, 0.3, 0.4, 0.5};
    std::vector<double> c2 = {0.2, 0.3, 0.4, 0.5, 0.6};

    KAN kan(2, 3, -2.0, 2.0, npool);

    // training data: y = 3x^2 + 7x + 2, the same x goes into both inputs
    std::vector<std::vector<double>> X;
    std::vector<double> Y;
    for (double a = -1.0; a <= 1.0; a += 0.25)
    {
        X.push_back({a, a});
        Y.push_back(3 * a * a + 7 * a + 2);
    }

    // graph 1: before training
    double p0 = kan.forward(X[0], w_in, c1, w_out, c2);
    npool.backward(2.0 * (p0 - Y[0])); // d(pred-y)^2/dpred as the seed
    save_dot(npool, "../out/graph_first.dot");

    kan.train(X, Y, 0.02, 50, w_in, c1, w_out, c2);

    // graph 2: after training (last sample)
    double pl = kan.forward(X.back(), w_in, c1, w_out, c2);
    npool.backward(2.0 * (pl - Y.back()));
    save_dot(npool, "../out/graph_last.dot");

    // graph 3: test on an unseen point
    double pt = kan.forward({0.53, 0.53}, w_in, c1, w_out, c2);
    std::cout << "test pred " << pt << " true " << 3 * 0.53 * 0.53 + 7 * 0.53 + 2 << "\n";
    save_dot(npool, "../out/graph_test.dot");
    return 0;
}