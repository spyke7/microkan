#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <random>
#include <vector>
#include "../headers/kan.hpp"

int main(int argc, char **argv)
{
    double lr = std::atof(argv[1]);
    int epochs = std::atoi(argv[2]);
    unsigned seed = std::atoi(argv[3]);

    // random init, uniform in [-1, 1] like micrograd (raw mt19937 output, so it is the same on every platform)
    std::mt19937 rng(seed);
    auto rnd = [&]() { return 2.0 * (rng() / 4294967296.0) - 1.0; };

    NeuronPool npool;
    std::vector<std::vector<double>> w_in(10, std::vector<double>(5));
    for (auto &row : w_in)
        for (double &v : row)
            v = rnd();
    std::vector<double> c1(10), w_out(5), c2(5);
    for (double &v : c1)
        v = rnd();
    for (double &v : w_out)
        v = rnd();
    for (double &v : c2)
        v = rnd();
    KAN kan(2, 3, -2.0, 2.0, npool);

    auto f = [](double x) { return 3 * x * x + 7 * x + 2; };

    // train: 9 points -1, -0.75, ..., 1 (same x goes into both inputs)
    std::vector<std::vector<double>> X;
    std::vector<double> Y;
    for (int i = 0; i < 9; i++)
    {
        double a = -1.0 + 0.25 * i;
        X.push_back({a, a});
        Y.push_back(f(a));
    }

    auto t0 = std::chrono::steady_clock::now();
    kan.train(X, Y, lr, epochs, w_in, c1, w_out, c2); // prints "step i loss L" per epoch
    double secs = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();

    // test: 20 unseen points inside [-1, 1]: -0.97, -0.87, ..., 0.93
    double mse = 0.0;
    for (int i = 0; i < 20; i++)
    {
        double a = -0.97 + 0.1 * i;
        double e = kan.forward({a, a}, w_in, c1, w_out, c2) - f(a);
        mse += e * e;
    }
    std::cout << "TIME " << secs << "\n";
    std::cout << "TEST " << mse / 20 << "\n";
    std::cout << "NPARAMS " << 10 * 5 + 10 + 5 + 5 << "\n";
}