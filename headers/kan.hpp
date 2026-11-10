#pragma once
#include <vector>
#include <string>
#include <limits>
#include <algorithm>
#include <cmath>
#include "spline.hpp"
#include "engine.hpp"

class KAN
{
public:
    int n;
    int H;
    Spline spline;
    NeuronPool &npool;
    size_t nparams;
    std::vector<Neuron *> hidden_layer;
    std::vector<Neuron *> outer_layer;

    KAN(int G, int k, double a, double b, NeuronPool &npool);
    void compute_first_layer(const std::vector<std::vector<double>> &w_in, const std::vector<double> &c1);
    void compute_hidden_neurons();
    void compute_outer_layer(const std::vector<double> &w_out, const std::vector<double> &c2);
    void output();

    double forward(const std::vector<double> &x, const std::vector<std::vector<double>> &w_in, const std::vector<double> &c1, const std::vector<double> &w_out, const std::vector<double> &c2);
    void train(const std::vector<std::vector<double>> &X, const std::vector<double> &Y, double lr, int steps, std::vector<std::vector<double>> &w_in, std::vector<double> &c1, std::vector<double> &w_out, std::vector<double> &c2);
};