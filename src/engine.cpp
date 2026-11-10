#include "../headers/engine.hpp"
#include <cmath>

// Neuron functions
Neuron &operator+(Neuron &a, Neuron &b)
{
    Neuron n(a.data + b.data, {&a, &b}, "+");
    return a.npool->append(n);
}
Neuron &operator*(Neuron &a, Neuron &b)
{
    Neuron n(a.data * b.data, {&a, &b}, "*");
    return a.npool->append(n);
}

// activation function
Neuron &tanh(Neuron &a)
{
    // (exp(2*a.data) -1)/(exp(2*a.data) +1)
    double t = std::tanh(a.data);
    Neuron n(t, {&a}, "tanh");
    return a.npool->append(n);
}
Neuron &silu(Neuron &a)
{
    double s;
    if (a.data >= 0.0)
    {
        s = (a.data / (1.0 + std::exp(-a.data)));
    }
    else
    {
        double e = std::exp(a.data);
        s = a.data * e / (1.0 + e);
    }
    Neuron n(s, {&a}, "silu");
    return a.npool->append(n);
}

// NeuronPool functions
void NeuronPool::backward(double seed)
{
    for (size_t i = pool.size() - 1; i > 0; i--)
    {
        if (i == pool.size() - 1)
        {
            pool[i].grad = seed;
        }
        if (pool[i].op == "tanh")
        {
            pool[i].prev[0]->grad += (1 - (pool[i].data * pool[i].data)) * pool[i].grad;
        }
        else if (pool[i].op == "silu")
        {
            double x = pool[i].prev[0]->data;
            double sig = 1.0 / (1.0 + std::exp(-x));
            pool[i].prev[0]->grad += sig * (1.0 + x * (1.0 - sig)) * pool[i].grad;
        }
        else if (pool[i].op == "spline")
        {
            for (size_t j = 0; j < pool[i].prev.size(); j++)
                pool[i].prev[j]->grad += pool[i].local_grad[j] * pool[i].grad;
        }
        else if (pool[i].op == "*")
        {
            pool[i].prev[0]->grad += pool[i].prev[1]->data * pool[i].grad;
            pool[i].prev[1]->grad += pool[i].prev[0]->data * pool[i].grad;
        }
        else if (pool[i].op == "+")
        {
            pool[i].prev[0]->grad += pool[i].grad;
            pool[i].prev[1]->grad += pool[i].grad;
        }
        else if (pool[i].op == "spline")
        {
            for (size_t j = 0; j < pool[i].prev.size(); j++)
            {
                pool[i].prev[j]->grad += pool[i].local_grad[j] * pool[i].grad;
            }
        }
    }
}