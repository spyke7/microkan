#pragma once
#include <iostream>
#include <vector>
#include <deque>
#include <functional>
#include <deque>
#include <string>

class NeuronPool;

class Neuron
{
public:
    double data;
    std::string op = "";
    double grad=0.0;
    std::string label;
    std::vector<Neuron *> prev;
    NeuronPool* npool;
    // std::function<void(Neuron&)> backward_fn; 

    Neuron(double data, std::vector<Neuron *> prev = {}, const char *op = ""){
        this->data = data;
        this->prev = prev;
        this->op = op;
        this->grad = 0.0;
        this->npool = nullptr;
    }
};

class NeuronPool
{
public:
    Neuron& append(Neuron n){
        n.npool=this;
        pool.emplace_back(n);
        return pool.back();
    }

    void backward();
    std::deque<Neuron> pool;
};

Neuron& operator+(Neuron& a, Neuron& b);
Neuron& operator*(Neuron& a, Neuron& b);

// activation functions
Neuron& tanh(Neuron& a);
Neuron& silu(Neuron &a);