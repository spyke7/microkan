#pragma once
#include <iostream>
#include <vector>
#include <deque>
#include <functional>
#include <deque>

class NeuronPool;

class Neuron
{
public:
    double data;
    const char *op = "";
    double grad;
    std::vector<Neuron *> prev;
    NeuronPool* npool;

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

private:
    std::deque<Neuron> pool;
};

Neuron& operator+(Neuron& a, Neuron& b);
Neuron& operator*(Neuron& a, Neuron& b);