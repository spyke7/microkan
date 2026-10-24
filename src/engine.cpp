#include "../headers/engine.hpp"
#include <cmath>

// Neuron functions
Neuron& operator+(Neuron& a, Neuron& b){
    Neuron n(a.data + b.data, {&a, &b}, "+");
    return a.npool->append(n);
}
Neuron& operator*(Neuron& a, Neuron& b){
    Neuron n(a.data * b.data, {&a, &b}, "*");
    return a.npool->append(n);
}
