#include <iostream>
#include "headers/engine.hpp"
#include "headers/graph.hpp"

int main()
{
    NeuronPool neuronpool;

    // Neuron &a = neuronpool.append(Neuron(5));
    // Neuron &b = neuronpool.append(Neuron(3));
    // Neuron &c = neuronpool.append(Neuron(2));
    // Neuron &d = a*b;
    // Neuron &e = b*c;
    // Neuron &f = d+c;
    // Neuron &g = a+e;
    // Neuron &h = g*f;
    // a.label = "a";
    // b.label = "b";
    // c.label = "c";
    // f.label = "f";
    // d.label = "d";
    // e.label = "e";
    // g.label = "g";
    // h.label = "h";

    Neuron &x1=neuronpool.append(Neuron(2.0)); x1.label = "x1";
    Neuron &x2 = neuronpool.append(Neuron(0.0)); x2.label = "x2";
    Neuron &w1=neuronpool.append(Neuron(-3.0)); w1.label = "w1";
    Neuron &w2=neuronpool.append(Neuron(1.0)); w2.label = "w2";

    Neuron &b=neuronpool.append(Neuron(6.8813735870195432)); b.label = "b";

    Neuron &x1w1 = x1*w1; x1w1.label="x1*w1";
    Neuron &x2w2 = x2*w2; x2w2.label="x2*w2";
    Neuron &x1w1x2w2 = x1w1+x2w2; x1w1x2w2.label = "x1*w1 + x2*w2";
    Neuron &n = x1w1x2w2 + b; n.label = "n";
    Neuron &o = tanh(n); o.label= "o";
    o.grad=1;

    neuronpool.backward();
    save_dot(neuronpool, "out/graph3.dot");

    return 0;
}