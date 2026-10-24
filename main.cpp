#include <iostream>
#include "headers/engine.hpp"

int main() {
    NeuronPool neuronpool;
      
    Neuron& a = neuronpool.append(Neuron(5));
    Neuron& b = neuronpool.append(Neuron(3));
    Neuron& c = neuronpool.append(Neuron(2));
    Neuron &f = a*b+c;
    Neuron &d= a+b*c;
    Neuron &e = d*f;
    std::cout<<"f = "<< f.data<<std::endl;
    std::cout << "d = " << d.data << std::endl;
    std::cout << "e = " << e.data << std::endl;
    for (auto p: e.prev)
    {
        std::cout<<p->data<<std::endl;
        for (auto k: p->prev)
        {
            std::cout<<k->data<<std::endl; 
        }
        std::cout<<std::endl;
        
    }
    
    return 0;
}