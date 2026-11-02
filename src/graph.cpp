#include "../headers/graph.hpp"
#include <fstream>
#include <iomanip>
#include <iostream>

void save_dot(const NeuronPool& npool, std::string path){
    std::ofstream f(path);
    if (!f)                                      
    {
        std::cout << "cannot open " << path << "\n";   
        return;                                  
    }

    // writing in DOT file
    f << "digraph {\n";
    f << "  rankdir=LR;\n"; 

    // drawing each neuron
    for (const Neuron& n : npool.pool){
        f << "  n" << &n 
          << " [shape=record, label=\"{ " << n.label         
          << " | data " << n.data
          << " | grad " << n.grad << " }\"];\n";

        if (n.op[0]!='\0'){
            f << "  n" << &n << "op [label=\"" << n.op << "\"];\n";
            f << "  n" << &n << "op -> n" << &n << ";\n"; 
        }

    }

    // drawing arrows
    for (const Neuron& n : npool.pool)         
    {
        for (Neuron* p : n.prev)                 
        {
            f << "  n" << p << " -> n" << &n << "op;\n";    
        }
    }

    f << "}\n";
}
