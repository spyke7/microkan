#pragma once
#include <vector>

class Spline{
    int G;
    int k;
    double a,b; 
    double h; // (b-a)/h
    std::vector <double> t;

    Spline(int G, int k, double a, double b){
        this->G=G;
        this->k=k;
        this->a=a;
        this->b=b;
        this->h = (b-a)/G;
        for (size_t i{0}; i<=G+2*k; i++){
            this->t.push_back(a+(i-k)*this->h);
        }
    }

    double coxDeBoor(int j, int p, double x);
    double spline_value(const std::vector<double>& c, double x);
};

