#pragma once
#include <vector>

class Spline
{
public:
    int G;
    int k;
    double a, b;
    double h; // (b-a)/h
    std::vector<double> t;
    bool tri_valid = false;
    double tri_x = 0.0;
    int tri_s = 0;
    std::vector<std::vector<double>> tri;

    Spline(int G, int k, double a, double b)
    {
        this->G = G;
        this->k = k;
        this->a = a;
        this->b = b;
        this->h = (b - a) / G;
        for (int i = 0; i <= G + 2 * k; i++)
        {
            this->t.push_back(a + (i - k) * this->h);
        }
        tri.resize(k + 1);
        for (int p = 0; p <= k; p++)
            tri[p].assign(p + 1, 0.0);
    }

    double coxDeBoor(int j, int p, double x);
    double spline_value(const std::vector<double> &c, double x);
    int span(double x);
};
