#include "../headers/spline.hpp"
#include <algorithm>

double Spline::coxDeBoor(int j, int p, double x)
{
    x = std::min(std::max(x, t[k]), t[G + k]);
    if (p == 0)
    {
        if (j < k || j >= G + k)
            return 0.0;    
        bool last = (j == G + k - 1);           
        if (x >= t[j] && (x < t[j + 1] || (last && x <= t[j + 1]))) 
            return 1.0;
        return 0.0;
    }
    else
    {
        double B = ((x - t[j]) / (t[j + p] - t[j])) * coxDeBoor(j, p - 1, x) + ((t[j + p + 1] - x) / (t[j + p + 1] - t[j + 1])) * coxDeBoor(j + 1, p - 1, x);
        return B;
    }
}

double Spline::spline_value(const std::vector<double> &c, double x)
{
    double s = 0;
    for (int i=0; i < G + k; i++)
    {
        s+=c[i]*coxDeBoor(i,k,x);
    }
    
    return s;
}
