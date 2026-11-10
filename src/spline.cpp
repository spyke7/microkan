#include "../headers/spline.hpp"

double Spline::coxDeBoor(int j, int p, double x)
{
    if (p == 0)
    {
        if (x >= t[j] && x < t[j + 1])
        {
            return 1;
        }
        else
        {
            return 0;
        }
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
