#include "../headers/spline.hpp"
#include <algorithm>
#include <cmath>

double Spline::coxDeBoor(int j, int p, double x)
{
    x = std::min(std::max(x, t[k]), t[G + k]);

    if (!tri_valid || x!=tri_x){
        int s = span(x);
        
        // building the triangle
        tri[0][0] = 1.0;                      
        for (int deg = 1; deg <= k; deg++)
            for (int pos = 0; pos <= deg; pos++)
            {
                int bump = s - deg + pos;                 // bump number
                double lo = (pos >= 1) ? tri[deg - 1][pos - 1] : 0.0;      // B_{j,p-1}
                double hi = (pos <= deg - 1) ? tri[deg - 1][pos] : 0.0;      // B_{j+1,p-1}
                tri[deg][pos] = ((x - t[bump]) / (t[bump + deg] - t[bump])) * lo
                        + ((t[bump + deg + 1] - x) / (t[bump + deg + 1] - t[bump + 1])) * hi;
            }
        tri_x = x;
        tri_s = s;
        tri_valid = true;
    }

    int r = j - (tri_s - p);                   
    if (r < 0 || r > p)
        return 0.0;                            // this bump is zero at x
    return tri[p][r];
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

int Spline::span(double x){
    x = std::min(std::max(x, t[k]), t[G + k]);
    int s = k + (int)std::floor((x - a) / h);
    s = std::min(s, G + k - 1);
    while (s > k && x < t[s]) s--;
    while (s < G + k - 1 && x >= t[s + 1]) s++;
    return s;
}