#include "../headers/kan.hpp"

// derivative of the basis function B_{j,k}(x)
static double dB(Spline &s, int j, double x)
{
    int k = s.k;
    return k * (s.coxDeBoor(j, k - 1, x) / (s.t[j + k] - s.t[j]) - s.coxDeBoor(j + 1, k - 1, x) / (s.t[j + k + 1] - s.t[j + 1]));
}

static double silu_prime(double x)
{
    double sig = 1.0 / (1.0 + std::exp(-x));
    return sig * (1.0 + x * (1.0 - sig));
}

KAN::KAN(int G, int k, double a, double b, NeuronPool &npool)
    : spline(G, k, a, b), npool(npool) {}

void KAN::compute_first_layer(const std::vector<std::vector<double>> &w_in, const std::vector<double> &c1)
{
    std::vector<std::vector<double>> Bvalues;
    std::vector<Neuron *> params;

    for (size_t i = 0; i < npool.pool.size(); i++)
    {
        Neuron &x = npool.pool[i];
        params.push_back(&x);

        std::vector<double> temp;
        for (size_t j = 0; j < spline.G + spline.k; j++)
        {
            double B = spline.coxDeBoor(j, spline.k, x.data);
            temp.push_back(B);
        }
        Bvalues.push_back(temp);
    }

    size_t idx = 0;
    for (size_t i = 0; i < params.size(); i++)
    {
        for (size_t j = 0; j < 2 * params.size() + 1; j++, idx++)
        {
            Neuron &phi = silu(*params[i]);
            double s = phi.data;
            double d = c1[idx] * silu_prime((*params[i]).data);
            phi.data *= c1[idx];

            for (size_t k = 0; k < spline.G + spline.k; k++)
            {
                phi.data += w_in[idx][k] * Bvalues[i][k];
                d += w_in[idx][k] * dB(spline, k, (*params[i]).data);
            }
            phi.label = "Φ" + std::to_string(i) + "," + std::to_string(j);
            phi.local_grad = {d, s};
            phi.local_grad.insert(phi.local_grad.end(), Bvalues[i].begin(), Bvalues[i].end());
            phi.op = "spline";
        }
    }
    nparams = params.size();
}

void KAN::compute_hidden_neurons()
{
    for (size_t j = nparams; j <= 2 * nparams + 2; j++)
    {
        Neuron &h = npool.pool[j] + npool.pool[j + 2 * nparams + 1];
        hidden_layer.push_back(&h);
    }
}

void KAN::compute_outer_layer(const std::vector<double> &w_out, const std::vector<double> &c2)
{
    std::vector<std::vector<double>> Bvalues;
    double maxEl = std::numeric_limits<double>::lowest(), minEl = std::numeric_limits<double>::max();
    
    for (size_t i = 0; i < hidden_layer.size(); i++)
    {
        maxEl = std::max(maxEl, (*hidden_layer[i]).data);
        minEl = std::min(minEl, (*hidden_layer[i]).data);
    }

    Spline outer_splines(spline.G, spline.k, minEl, maxEl);

    for (size_t i = 0; i < hidden_layer.size(); i++)
    {
        std::vector<double> temp;
        for (size_t j = 0; j < spline.G + spline.k; j++)
        {
            double B = outer_splines.coxDeBoor(j, spline.k, (*hidden_layer[i]).data);
            temp.push_back(B);
        }

        Bvalues.push_back(temp);
    }

    for (size_t i = 0; i < hidden_layer.size(); i++)
    {
        Neuron &phi_outer = silu(*hidden_layer[i]);
        double s = phi_outer.data;
        double d = c2[i] * silu_prime((*hidden_layer[i]).data);
        phi_outer.data *= c2[i];
        for (size_t j = 0; j < spline.G + spline.k; j++)
        {
            phi_outer.data += w_out[j] * Bvalues[i][j];
            d += w_out[j] * dB(outer_splines, j, (*hidden_layer[i]).data);
        }
        phi_outer.op = "spline";
        phi_outer.local_grad = {d,s};
        phi_outer.local_grad.insert(phi_outer.local_grad.end(), Bvalues[i].begin(), Bvalues[i].end());
        outer_layer.push_back(&phi_outer);
    }
}

void KAN::output()
{
    auto sum = [&](auto self, size_t i) -> Neuron &
    {
        if (i == outer_layer.size() - 2)
            return *outer_layer[i] + *outer_layer[i + 1];

        return *outer_layer[i] + self(self, i + 1);
    };

    Neuron &out = sum(sum, 0);
}

double KAN::forward(const std::vector<double> &x, const std::vector<std::vector<double>> &w_in, const std::vector<double> &c1, const std::vector<double> &w_out, const std::vector<double> &c2)
{
    npool.pool.clear();
    for (double v : x)
        npool.append(Neuron(v));
    hidden_layer.clear();
    outer_layer.clear();

    compute_first_layer(w_in, c1);
    compute_hidden_neurons();
    compute_outer_layer(w_out, c2);
    output();
    return npool.pool.back().data;
}

void KAN::train(const std::vector<std::vector<double>> &X, const std::vector<double> &Y, double lr, int steps, std::vector<std::vector<double>> &w_in, std::vector<double> &c1, std::vector<double> &w_out, std::vector<double> &c2)
{
    for (int step = 0; step < steps; step++)
    {
        double total = 0.0;
        for (size_t s = 0; s < X.size(); s++)
        {
            // build the network with already tweaked weights
            double e = forward(X[s], w_in, c1, w_out, c2) - Y[s];
            total += e * e;

            // grad of (pred - y)^2 goes to the output neuron, backward fills grad of every phi
            npool.backward(2.0 * e);

            // first layer: phi number idx sits at nparams + idx
            for (size_t idx = 0; idx < w_in.size(); idx++)
            {
                Neuron &phi = npool.pool[nparams + idx];
                c1[idx] -= lr * phi.grad * phi.local_grad[1];
                for (size_t k = 0; k < w_in[idx].size(); k++)
                    w_in[idx][k] -= lr * phi.grad * phi.local_grad[2 + k];
            }

            // outer layer: w_out is shared, so every outer phi adds to it
            for (size_t i = 0; i < outer_layer.size(); i++)
            {
                Neuron &phi = *outer_layer[i];
                c2[i] -= lr * phi.grad * phi.local_grad[1];
                for (size_t j = 0; j < w_out.size(); j++)
                    w_out[j] -= lr * phi.grad * phi.local_grad[2 + j];
            }
        }
        std::cout << "step " << step << " loss " << total / X.size() << "\n";
    }
}
