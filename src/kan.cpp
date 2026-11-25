#include "../headers/kan.hpp"

// derivative of the basis function B_{j,k}(x)
static double dB(Spline &s, int j, double x)
{
    if (x < s.t[s.k] || x > s.t[s.G + s.k])
        return 0.0;
    int k = s.k;
    return k * (s.coxDeBoor(j, k - 1, x) / (s.t[j + k] - s.t[j]) - s.coxDeBoor(j + 1, k - 1, x) / (s.t[j + k + 1] - s.t[j + 1]));
}

static double silu_prime(double x)
{
    double sig = 1.0 / (1.0 + std::exp(-x));
    return sig * (1.0 + x * (1.0 - sig));
}

KAN::KAN(int G, int k, double a, double b, NeuronPool &npool)
    : spline(G, k, a, b), outer_spline(G, k, a, b), npool(npool) {}

void KAN::compute_first_layer(const std::vector<std::vector<double>> &w_in, const std::vector<double> &c1)
{
    std::vector<std::vector<double>> Bvalues, dBvalues;
    std::vector<int> first;
    std::vector<Neuron *> params;

    for (size_t i = 0; i < npool.pool.size(); i++)
    {
        Neuron &x = npool.pool[i];
        params.push_back(&x);

        int s = spline.span(x.data);
        first.push_back(s - spline.k);

        std::vector<double> temp(spline.G + spline.k, 0.0), dtemp(spline.G + spline.k, 0.0);
        for (size_t j = s - spline.k; j <= s; j++)
        {
            temp[j] = spline.coxDeBoor(j, spline.k, x.data);
            dtemp[j] = dB(spline, j, x.data);
        }
        Bvalues.push_back(temp);
        dBvalues.push_back(dtemp);
    }

    size_t idx = 0;
    for (size_t i = 0; i < params.size(); i++)
    {
        double sp = silu_prime(params[i]->data);
        for (size_t j = 0; j < 2 * params.size() + 1; j++, idx++)
        {
            Neuron &phi = silu(*params[i]);
            double s = phi.data;
            double d = c1[idx] * sp;
            phi.data *= c1[idx];

            for (int k = first[i]; k <= first[i] + spline.k; k++)
            {
                phi.data += w_in[idx][k] * Bvalues[i][k];
                d += w_in[idx][k] * dBvalues[i][k];
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
    const size_t n = nparams;
    const size_t Q = 2 * n + 1;

    for (size_t q = 0; q < Q; q++)
    {
        Neuron *acc = &npool.pool[n + q];
        for (size_t i = 1; i < n; i++)
            acc = &(*acc + npool.pool[n + i * Q + q]);
        hidden_layer.push_back(acc);
    }
}

void KAN::update_outer_grid(const std::vector<std::vector<double>> &X,
                            const std::vector<std::vector<double>> &w_in,
                            const std::vector<double> &c1,
                            double margin)
{
    double lo = std::numeric_limits<double>::max();
    double hi = std::numeric_limits<double>::lowest();

    for (const auto &x : X)
    {
        npool.pool.clear();
        for (double v : x)
            npool.append(Neuron(v));
        hidden_layer.clear();

        compute_first_layer(w_in, c1);
        compute_hidden_neurons();

        for (Neuron *h : hidden_layer)
        {
            lo = std::min(lo, h->data);
            hi = std::max(hi, h->data);
        }
    }

    double w = hi - lo;
    if (w < 1e-6)
    {
        double mid = 0.5 * (lo + hi);
        lo = mid - 0.5;
        hi = mid + 0.5;
        w = 1.0;
    }
    outer_spline = Spline(spline.G, spline.k, lo - margin * w, hi + margin * w);
}

void KAN::compute_outer_layer(const std::vector<std::vector<double>> &w_out, const std::vector<double> &c2)
{
    const size_t nb = spline.G + spline.k;

    for (size_t q = 0; q < hidden_layer.size(); q++)
    {
        Neuron &h = *hidden_layer[q];
        Neuron &phi = silu(h);
        double s = phi.data;
        double d = c2[q] * silu_prime(h.data);
        phi.data *= c2[q];

        std::vector<double> B(nb);
        int outer_span = outer_spline.span(h.data);
        for (int m = outer_span - spline.k; m <= outer_span; m++)
        {
            B[m] = outer_spline.coxDeBoor(m, spline.k, h.data);
            phi.data += w_out[q][m] * B[m];
            d += w_out[q][m] * dB(outer_spline, m, h.data);
        }
        phi.op = "spline";
        phi.local_grad = {d, s};
        phi.local_grad.insert(phi.local_grad.end(), B.begin(), B.end());
        outer_layer.push_back(&phi);
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

double KAN::forward(const std::vector<double> &x, const std::vector<std::vector<double>> &w_in, const std::vector<double> &c1, const std::vector<std::vector<double>> &w_out, const std::vector<double> &c2)
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

void KAN::train(const std::vector<std::vector<double>> &X, const std::vector<double> &Y, double lr, int steps, std::vector<std::vector<double>> &w_in, std::vector<double> &c1, std::vector<std::vector<double>> &w_out, std::vector<double> &c2)
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
                for (size_t j = 0; j < w_out[i].size(); j++)
                    w_out[i][j] -= lr * phi.grad * phi.local_grad[2 + j];
            }
        }
        std::cout << "step " << step << " loss " << total / X.size() << "\n";
    }
}
