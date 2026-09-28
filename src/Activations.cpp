#include"Activations.h"
#include<cmath>
#include<algorithm>
using namespace std;

double Sigmoid::activate(double x) const {
    return 1.0 / (1.0 + exp(-x));
}

double Sigmoid::derivative(double x) const {
    double temp = Sigmoid::activate(x);
    return temp * (1 - temp);
}

double ReLU::activate(double x) const {
    return x >= 0.0 ? x : 0.0;
}

double ReLU::derivative(double x) const {
    return x >= 0.0 ? 1.0 : 0.0;
}

double Tanh::activate(double x) const {
    return tanh(x);
}

double Tanh::derivative(double x) const {
    double temp = tanh(x);
    return 1.0 - temp * temp;
}