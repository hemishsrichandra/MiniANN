#include "../include/Activation.h"
#include <cmath>

Matrix Sigmoid::forward(const Matrix& input) {
    return input.apply([](double x) {
        return 1.0 / (1.0 + exp(-x));
    });
}

Matrix Sigmoid::derivative(const Matrix& input) {
    return input.apply([](double x) {
        double s = 1.0 / (1.0 + exp(-x));
        return s * (1.0 - s);
    });
}

Matrix ReLU::forward(const Matrix& input) {
    return input.apply([](double x) {
        return x > 0 ? x : 0.0;
    });
}

Matrix ReLU::derivative(const Matrix& input) {
    return input.apply([](double x) {
        return x > 0 ? 1.0 : 0.0;
    });
}

Matrix Tanh::forward(const Matrix& input) {
    return input.apply([](double x) {
        return tanh(x);
    });
}

Matrix Tanh::derivative(const Matrix& input) {
    return input.apply([](double x) {
        double t = tanh(x);
        return 1.0 - t * t;
    });
}