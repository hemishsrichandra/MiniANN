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
Matrix SoftMax::forward(const Matrix& input) {
    int rows = input.getRows();
    int cols = input.getCols();
    Matrix result(rows, cols);

    for (int j = 0; j < cols; ++j) {
        double maxVal = input(0, j);
        for (int i = 1; i < rows; ++i) {
            if (input(i, j) > maxVal) {
                maxVal = input(i, j);
            }
        }

        double sumExp = 0.0;
        for (int i = 0; i < rows; ++i) {
            double e = exp(input(i, j) - maxVal);
            result(i, j) = e;
            sumExp += e;
        }

        for (int i = 0; i < rows; ++i) {
            result(i, j) /= sumExp;
        }
    }
    return result;
}

Matrix SoftMax::derivative(const Matrix& input) {
    // This is a placeholder; actual derivative is typically combined with CategoricalCrossEntropy
    // in NeuralNetwork::backward for numerical stability and efficiency.
    return Matrix(input.getRows(), input.getCols(), 1.0);
}
