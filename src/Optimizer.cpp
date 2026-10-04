#include "Optimizer.h"
#include <cmath>

SGD::SGD(double learningRate) {
    this->learningRate = learningRate;
}

void SGD::update(Matrix& params, const Matrix& gradients) {
    params = params - gradients * learningRate;
}

Momentum::Momentum(double learningRate, double beta) {
    this->learningRate = learningRate;
    this->beta = beta;
    this->initialized = false;
}

void Momentum::update(Matrix& params, const Matrix& gradients) {
    if (!initialized) {
        velocity = Matrix(gradients.getRows(), gradients.getCols(), 0.0);
        initialized = true;
    }
    velocity = velocity * beta + gradients * (1.0 - beta);
    params = params - velocity * learningRate;
}

Adam::Adam(double learningRate, double beta1, double beta2, double epsilon) {
    this->learningRate = learningRate;
    this->beta1 = beta1;
    this->beta2 = beta2;
    this->epsilon = epsilon;
    this->t = 0;
    this->initialized = false;
}

void Adam::update(Matrix& params, const Matrix& gradients) {
    if (!initialized) {
        m = Matrix(gradients.getRows(), gradients.getCols(), 0.0);
        v = Matrix(gradients.getRows(), gradients.getCols(), 0.0);
        initialized = true;
    }
    t++;
    m = m * beta1 + gradients * (1.0 - beta1);
    Matrix squaredGradients = gradients.apply([](double x) {
        return x * x;
    });
    v = v * beta2 + squaredGradients * (1.0 - beta2);
    double beta1Correction = 1.0 - pow(beta1, t);
    double beta2Correction = 1.0 - pow(beta2, t);
    Matrix mHat = m * (1.0 / beta1Correction);
    Matrix vHat = v * (1.0 / beta2Correction);
    Matrix update = mHat.apply([&](double x) {
        return x;
    });
    Matrix denominator = vHat.apply([&](double x) {
        return sqrt(x) + epsilon;
    });
    for (int i = 0; i < update.getRows(); i++) {
        for (int j = 0; j < update.getCols(); j++) {
            update(i, j) = mHat(i, j) / denominator(i, j);
        }
    }
    params = params - update * learningRate;
}