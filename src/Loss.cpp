#include "../include/Loss.h"
#include<cmath>

double MSE::calculate(const Matrix& predictions, const Matrix& targets) {
    double sum = 0.0;
    int rows = predictions.getRows();
    int cols = predictions.getCols();
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            double difference = predictions(i, j) - targets(i, j);
            sum += difference * difference;
        }
    }
    return sum / (rows * cols);
}

Matrix MSE::derivative(const Matrix& predictions, const Matrix& targets) {
    int rows = predictions.getRows();
    int cols = predictions.getCols();
    Matrix result(rows, cols);
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            result(i, j) = 2.0 * (predictions(i, j) - targets(i, j)) / (rows * cols);
        }
    }
    return result;
}

double BinaryCrossEntropy::calculate(const Matrix& predictions, const Matrix& targets) {
    double sum = 0.0;
    int rows = predictions.getRows();
    int cols = predictions.getCols();
    const double epsilon = 1e-12;
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            double p = predictions(i, j);
            double y = targets(i, j);
            p = max(epsilon, min(1.0 - epsilon, p));
            sum += -(y * log(p) + (1.0 - y) * log(1.0 - p));
        }
    }
    return sum / (rows * cols);
}

Matrix BinaryCrossEntropy::derivative(const Matrix& predictions, const Matrix& targets) {
    int rows = predictions.getRows();
    int cols = predictions.getCols();
    Matrix result(rows, cols);
    const double epsilon = 1e-12;
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            double p = predictions(i, j);
            double y = targets(i, j);
            p = max(epsilon, min(1.0 - epsilon, p));
            result(i, j) = (-y / p + (1.0 - y) / (1.0 - p)) / (rows * cols);
        }
    }
    return result;
}

double CategoricalCrossEntropy::calculate(const Matrix& predictions, const Matrix& targets) {
    double sum = 0.0;
    int rows = predictions.getRows();
    int cols = predictions.getCols();
    const double epsilon = 1e-12;
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            double p = predictions(i, j);
            double y = targets(i, j);
            p = max(epsilon, min(1.0 - epsilon, p));
            sum += -y * log(p);
        }
    }
    return sum / (rows * cols);
}

Matrix CategoricalCrossEntropy::derivative(
    const Matrix& predictions,
    const Matrix& targets) {
    int rows = predictions.getRows();
    int cols = predictions.getCols();
    Matrix result(rows, cols);
    const double epsilon = 1e-12;
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            double p = predictions(i, j);
            double y = targets(i, j);
            p = max(epsilon, min(1.0 - epsilon, p));
            result(i, j) = (-y / p) / (rows * cols);
        }
    }
    return result;
}