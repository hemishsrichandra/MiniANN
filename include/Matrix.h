#pragma once
#include <vector>
#include <functional>
#include <iostream>

using namespace std;


class Matrix {
private:
    int rows;
    int cols;
    vector<vector<double>> data;

public:
    Matrix();
    Matrix(int rows, int cols);
    Matrix(int rows, int cols, double value);
    Matrix(const vector<vector<double>>& data);

    int getRows() const { return rows; }
    int getCols() const { return cols; }

    // Element access
    double& operator()(int i, int j);
    double operator()(int i, int j) const;

    // Operations
    Matrix operator+(const Matrix& other) const;
    Matrix operator-(const Matrix& other) const;
    Matrix operator*(const Matrix& other) const; // Matrix multiplication
    Matrix operator*(double scalar) const;       // Scalar multiplication
    Matrix operator+(double scalar) const;       // Scalar addition
    Matrix operator-(double scalar) const;       // Scalar subtraction

    // Hadamard product (element-wise multiplication)
    Matrix hadamard(const Matrix& other) const;
    
    // Transpose
    Matrix transpose() const;

    // Apply function to each element (useful for activations)
    Matrix apply(function<double(double)> func) const;

    // Print
    void print() const;
};
