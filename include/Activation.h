#pragma once

#include "Matrix.h"

class IActivation {
public:
    virtual ~IActivation() = default;
    virtual Matrix forward(const Matrix& input) = 0;
    virtual Matrix derivative(const Matrix& input) = 0;
};

class Sigmoid : public IActivation {
    public:
        Matrix forward(const Matrix& input) override;
        Matrix derivative(const Matrix& input) override;
};

class ReLU : public IActivation {
    public:
        Matrix forward(const Matrix& input) override;
        Matrix derivative(const Matrix& input) override;
};

class Tanh : public IActivation {
    public:
        Matrix forward(const Matrix& input) override;
        Matrix derivative(const Matrix& input) override;
};