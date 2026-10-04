#pragma once

#include "Matrix.h"

class ILoss {
public:
    virtual ~ILoss() = default;
    virtual double calculate(const Matrix& predictions, const Matrix& targets) = 0;
    virtual Matrix derivative(const Matrix& predictions, const Matrix& targets) = 0;
};

class MSE : public ILoss {
    public:
        double calculate(const Matrix& predictions, const Matrix& targets) override;
        Matrix derivative(const Matrix& predictions, const Matrix& targets) override;
};

class BinaryCrossEntropy : public ILoss {
    public:
        double calculate(const Matrix& predictions, const Matrix& targets) override;
        Matrix derivative(const Matrix& predictions, const Matrix& targets) override;
};

class CategoricalCrossEntropy : public ILoss {
    public:
        double calculate(const Matrix& predictions, const Matrix& targets) override;
        Matrix derivative(const Matrix& predictions, const Matrix& targets) override;
};