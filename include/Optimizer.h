#pragma once
#include "Matrix.h"

class IOptimizer {
public:
    virtual ~IOptimizer() = default;
    virtual void update(Matrix& params, const Matrix& gradients) = 0;
};

class SGD : public IOptimizer {
    private:
        double learningRate;

    public:
        SGD(double learningRate = 0.01);
        
        void update(Matrix& params, const Matrix& gradients) override;
};

class Momentum : public IOptimizer {
    private:
        double learningRate;
        double beta;
        Matrix velocity;
        bool initialized;

    public:
        Momentum(double learningRate = 0.01, double beta = 0.9);

        void update(Matrix& params, const Matrix& gradients) override;
};

class Adam : public IOptimizer {
    private:
        double learningRate;
        double beta1;
        double beta2;
        double epsilon;
        Matrix m;
        Matrix v;
        int t;
        bool initialized;

    public:
        Adam(
            double learningRate = 0.001,
            double beta1 = 0.9,
            double beta2 = 0.999,
            double epsilon = 1e-8
        );

        void update(Matrix& params, const Matrix& gradients) override;
};