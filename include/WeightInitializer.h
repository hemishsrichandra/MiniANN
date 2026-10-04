#pragma once
#include "Matrix.h"
#include <string>

class IWeightInitializer {
public:
    virtual ~IWeightInitializer() = default;
    virtual void initialize(Matrix& weights, int fanIn, int fanOut) = 0;
};

class RandomUniform : public IWeightInitializer {
private:
    double min_val;
    double max_val;
public:
    RandomUniform(double min = -0.5, double max = 0.5);
    void initialize(Matrix& weights, int fanIn, int fanOut) override;
};

class Xavier : public IWeightInitializer {
public:
    void initialize(Matrix& weights, int fanIn, int fanOut) override;
};

class He : public IWeightInitializer {
public:
    void initialize(Matrix& weights, int fanIn, int fanOut) override;
};
