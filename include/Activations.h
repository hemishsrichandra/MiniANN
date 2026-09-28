#ifndef MINIANN_ACTIVATIONS_H
#define MINIANN_ACTIVATIONS_H

#include "IActivation.h"

class Sigmoid : public IActivation {
public:
    double activate(double x) const override;
    double derivative(double x) const override;
};

class ReLU : public IActivation {
public:
    double activate(double x) const override;
    double derivative(double x) const override;
};

class Tanh : public IActivation {
public:
    double activate(double x) const override;
    double derivative(double x) const override;
};

#endif // MINIANN_ACTIVATIONS_H