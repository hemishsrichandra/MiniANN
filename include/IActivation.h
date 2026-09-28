#ifndef MINIANN_IACTIVATION_H
#define MINIANN_IACTIVATION_H

class IActivation {
public:
    virtual ~IActivation() = default;
    virtual double activate(double x) const = 0;
    virtual double derivative(double x) const = 0; // for backprop
};

#endif // MINIANN_IACTIVATION_H