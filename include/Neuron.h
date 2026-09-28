#ifndef MINIANN_NEURON_H
#define MINIANN_NEURON_H

#include<iostream>
#include<vector>
#include "IActivation.h"
using namespace std;

class Neuron {
private:
    vector<double> weights;
    double bias;
    shared_ptr<IActivation> activation;
    double last_output; // cached for learning
public:
    Neuron(int numInputs, shared_ptr<IActivation> act);
    double forward(const std::vector <double >& inputs);
    // Getters and Setters
    const vector<double>& getWeights() const;
    void setWeight(int index, double weight);
    double getBias() const;
    void setBias(double b);
    double getLastOutput() const;
};

#endif // MINIANN_NEURON_H