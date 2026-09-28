#ifndef MINIANN_LAYER_H
#define MINIANN_LAYER_H

#include<iostream>
#include "Neuron.h"
using namespace std;

class Layer {
private:
    vector<Neuron> neurons;
public:
    Layer(int numNuerons, int numInputsPerNeuron, shared_ptr <IActivation > act);
    vector <double > forward(const vector <double >& inputs);
    vector<Neuron> &getNeurons();
};

#endif // MINIANN_LAYER_H