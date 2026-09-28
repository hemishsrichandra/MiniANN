#ifndef MINIANN_NEURALNETWORK_H
#define MINIANN_NEURALNETWORK_H

#include<iostream>
#include<vector>
#include "Layer.h"
using namespace std;

class NeuralNetwork {
private:
    vector <Layer > layers;
public:
    void addLayer(const Layer& layer);
    vector <double > predict(const std::vector <double >& inputs);
    void train(const vector <std::vector <double >>& X,
    const vector <std::vector <double >>& y, int epochs , double learningRate);
};

#endif // MINIANN_NEURALNETWORK_H   