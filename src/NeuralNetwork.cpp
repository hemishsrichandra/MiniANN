#include "NeuralNetwork.h"
using namespace std;

void NeuralNetwork::addLayer(const Layer& layer) {
    layers.push_back(layer);
}

vector<double> NeuralNetwork::predict(const vector<double>& inputs) {
    vector<double> current_inputs = inputs;
    for (auto& layer : layers) {
        current_inputs = layer.forward(current_inputs);
    }
    return current_inputs;
}

void NeuralNetwork::train(const vector<vector<double>>& X, const vector<vector<double>>& y, int epochs, double learningRate) {
    // Backpropagation implementation
}