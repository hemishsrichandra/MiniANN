#include "Layer.h"
using namespace std;

Layer::Layer(int numNeurons, int numInputsPerNeuron, shared_ptr<IActivation> act) {
    neurons.reserve(numNeurons);
    for (int i = 0; i < numNeurons; i++) {
        neurons.emplace_back(numInputsPerNeuron, act);
    }
}

vector<double> Layer::forward(const vector<double> &inputs) {
  vector<double> outputs;
  outputs.reserve(neurons.size());
  for (auto &neuron : neurons) {
    outputs.push_back(neuron.forward(inputs));
  }
  return outputs;
}

vector<Neuron> &Layer::getNeurons() { return neurons; }