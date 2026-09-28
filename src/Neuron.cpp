#include "Neuron.h"
#include <random>
#include <utility>
using namespace std;

Neuron::Neuron(int numInputs, shared_ptr<IActivation> act)
    : bias(0.0), activation(move(act)), last_output(0.0) {

    random_device rd;
    mt19937 gen(rd());
    uniform_real_distribution<> dis(-1.0, 1.0);

    weights.reserve(numInputs);
    for (int i = 0; i < numInputs; i++) {
        weights.push_back(dis(gen));
    }
}

double Neuron::forward(const vector<double> &inputs) {
  double sum = bias;
  for (int i = 0; i < inputs.size(); i++) {
    sum += inputs[i] * weights[i];
  }
  last_output = activation->activate(sum);
  return last_output;
}

const vector<double> &Neuron::getWeights() const { return weights; }
void Neuron::setWeight(int index, double weight) { weights[index] = weight; }
double Neuron::getBias() const { return bias; }
void Neuron::setBias(double b) { bias = b; }
double Neuron::getLastOutput() const { return last_output; }