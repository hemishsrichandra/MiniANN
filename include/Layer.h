#pragma once
#include "Matrix.h"
#include "Activation.h"
#include "Optimizer.h"
#include "WeightInitializer.h"
#include <memory>

using namespace std;


class NeuralNetwork;

class Layer {
    friend class NeuralNetwork;
private:
    Matrix weights;
    Matrix biases;
    shared_ptr<IActivation> activation;
    shared_ptr<IOptimizer> optimizer_weights;
    shared_ptr<IOptimizer> optimizer_biases;
    Matrix inputCache; // Store inputs for backprop
    Matrix zCache;     // Store z (wx+b) for backprop

public:
    Layer(int inputSize, int outputSize, 
          shared_ptr<IActivation> activationFunction, 
          shared_ptr<IWeightInitializer> weightInitializer);

    void setOptimizer(shared_ptr<IOptimizer> optWeights, shared_ptr<IOptimizer> optBiases);

    Matrix forward(const Matrix& input);

    Matrix getWeights() const { return weights; }
    Matrix getBiases() const { return biases; }
    void setWeights(const Matrix& w) { weights = w; }
    void setBiases(const Matrix& b) { biases = b; }
    shared_ptr<IActivation> getActivation() const { return activation; }
};