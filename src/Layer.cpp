#include "../include/Layer.h"
#include <stdexcept>

Layer::Layer(int inputSize, int outputSize, 
             shared_ptr<IActivation> activationFunction, 
             shared_ptr<IWeightInitializer> weightInitializer)
    : weights(outputSize, inputSize), biases(outputSize, 1), activation(activationFunction) {
    
    if (weightInitializer) {
        weightInitializer->initialize(weights, inputSize, outputSize);
        
        // Biases are initialized to zero
        for(int i = 0; i < outputSize; ++i) {
            biases(i, 0) = 0.0;
        }
    }
}

void Layer::setOptimizer(shared_ptr<IOptimizer> optWeights, shared_ptr<IOptimizer> optBiases) {
    this->optimizer_weights = optWeights;
    this->optimizer_biases = optBiases;
}

Matrix Layer::forward(const Matrix& input) {
    this->inputCache = input;
    
    Matrix z = weights * input;
    // Broadcast biases if needed (assuming input might be a batch where cols > 1)
    for(int i = 0; i < z.getRows(); ++i) {
        for(int j = 0; j < z.getCols(); ++j) {
            z(i, j) += biases(i, 0);
        }
    }
    
    this->zCache = z;
    
    if (activation) {
        return activation->forward(z);
    }
    return z;
}