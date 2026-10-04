#pragma once

#include "Layer.h"
#include "Matrix.h"
#include "Activation.h"
#include "Loss.h"
#include "Optimizer.h"
#include "WeightInitializer.h"
#include "DataLoader.h"
#include <vector>
#include <memory>
#include <string>

using namespace std;


class NeuralNetwork {
private:
    vector<Layer> layers;
    shared_ptr<ILoss> lossFunction;
    shared_ptr<IOptimizer> defaultOptimizer;
    string optimizerType;
    double learningRate;
    double beta1;
    double beta2;
    vector<double> lossHistory;

    // Helper to configure optimizers for layers if not already set
    void configureLayerOptimizers();

public:
    NeuralNetwork();

    // Layer management
    void addLayer(const Layer& layer);
    void addLayer(int inputSize, int outputSize, 
                  shared_ptr<IActivation> activationFunction = nullptr, 
                  shared_ptr<IWeightInitializer> weightInitializer = nullptr);

    // Network configuration
    void setLoss(shared_ptr<ILoss> loss);
    void setOptimizer(shared_ptr<IOptimizer> opt);
    void setOptimizer(const string& type, double lr = 0.01, double beta1 = 0.9, double beta2 = 0.999);
    void setLearningRate(double lr);

    // Forward and prediction
    Matrix forward(const Matrix& input);
    Matrix predict(const Matrix& input);

    // Backpropagation
    void backward(const Matrix& lossGrad, const Matrix& targets);

    // Training methods
    vector<double> train(const Matrix& X, const Matrix& Y, int epochs, int batchSize = 32, bool verbose = true);
    vector<double> train(DataLoader& dataLoader, int epochs, int batchSize = 32, bool verbose = true);
    vector<double> train(DataLoader& trainLoader, DataLoader& valLoader, int epochs, int batchSize = 32, bool verbose = true);

    // Evaluation
    double evaluate(const Matrix& X, const Matrix& Y);
    double evaluate(DataLoader& dataLoader);

    // Inspection
    int getNumLayers() const { return static_cast<int>(layers.size()); }
    const vector<Layer>& getLayers() const { return layers; }
    vector<Layer>& getLayers() { return layers; }
    const vector<double>& getLossHistory() const { return lossHistory; }
    void summary() const;

    bool saveModel(const string& filename) const;
    bool loadModel(const string& filename);
    string getLossName() const;
};