#include "../include/NeuralNetwork.h"
#include <iostream>
#include <iomanip>
#include <stdexcept>
#include <algorithm>

NeuralNetwork::NeuralNetwork() 
    : lossFunction(make_shared<MSE>()),
      defaultOptimizer(make_shared<SGD>(0.01)),
      optimizerType("sgd"),
      learningRate(0.01),
      beta1(0.9),
      beta2(0.999) {}

void NeuralNetwork::addLayer(const Layer& layer) {
    const size_t MAX_LAYERS = 10;
    if (layers.size() >= MAX_LAYERS) {
        throw runtime_error("NeuralNetwork::addLayer: Maximum number of layers (" + to_string(MAX_LAYERS) + ") exceeded.");
    }
    layers.push_back(layer);
}

void NeuralNetwork::addLayer(int inputSize, int outputSize, 
                            shared_ptr<IActivation> activationFunction, 
                            shared_ptr<IWeightInitializer> weightInitializer) {
    const size_t MAX_LAYERS = 10;
    const int MAX_NEURONS = 1024;
    
    if (layers.size() >= MAX_LAYERS) {
        throw runtime_error("NeuralNetwork::addLayer: Maximum number of layers (" + to_string(MAX_LAYERS) + ") exceeded.");
    }
    if (inputSize <= 0 || outputSize <= 0) {
        throw invalid_argument("NeuralNetwork::addLayer: Neuron counts must be positive.");
    }
    if (inputSize > MAX_NEURONS || outputSize > MAX_NEURONS) {
        throw invalid_argument("NeuralNetwork::addLayer: Neuron count exceeds maximum limit of " + to_string(MAX_NEURONS) + ".");
    }

    // If no weight initializer provided, use Xavier by default
    if (!weightInitializer) {
        weightInitializer = make_shared<Xavier>();
    }
    layers.emplace_back(inputSize, outputSize, activationFunction, weightInitializer);
}

void NeuralNetwork::setLoss(shared_ptr<ILoss> loss) {
    if (!loss) {
        throw invalid_argument("NeuralNetwork::setLoss: loss cannot be null");
    }
    this->lossFunction = loss;
}

void NeuralNetwork::setOptimizer(shared_ptr<IOptimizer> opt) {
    if (!opt) {
        throw invalid_argument("NeuralNetwork::setOptimizer: optimizer cannot be null");
    }
    this->defaultOptimizer = opt;
    // Give each layer its own independent clone so stateful optimizers
    // (Adam, Momentum) don't share accumulators across layers.
    for (auto& layer : layers) {
        layer.setOptimizer(opt->clone(), opt->clone());
    }
}

void NeuralNetwork::setOptimizer(const string& type, double lr, double b1, double b2) {
    if (lr < 1e-6 || lr > 10.0) {
        throw invalid_argument("NeuralNetwork::setOptimizer: Learning rate must be between 1e-6 and 10.0.");
    }
    this->optimizerType = type;
    this->learningRate = lr;
    this->beta1 = b1;
    this->beta2 = b2;

    string optLower = type;
    transform(optLower.begin(), optLower.end(), optLower.begin(), ::tolower);

    for (auto& layer : layers) {
        if (optLower == "adam") {
            layer.setOptimizer(make_shared<Adam>(lr, b1, b2), make_shared<Adam>(lr, b1, b2));
        } else if (optLower == "momentum") {
            layer.setOptimizer(make_shared<Momentum>(lr, b1), make_shared<Momentum>(lr, b1));
        } else {
            cerr << "[NeuralNetwork] Warning: Unknown optimizer \"" << type
                 << "\". Falling back to SGD." << endl;
            layer.setOptimizer(make_shared<SGD>(lr), make_shared<SGD>(lr));
        }
    }
}

void NeuralNetwork::setLearningRate(double lr) {
    if (lr < 1e-6 || lr > 10.0) {
        throw invalid_argument("NeuralNetwork::setLearningRate: Learning rate must be between 1e-6 and 10.0.");
    }
    this->learningRate = lr;
    setOptimizer(this->optimizerType, lr, this->beta1, this->beta2);
}

void NeuralNetwork::configureLayerOptimizers() {
    string optLower = optimizerType;
    transform(optLower.begin(), optLower.end(), optLower.begin(), ::tolower);

    for (size_t i = 0; i < layers.size(); ++i) {
        Layer& layerInt = layers[i];
        if (!layerInt.optimizer_weights || !layerInt.optimizer_biases) {
            if (optLower == "adam") {
                layers[i].setOptimizer(make_shared<Adam>(learningRate, beta1, beta2),
                                       make_shared<Adam>(learningRate, beta1, beta2));
            } else if (optLower == "momentum") {
                layers[i].setOptimizer(make_shared<Momentum>(learningRate, beta1),
                                       make_shared<Momentum>(learningRate, beta1));
            } else {
                layers[i].setOptimizer(make_shared<SGD>(learningRate),
                                       make_shared<SGD>(learningRate));
            }
        }
    }
}

Matrix NeuralNetwork::forward(const Matrix& input) {
    if (layers.empty()) {
        throw runtime_error("NeuralNetwork::forward: network has no layers");
    }

    Matrix current = input;
    for (size_t i = 0; i < layers.size(); ++i) {
        current = layers[i].forward(current);
    }
    return current;
}

Matrix NeuralNetwork::predict(const Matrix& input) {
    return forward(input);
}

void NeuralNetwork::backward(const Matrix& lossGrad) {
    if (layers.empty()) {
        throw runtime_error("NeuralNetwork::backward: network has no layers");
    }

    Matrix dA = lossGrad;
    int numLayers = static_cast<int>(layers.size());

    for (int i = numLayers - 1; i >= 0; --i) {
        Layer& layerInt = layers[i];

        // 1. Calculate dZ
        Matrix dZ;
        if (layerInt.activation) {
            Matrix actDeriv = layerInt.activation->derivative(layerInt.zCache);
            dZ = dA.hadamard(actDeriv);
        } else {
            dZ = dA;
        }

        // 2. Calculate gradients: dW = dZ * (inputCache)^T
        Matrix inputT = layerInt.inputCache.transpose();
        Matrix dW = dZ * inputT;

        // 3. Calculate db = sum of dZ columns
        Matrix db(dZ.getRows(), 1, 0.0);
        for (int r = 0; r < dZ.getRows(); ++r) {
            for (int c = 0; c < dZ.getCols(); ++c) {
                db(r, 0) += dZ(r, c);
            }
        }

        // 4. Calculate dA for previous layer before modifying weights
        if (i > 0) {
            dA = layerInt.weights.transpose() * dZ;
        }

        // 5. Update weights and biases using layer optimizers
        if (layerInt.optimizer_weights) {
            layerInt.optimizer_weights->update(layerInt.weights, dW);
        }
        if (layerInt.optimizer_biases) {
            layerInt.optimizer_biases->update(layerInt.biases, db);
        }
    }
}

vector<double> NeuralNetwork::train(const Matrix& X, const Matrix& Y, int epochs, int batchSize, bool verbose) {
    if (layers.empty()) {
        throw runtime_error("NeuralNetwork::train: no layers in network");
    }
    if (!lossFunction) {
        throw runtime_error("NeuralNetwork::train: no loss function set");
    }
    if (epochs <= 0 || epochs > 10000) {
        throw invalid_argument("NeuralNetwork::train: epochs must be between 1 and 10000.");
    }
    if (batchSize <= 0 || batchSize > 65536) {
        throw invalid_argument("NeuralNetwork::train: batchSize must be between 1 and 65536.");
    }
    if (X.getCols() != Y.getCols()) {
        throw invalid_argument("NeuralNetwork::train: X and Y sample counts do not match");
    }

    configureLayerOptimizers();
    lossHistory.clear();
    lossHistory.reserve(epochs);

    int numSamples = X.getCols();
    int numFeatures = X.getRows();
    int numTargets = Y.getRows();
    int totalBatches = (numSamples + batchSize - 1) / batchSize;

    int printStep = max(1, epochs / 10);

    for (int epoch = 1; epoch <= epochs; ++epoch) {
        double totalLoss = 0.0;

        for (int b = 0; b < totalBatches; ++b) {
            int startIdx = b * batchSize;
            int endIdx = min(startIdx + batchSize, numSamples);
            int curBatchSize = endIdx - startIdx;

            // Extract batch
            vector<vector<double>> bXData(numFeatures, vector<double>(curBatchSize));
            vector<vector<double>> bYData(numTargets, vector<double>(curBatchSize));

            for (int j = 0; j < curBatchSize; ++j) {
                int src = startIdx + j;
                for (int f = 0; f < numFeatures; ++f) bXData[f][j] = X(f, src);
                for (int t = 0; t < numTargets; ++t) bYData[t][j] = Y(t, src);
            }

            Matrix batchX(bXData);
            Matrix batchY(bYData);

            // Forward
            Matrix preds = forward(batchX);

            // Loss
            double bLoss = lossFunction->calculate(preds, batchY);
            totalLoss += bLoss;

            // Backward
            Matrix grad = lossFunction->derivative(preds, batchY);
            backward(grad);
        }

        double avgLoss = totalLoss / totalBatches;
        lossHistory.push_back(avgLoss);

        if (verbose && (epoch % printStep == 0 || epoch == 1 || epoch == epochs)) {
            cout << "Epoch [" << setw(5) << epoch << "/" << epochs << "] "
                 << "- Loss: " << fixed << setprecision(6) << avgLoss << endl;
        }
    }

    return lossHistory;
}

vector<double> NeuralNetwork::train(DataLoader& dataLoader, int epochs, int batchSize, bool verbose) {
    return train(dataLoader.getX(), dataLoader.getY(), epochs, batchSize, verbose);
}

vector<double> NeuralNetwork::train(DataLoader& trainLoader, DataLoader& valLoader, int epochs, int batchSize, bool verbose) {
    if (layers.empty()) {
        throw runtime_error("NeuralNetwork::train: no layers in network");
    }
    if (!lossFunction) {
        throw runtime_error("NeuralNetwork::train: no loss function set");
    }

    configureLayerOptimizers();
    lossHistory.clear();
    lossHistory.reserve(epochs);

    const Matrix& X = trainLoader.getX();
    const Matrix& Y = trainLoader.getY();
    int numSamples = X.getCols();
    int numFeatures = X.getRows();
    int numTargets = Y.getRows();
    int totalBatches = (numSamples + batchSize - 1) / batchSize;

    int printStep = max(1, epochs / 10);

    for (int epoch = 1; epoch <= epochs; ++epoch) {
        double totalLoss = 0.0;

        for (int b = 0; b < totalBatches; ++b) {
            int startIdx = b * batchSize;
            int endIdx = min(startIdx + batchSize, numSamples);
            int curBatchSize = endIdx - startIdx;

            vector<vector<double>> bXData(numFeatures, vector<double>(curBatchSize));
            vector<vector<double>> bYData(numTargets, vector<double>(curBatchSize));

            for (int j = 0; j < curBatchSize; ++j) {
                int src = startIdx + j;
                for (int f = 0; f < numFeatures; ++f) bXData[f][j] = X(f, src);
                for (int t = 0; t < numTargets; ++t) bYData[t][j] = Y(t, src);
            }

            Matrix batchX(bXData);
            Matrix batchY(bYData);

            Matrix preds = forward(batchX);
            double bLoss = lossFunction->calculate(preds, batchY);
            totalLoss += bLoss;

            Matrix grad = lossFunction->derivative(preds, batchY);
            backward(grad);
        }

        double trainLoss = totalLoss / totalBatches;
        lossHistory.push_back(trainLoss);

        if (verbose && (epoch % printStep == 0 || epoch == 1 || epoch == epochs)) {
            Matrix valPreds = forward(valLoader.getX());
            double valLoss = lossFunction->calculate(valPreds, valLoader.getY());

            cout << "Epoch [" << setw(5) << epoch << "/" << epochs << "] "
                 << "- Train Loss: " << fixed << setprecision(6) << trainLoss
                 << " | Val Loss: " << valLoss << endl;
        }
    }

    return lossHistory;
}

double NeuralNetwork::evaluate(const Matrix& X, const Matrix& Y) {
    if (!lossFunction) {
        throw runtime_error("NeuralNetwork::evaluate: no loss function set");
    }
    Matrix preds = forward(X);
    return lossFunction->calculate(preds, Y);
}

double NeuralNetwork::evaluate(DataLoader& dataLoader) {
    return evaluate(dataLoader.getX(), dataLoader.getY());
}

void NeuralNetwork::summary() const {
    cout << "\n================ Neural Network Architecture ================" << endl;
    cout << left << setw(8) << "Layer"
         << setw(16) << "Weights Shape"
         << setw(16) << "Biases Shape"
         << setw(16) << "Parameters" << endl;
    cout << string(56, '-') << endl;

    int totalParams = 0;
    for (size_t i = 0; i < layers.size(); ++i) {
        Matrix w = layers[i].getWeights();
        Matrix b = layers[i].getBiases();
        int params = (w.getRows() * w.getCols()) + (b.getRows() * b.getCols());
        totalParams += params;

        string wShape = "(" + to_string(w.getRows()) + ", " + to_string(w.getCols()) + ")";
        string bShape = "(" + to_string(b.getRows()) + ", " + to_string(b.getCols()) + ")";

        cout << left << setw(8) << (i + 1)
             << setw(16) << wShape
             << setw(16) << bShape
             << setw(16) << params << endl;
    }
    cout << string(56, '-') << endl;
    cout << "Total Trainable Parameters: " << totalParams << endl;
    cout << "Optimizer: " << optimizerType << " (lr=" << learningRate << ")" << endl;
    cout << "============================================================\n" << endl;
}