#include "Matrix.h"
#include "Layer.h"
#include "Activation.h"
#include "Loss.h"
#include "Optimizer.h"
#include "WeightInitializer.h"
#include "DataLoader.h"
#include "NeuralNetwork.h"
#include "Evaluator.h"

#include <iostream>
#include <cassert>
#include <cmath>
#include <vector>
#include <memory>

using namespace std;

#define TEST_ASSERT(cond, msg) \
    do { \
        if (!(cond)) { \
            cerr << "[FAILED] " << msg << " (" << __FILE__ << ":" << __LINE__ << ")" << endl; \
            return false; \
        } \
    } while(0)

bool test_matrix_layer_integration() {
    cout << "[TEST] Matrix -> Layer Integration..." << endl;
    Matrix input(3, 1, 1.0);
    Layer layer(3, 2, make_shared<ReLU>(), make_shared<Xavier>());
    Matrix out = layer.forward(input);
    TEST_ASSERT(out.getRows() == 2, "Layer output rows must match outputSize");
    TEST_ASSERT(out.getCols() == 1, "Layer output cols must match input cols");
    cout << "  -> PASSED" << endl;
    return true;
}

bool test_activation_layer_network() {
    cout << "[TEST] Activation -> Layer / Network Integration..." << endl;
    NeuralNetwork nn;
    nn.addLayer(4, 8, make_shared<Tanh>(), make_shared<Xavier>());
    nn.addLayer(8, 4, make_shared<ReLU>(), make_shared<He>());
    nn.addLayer(4, 2, make_shared<Sigmoid>(), make_shared<RandomUniform>());

    Matrix input(4, 5, 0.5); // 4 features, 5 samples
    Matrix out = nn.forward(input);
    TEST_ASSERT(out.getRows() == 2, "Output rows must be 2");
    TEST_ASSERT(out.getCols() == 5, "Output cols must be 5");
    // Check Sigmoid output range [0, 1]
    for (int r = 0; r < 2; ++r) {
        for (int c = 0; c < 5; ++c) {
            TEST_ASSERT(out(r, c) >= 0.0 && out(r, c) <= 1.0, "Sigmoid outputs must be within [0, 1]");
        }
    }
    cout << "  -> PASSED" << endl;
    return true;
}

bool test_loss_optimizers_training() {
    cout << "[TEST] Loss & Optimizers -> Training Integration..." << endl;

    // Test XOR problem with SGD, Momentum, and Adam
    vector<vector<double>> xData = {
        {0.0, 0.0, 1.0, 1.0},
        {0.0, 1.0, 0.0, 1.0}
    };
    vector<vector<double>> yData = {
        {0.0, 1.0, 1.0, 0.0}
    };
    Matrix X(xData);
    Matrix Y(yData);

    // 1. Test with Adam & MSE
    {
        NeuralNetwork nn;
        nn.addLayer(2, 4, make_shared<Tanh>(), make_shared<Xavier>());
        nn.addLayer(4, 1, make_shared<Sigmoid>(), make_shared<Xavier>());
        nn.setLoss(make_shared<MSE>());
        nn.setOptimizer("adam", 0.1);

        double initialLoss = nn.evaluate(X, Y);
        auto history = nn.train(X, Y, 200, 4, false);
        double finalLoss = nn.evaluate(X, Y);

        TEST_ASSERT(finalLoss < initialLoss, "Adam training must decrease loss");
        cout << "  Adam Training Loss: " << initialLoss << " -> " << finalLoss << endl;
    }

    // 2. Test with Momentum & BinaryCrossEntropy
    {
        NeuralNetwork nn;
        nn.addLayer(2, 4, make_shared<Tanh>(), make_shared<Xavier>());
        nn.addLayer(4, 1, make_shared<Sigmoid>(), make_shared<Xavier>());
        nn.setLoss(make_shared<BinaryCrossEntropy>());
        nn.setOptimizer("momentum", 0.05, 0.9);

        double initialLoss = nn.evaluate(X, Y);
        auto history = nn.train(X, Y, 100, 4, false);
        double finalLoss = nn.evaluate(X, Y);

        TEST_ASSERT(finalLoss < initialLoss, "Momentum training must decrease loss");
        cout << "  Momentum Training Loss: " << initialLoss << " -> " << finalLoss << endl;
    }

    cout << "  -> PASSED" << endl;
    return true;
}

bool test_dataloader_integration() {
    cout << "[TEST] DataLoader -> NeuralNetwork Integration..." << endl;
    DataLoader loader;
    bool loaded = loader.loadCSV("data/sample_dataset.csv", -1, true, ',');
    if (!loaded) {
        loaded = loader.loadCSV("../data/sample_dataset.csv", -1, true, ',');
    }
    TEST_ASSERT(loaded, "DataLoader must load CSV successfully");
    TEST_ASSERT(loader.getNumSamples() == 60, "Dataset should have 60 samples");
    TEST_ASSERT(loader.getNumFeatures() == 4, "Dataset should have 4 features");

    loader.normalize();
    loader.oneHotEncode(3);
    TEST_ASSERT(loader.getNumTargets() == 3, "One-hot targets should have 3 classes");

    auto split = loader.split(0.75, true);
    DataLoader trainL = split.first;
    DataLoader testL = split.second;

    TEST_ASSERT(trainL.getNumSamples() == 45, "Train samples should be 45");
    TEST_ASSERT(testL.getNumSamples() == 15, "Test samples should be 15");

    // Minibatch generation
    auto batches = trainL.getBatches(10);
    TEST_ASSERT(batches.size() == 5, "45 samples with batch size 10 should produce 5 batches");
    TEST_ASSERT(batches[0].first.getCols() == 10, "First batch size should be 10");
    TEST_ASSERT(batches[4].first.getCols() == 5, "Last batch size should be 5");

    cout << "  -> PASSED" << endl;
    return true;
}

bool test_evaluator_metrics() {
    cout << "[TEST] Evaluator Metrics..." << endl;

    // Perfect predictions test
    Matrix preds(3, 4);
    preds(0, 0) = 0.9; preds(1, 0) = 0.1; preds(2, 0) = 0.0; // class 0
    preds(0, 1) = 0.1; preds(1, 1) = 0.8; preds(2, 1) = 0.1; // class 1
    preds(0, 2) = 0.0; preds(1, 2) = 0.2; preds(2, 2) = 0.8; // class 2
    preds(0, 3) = 0.1; preds(1, 3) = 0.7; preds(2, 3) = 0.2; // class 1

    Matrix targets(3, 4, 0.0);
    targets(0, 0) = 1.0;
    targets(1, 1) = 1.0;
    targets(2, 2) = 1.0;
    targets(1, 3) = 1.0;

    double acc = Evaluator::accuracy(preds, targets);
    TEST_ASSERT(fabs(acc - 1.0) < 1e-6, "Accuracy on perfect predictions must be 1.0");

    double prec = Evaluator::precision(preds, targets, -1);
    TEST_ASSERT(fabs(prec - 1.0) < 1e-6, "Precision on perfect predictions must be 1.0");

    double rec = Evaluator::recall(preds, targets, -1);
    TEST_ASSERT(fabs(rec - 1.0) < 1e-6, "Recall on perfect predictions must be 1.0");

    double f1 = Evaluator::f1Score(preds, targets, -1);
    TEST_ASSERT(fabs(f1 - 1.0) < 1e-6, "F1 on perfect predictions must be 1.0");

    Matrix cm = Evaluator::confusionMatrix(preds, targets, 3);
    TEST_ASSERT(cm(0, 0) == 1.0 && cm(1, 1) == 2.0 && cm(2, 2) == 1.0, "Confusion matrix diagonal must match");

    cout << "  -> PASSED" << endl;
    return true;
}

bool test_full_end_to_end_pipeline() {
    cout << "[TEST] Full End-to-End Pipeline..." << endl;
    DataLoader loader;
    bool loaded = loader.loadCSV("data/sample_dataset.csv", -1, true, ',');
    if (!loaded) loader.loadCSV("../data/sample_dataset.csv", -1, true, ',');
    TEST_ASSERT(loaded, "CSV must be loaded");

    loader.normalize();
    loader.oneHotEncode(3);

    auto split = loader.split(0.8, true);
    DataLoader trainL = split.first;
    DataLoader testL = split.second;

    NeuralNetwork nn;
    nn.addLayer(4, 8, make_shared<ReLU>(), make_shared<He>());
    nn.addLayer(8, 3, make_shared<Sigmoid>(), make_shared<Xavier>());
    nn.setLoss(make_shared<MSE>());
    nn.setOptimizer("adam", 0.05);

    double initLoss = nn.evaluate(trainL);
    nn.train(trainL, testL, 100, 8, false);
    double finalLoss = nn.evaluate(trainL);

    TEST_ASSERT(finalLoss < initLoss, "Training must reduce loss over dataset");

    Matrix testPreds = nn.predict(testL.getX());
    double acc = Evaluator::accuracy(testPreds, testL.getY());
    cout << "  Test Accuracy: " << acc * 100.0 << "%" << endl;
    TEST_ASSERT(acc >= 0.5, "Network should learn reasonable accuracy");

    cout << "  -> PASSED" << endl;
    return true;
}

int main() {
    cout << "==========================================================" << endl;
    cout << "           MiniANN Integration Test Suite                 " << endl;
    cout << "==========================================================" << endl;

    int passed = 0;
    int total = 6;

    if (test_matrix_layer_integration()) passed++;
    if (test_activation_layer_network()) passed++;
    if (test_loss_optimizers_training()) passed++;
    if (test_dataloader_integration()) passed++;
    if (test_evaluator_metrics()) passed++;
    if (test_full_end_to_end_pipeline()) passed++;

    cout << "==========================================================" << endl;
    cout << " Results: " << passed << " / " << total << " tests passed." << endl;
    cout << "==========================================================" << endl;

    return (passed == total) ? 0 : 1;
}
