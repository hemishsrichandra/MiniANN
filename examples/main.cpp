#include "../include/Matrix.h"         
#include "../include/Layer.h"
#include "../include/Activation.h"
#include "../include/Loss.h"
#include "../include/Optimizer.h"
#include "../include/WeightInitializer.h"
#include "../include/DataLoader.h"
#include "../include/NeuralNetwork.h"
#include "../include/Evaluator.h"

#include <iostream>
#include <string>
#include <sstream>
#include <memory>
#include <iomanip>
#include <vector>

using namespace std;

static string promptString(const string& prompt, const string& defaultValue) {
    cout << prompt << " [" << defaultValue << "]: ";
    string input;
    getline(cin, input);
    if (input.empty()) return defaultValue;
    return input;
}

static int promptInt(const string& prompt, int defaultValue) {
    cout << prompt << " [" << defaultValue << "]: ";
    string input;
    getline(cin, input);
    if (input.empty()) return defaultValue;
    try {
        return stoi(input);
    } catch (...) {
        return defaultValue;
    }
}

static double promptDouble(const string& prompt, double defaultValue) {
    cout << prompt << " [" << defaultValue << "]: ";
    string input;
    getline(cin, input);
    if (input.empty()) return defaultValue;
    try {
        return stod(input);
    } catch (...) {
        return defaultValue;
    }
}

int main(int argc, char* argv[]) {
    cout << "==========================================================" << endl;
    cout << "           MiniANN - Neural Network Demonstration         " << endl;
    cout << "==========================================================" << endl;

    string datasetPath = "data/bedtime_screentime_sleep_debt_converted.csv";
    int epochs = 30;
    double learningRate = 0.01;
    int batchSize = 64;
    double trainRatio = 0.8;
    vector<int> hiddenNeurons;
    int actChoice = 1; // 1: ReLU, 2: Tanh, 3: Sigmoid
    int optChoice = 1; // 1: Adam, 2: SGD, 3: Momentum
    string optName = "adam";

    bool isInteractive = (argc <= 1);

    if (isInteractive) {
        cout << "\n--- Interactive Neural Network Setup ---\n" << endl;

        datasetPath = promptString("1. Enter Dataset CSV Path", "data/bedtime_screentime_sleep_debt_converted.csv");
        
        int numHiddenLayers = promptInt("2. Enter Number of Hidden Layers", 2);
        if (numHiddenLayers < 1) numHiddenLayers = 1;

        hiddenNeurons.resize(numHiddenLayers);
        for (int i = 0; i < numHiddenLayers; ++i) {
            int defaultNeurons = (i == 0) ? 64 : ((i == 1) ? 32 : 16);
            string prompt = "   - Enter Neurons for Hidden Layer " + to_string(i + 1);
            hiddenNeurons[i] = promptInt(prompt, defaultNeurons);
            if (hiddenNeurons[i] < 1) hiddenNeurons[i] = 1;
        }

        cout << "\n3. Select Activation for Hidden Layers:\n"
             << "   1. ReLU (Recommended)\n"
             << "   2. Tanh\n"
             << "   3. Sigmoid\n";
        actChoice = promptInt("   Select (1-3)", 1);

        cout << "\n4. Select Optimizer:\n"
             << "   1. Adam (Recommended)\n"
             << "   2. SGD\n"
             << "   3. Momentum\n";
        optChoice = promptInt("   Select (1-3)", 1);
        if (optChoice == 2) optName = "sgd";
        else if (optChoice == 3) optName = "momentum";
        else optName = "adam";

        learningRate = promptDouble("\n5. Enter Learning Rate", 0.01);
        epochs = promptInt("6. Enter Number of Epochs", 30);
        batchSize = promptInt("7. Enter Batch Size", 64);
        trainRatio = promptDouble("8. Enter Train/Test Split Ratio (0.5 to 0.95)", 0.8);

    } else {
        // CLI argument mode
        if (argc >= 2) datasetPath = argv[1];
        if (argc >= 3) epochs = stoi(argv[2]);
        if (argc >= 4) learningRate = stod(argv[3]);
        if (argc >= 5) batchSize = stoi(argv[4]);
        
        string hiddenArch = "64,32";
        if (argc >= 6) hiddenArch = argv[5];

        stringstream ss(hiddenArch);
        string token;
        while (getline(ss, token, ',')) {
            if (!token.empty()) hiddenNeurons.push_back(stoi(token));
        }
        if (hiddenNeurons.empty()) hiddenNeurons.push_back(16);
    }

    // Step 1: Load Dataset
    cout << "\n[Step 1] Loading Dataset from CSV: " << datasetPath << endl;
    DataLoader dataLoader;
    if (!dataLoader.loadCSV(datasetPath, -1, true, ',')) {
        cerr << "Failed to load CSV from " << datasetPath << ". Trying fallback path..." << endl;
        if (!dataLoader.loadCSV("data/sample_dataset.csv", -1, true, ',')) {
            cerr << "Could not open dataset file. Exiting." << endl;
            return 1;
        }
    }

    dataLoader.printSummary();

    // Step 2: Preprocess Data
    cout << "\n[Step 2] Preprocessing Data (Normalizing features & One-hot encoding targets)..." << endl;
    dataLoader.normalize();
    dataLoader.oneHotEncode();

    auto splitData = dataLoader.split(trainRatio, true);
    DataLoader trainLoader = splitData.first;
    DataLoader testLoader = splitData.second;

    cout << " Train Set: " << trainLoader.getNumSamples() << " samples (" << fixed << setprecision(1) << (trainRatio * 100.0) << "%)" << endl;
    cout << " Test Set : " << testLoader.getNumSamples() << " samples (" << ((1.0 - trainRatio) * 100.0) << "%)" << endl;

    int inputSize = trainLoader.getNumFeatures();
    int outputSize = trainLoader.getNumTargets();

    // Select activation instance
    shared_ptr<IActivation> hiddenActivation;
    string actName;
    if (actChoice == 2) {
        hiddenActivation = make_shared<Tanh>();
        actName = "Tanh";
    } else if (actChoice == 3) {
        hiddenActivation = make_shared<Sigmoid>();
        actName = "Sigmoid";
    } else {
        hiddenActivation = make_shared<ReLU>();
        actName = "ReLU";
    }

    // Step 3: Build Neural Network
    cout << "\n[Step 3] Constructing Neural Network..." << endl;
    cout << " Architecture: " << inputSize << " inputs";
    for (size_t i = 0; i < hiddenNeurons.size(); ++i) {
        cout << " -> " << hiddenNeurons[i] << " hidden (" << actName << ")";
    }
    cout << " -> " << outputSize << " outputs (Sigmoid)" << endl;

    NeuralNetwork nn;
    int prevDim = inputSize;
    for (size_t i = 0; i < hiddenNeurons.size(); ++i) {
        nn.addLayer(prevDim, hiddenNeurons[i], hiddenActivation, make_shared<He>());
        prevDim = hiddenNeurons[i];
    }
    nn.addLayer(prevDim, outputSize, make_shared<Sigmoid>(), make_shared<Xavier>());

    nn.setLoss(make_shared<MSE>());
    nn.setOptimizer(optName, learningRate);

    nn.summary();

    // Step 4: Training
    cout << "[Step 4] Training Neural Network..." << endl;
    cout << " Hyperparameters: Epochs=" << epochs 
         << ", LearningRate=" << learningRate 
         << ", BatchSize=" << batchSize 
         << ", Optimizer=" << optName << endl;

    nn.train(trainLoader, testLoader, epochs, batchSize, true);

    // Step 5: Prediction
    cout << "\n[Step 5] Predicting on Test Set..." << endl;
    Matrix testPreds = nn.predict(testLoader.getX());

    // Step 6: Evaluation
    cout << "\n[Step 6] Evaluating Performance with Evaluator..." << endl;
    double acc = Evaluator::accuracy(testPreds, testLoader.getY());
    double prec = Evaluator::precision(testPreds, testLoader.getY(), -1);
    double rec = Evaluator::recall(testPreds, testLoader.getY(), -1);
    double f1 = Evaluator::f1Score(testPreds, testLoader.getY(), -1);

    cout << fixed << setprecision(4);
    cout << " Test Accuracy : " << acc * 100.0 << "%" << endl;
    cout << " Test Precision: " << prec << endl;
    cout << " Test Recall   : " << rec << endl;
    cout << " Test F1-Score : " << f1 << endl;

    Matrix cm = Evaluator::confusionMatrix(testPreds, testLoader.getY());
    Evaluator::printConfusionMatrix(cm);
    Evaluator::printClassificationReport(testPreds, testLoader.getY());

    cout << "==========================================================" << endl;
    cout << "              Demonstration Completed Successfully!       " << endl;
    cout << "==========================================================" << endl;

    return 0;
}
