#include "../include/Matrix.h"
#include "../include/Layer.h"
#include "../include/Activation.h"
#include "../include/Loss.h"
#include "../include/WeightInitializer.h"
#include "../include/DataLoader.h"
#include "../include/NeuralNetwork.h"
#include "../include/Evaluator.h"

#include <iostream>
#include <fstream>
#include <string>
#include <sstream>
#include <memory>
#include <iomanip>
#include <vector>
#include <map>
#include <filesystem>
#include <algorithm>
using namespace std;

// --- Helper Functions for User Input ---

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

// --- Filesystem & Path Utilities ---

static bool directoryExists(const string& path) {
    return filesystem::is_directory(path);
}

static bool fileExists(const string& path) {
    return filesystem::is_regular_file(path);
}

static void ensureDirectory(const string& path) {
    filesystem::create_directories(path);
}

static string extractBaseFilename(const string& path) {
    string filename = path;
    size_t lastSlash = filename.find_last_of("/\\");
    if (lastSlash != string::npos) {
        filename = filename.substr(lastSlash + 1);
    }
    size_t lastDot = filename.find_last_of('.');
    if (lastDot != string::npos) {
        filename = filename.substr(0, lastDot);
    }
    if (filename.empty()) filename = "model";
    return filename;
}

static string getUniqueModelDirectory(const string& datasetPath, const string& baseRoot = "saved_models") {
    ensureDirectory(baseRoot);
    string baseName = extractBaseFilename(datasetPath);

    string primary = baseRoot + "/" + baseName;
    if (!directoryExists(primary)) {
        ensureDirectory(primary);
        return primary;
    }

    int counter = 1;
    while (true) {
        string nextCandidate = baseRoot + "/" + baseName + to_string(counter);
        if (!directoryExists(nextCandidate)) {
            ensureDirectory(nextCandidate);
            return nextCandidate;
        }
        counter++;
    }
}

static vector<string> listModelDirectories(const string& baseDir = "saved_models") {
    vector<string> results;
    if (filesystem::is_directory(baseDir)) {
        for (const auto& entry : filesystem::directory_iterator(baseDir)) {
            if (entry.is_directory() && filesystem::is_regular_file(entry.path() / "model.txt")) {
                results.push_back(entry.path().string());
            }
        }
    }
    sort(results.begin(), results.end());
    return results;
}

// --- Metadata Structure & Serialization ---

struct ModelMetadata {
    string datasetPath;
    string targetName;
    int numFeatures = 0;
    int numClasses = 0;
    vector<string> featureNames;
    vector<double> minValues;
    vector<double> maxValues;
    map<int, string> classNames;
    map<string, int> labelMapping;
};

static void saveModelMetadata(const string& filepath, const DataLoader& loader, const string& datasetPath) {
    ofstream out(filepath);
    if (!out.is_open()) return;

    out << "DATASET_PATH " << datasetPath << "\n";
    out << "TARGET_NAME " << (loader.getTargetName().empty() ? "target" : loader.getTargetName()) << "\n";
    out << "NUM_FEATURES " << loader.getNumFeatures() << "\n";

    out << "FEATURE_NAMES";
    const auto& fNames = loader.getFeatureNames();
    for (size_t i = 0; i < fNames.size(); ++i) {
        out << " " << fNames[i];
    }
    out << "\n";

    out << "MIN_VALUES";
    const auto& mins = loader.getMinValues();
    for (size_t i = 0; i < mins.size(); ++i) {
        out << " " << setprecision(10) << mins[i];
    }
    out << "\n";

    out << "MAX_VALUES";
    const auto& maxs = loader.getMaxValues();
    for (size_t i = 0; i < maxs.size(); ++i) {
        out << " " << setprecision(10) << maxs[i];
    }
    out << "\n";

    const auto& labelMap = loader.getLabelMapping();
    out << "NUM_CLASSES " << labelMap.size() << "\n";
    for (const auto& pair : labelMap) {
        out << "LABEL " << pair.second << " " << pair.first << "\n";
    }

    out.close();
}

static bool loadModelMetadata(const string& filepath, ModelMetadata& meta) {
    ifstream in(filepath);
    if (!in.is_open()) return false;

    meta.featureNames.clear();
    meta.minValues.clear();
    meta.maxValues.clear();
    meta.classNames.clear();
    meta.labelMapping.clear();

    string line;
    while (getline(in, line)) {
        stringstream ss(line);
        string key;
        if (!(ss >> key)) continue;

        if (key == "DATASET_PATH") {
            ss >> meta.datasetPath;
        } else if (key == "TARGET_NAME") {
            ss >> meta.targetName;
        } else if (key == "NUM_FEATURES") {
            ss >> meta.numFeatures;
        } else if (key == "FEATURE_NAMES") {
            string name;
            while (ss >> name) meta.featureNames.push_back(name);
        } else if (key == "MIN_VALUES") {
            double v;
            while (ss >> v) meta.minValues.push_back(v);
        } else if (key == "MAX_VALUES") {
            double v;
            while (ss >> v) meta.maxValues.push_back(v);
        } else if (key == "NUM_CLASSES") {
            ss >> meta.numClasses;
        } else if (key == "LABEL") {
            int idx;
            string label;
            ss >> idx >> label;
            meta.classNames[idx] = label;
            meta.labelMapping[label] = idx;
        }
    }
    in.close();
    return true;
}

static void saveMatrixToCSV(const Matrix& m, const string& filepath) {
    ofstream out(filepath);
    if (!out.is_open()) return;
    for (int r = 0; r < m.getRows(); ++r) {
        for (int c = 0; c < m.getCols(); ++c) {
            out << setprecision(10) << m(r, c);
            if (c + 1 < m.getCols()) out << ",";
        }
        out << "\n";
    }
    out.close();
}

static void saveWeightsAndBiasesReport(const NeuralNetwork& nn, const string& filepath) {
    ofstream out(filepath);
    if (!out.is_open()) return;

    out << "==========================================================\n";
    out << "         MiniANN - Trained Weights & Biases Report        \n";
    out << "==========================================================\n\n";

    const auto& layers = nn.getLayers();
    for (size_t i = 0; i < layers.size(); ++i) {
        Matrix w = layers[i].getWeights();
        Matrix b = layers[i].getBiases();

        out << "--- Layer " << (i + 1) << " ---\n";
        out << "Weight Matrix Shape: (" << w.getRows() << " rows x " << w.getCols() << " cols)\n";
        out << "Weights:\n";
        for (int r = 0; r < w.getRows(); ++r) {
            out << "  [";
            for (int c = 0; c < w.getCols(); ++c) {
                out << setw(12) << fixed << setprecision(6) << w(r, c) << (c + 1 < w.getCols() ? ", " : "");
            }
            out << "]\n";
        }

        out << "\nBias Vector Shape: (" << b.getRows() << " rows x " << b.getCols() << " cols)\n";
        out << "Biases:\n  [";
        for (int r = 0; r < b.getRows(); ++r) {
            out << setw(12) << fixed << setprecision(6) << b(r, 0) << (r + 1 < b.getRows() ? ", " : "");
        }
        out << "]\n\n";
    }
    out.close();
}

static void saveTestResultsReport(const string& filepath,
                                  const string& datasetPath,
                                  const string& archStr,
                                  const string& optName,
                                  const string& lossName,
                                  double lr,
                                  int epochs,
                                  int batchSize,
                                  double trainRatio,
                                  int trainSamples,
                                  int testSamples,
                                  double acc,
                                  double prec,
                                  double rec,
                                  double f1,
                                  const Matrix& cm,
                                  const Matrix& testPreds,
                                  const Matrix& testY,
                                  const vector<string>& classNames) {
    ofstream out(filepath);
    if (!out.is_open()) return;

    out << "==========================================================\n";
    out << "           MiniANN - Neural Network Demonstration         \n";
    out << "==========================================================\n\n";
    out << "Dataset Path        : " << datasetPath << "\n";
    out << "Network Architecture: " << archStr << "\n";
    out << "Optimizer           : " << optName << " (Learning Rate = " << lr << ")\n";
    out << "Loss Function       : " << lossName << "\n";
    out << "Epochs              : " << epochs << "\n";
    out << "Batch Size          : " << batchSize << "\n";
    out << "Train / Test Split  : " << (trainRatio * 100.0) << "% (" << trainSamples << " samples) / "
        << ((1.0 - trainRatio) * 100.0) << "% (" << testSamples << " samples)\n\n";

    out << "==================== Evaluation Metrics ====================\n";
    out << fixed << setprecision(4);
    out << " Test Accuracy : " << acc * 100.0 << "%\n";
    out << " Test Precision: " << prec << "\n";
    out << " Test Recall   : " << rec << "\n";
    out << " Test F1-Score : " << f1 << "\n\n";

    out << "==================== Confusion Matrix ====================\n";
    int numClasses = cm.getRows();
    out << setw(12) << "True \\ Pred";
    for (int j = 0; j < numClasses; ++j) {
        string name = (j < static_cast<int>(classNames.size())) ? classNames[j] : ("Class " + to_string(j));
        if (name.length() > 8) name = name.substr(0, 8);
        out << setw(10) << name;
    }
    out << "\n" << string(12 + numClasses * 10, '-') << "\n";

    for (int i = 0; i < numClasses; ++i) {
        string name = (i < static_cast<int>(classNames.size())) ? classNames[i] : ("Class " + to_string(i));
        if (name.length() > 10) name = name.substr(0, 10);
        out << setw(12) << name;
        for (int j = 0; j < numClasses; ++j) {
            out << setw(10) << static_cast<int>(cm(i, j));
        }
        out << "\n";
    }

    out << "\n================ Classification Report ================\n";
    out << left << setw(16) << "Class"
        << setw(12) << "Precision"
        << setw(12) << "Recall"
        << setw(12) << "F1-Score"
        << setw(10) << "Support" << endl;
    out << string(62, '-') << endl;

    vector<int> trueClasses = Evaluator::getTrueClasses(testY);
    vector<int> predClasses = Evaluator::getPredictedClasses(testPreds);

    for (int c = 0; c < numClasses; ++c) {
        double p = Evaluator::precision(testPreds, testY, c);
        double r = Evaluator::recall(testPreds, testY, c);
        double f = Evaluator::f1Score(testPreds, testY, c);
        int support = 0;
        for (int tc : trueClasses) if (tc == c) support++;

        string cName = (c < static_cast<int>(classNames.size())) ? classNames[c] : ("Class " + to_string(c));
        out << left << setw(16) << cName
            << setw(12) << fixed << setprecision(4) << p
            << setw(12) << r
            << setw(12) << f
            << setw(10) << support << endl;
    }
    out << string(62, '-') << endl;
    out << "\n==========================================================\n";
    out << "              Demonstration Completed Successfully!       \n";
    out << "==========================================================\n";

    out.close();
}

// --- Mode 1: Training Pipeline ---

static void runTrainingPipeline(int argc, char* argv[]) {
    string datasetPath = "data/bedtime_screentime_sleep_debt_converted.csv";
    int epochs = 30;
    double learningRate = 0.01;
    int batchSize = 64;
    double trainRatio = 0.8;
    vector<int> hiddenNeurons;
    int actChoice = 1; // 1: ReLU, 2: Tanh, 3: Sigmoid
    int optChoice = 1; // 1: Adam, 2: SGD, 3: Momentum
    int lossChoice = 1; // 1: MSE, 2: Binary Cross-Entropy, 3: Categorical Cross-Entropy
    string optName = "adam";
    string lossName = "MSE";

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

        cout << "\n5. Select Loss Function:\n"
             << "   1. Mean Squared Error (MSE) (Recommended)\n"
             << "   2. Binary Cross-Entropy\n"
             << "   3. Categorical Cross-Entropy\n";
        lossChoice = promptInt("   Select (1-3)", 1);
        if (lossChoice == 2) lossName = "BinaryCrossEntropy";
        else if (lossChoice == 3) lossName = "CategoricalCrossEntropy";
        else lossName = "MSE";

        learningRate = promptDouble("\n6. Enter Learning Rate", 0.01);
        epochs = promptInt("7. Enter Number of Epochs", 30);
        batchSize = promptInt("8. Enter Batch Size", 64);
        trainRatio = promptDouble("9. Enter Train/Test Split Ratio (0.5 to 0.95)", 0.8);

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
            return;
        }
        datasetPath = "data/sample_dataset.csv";
    }

    dataLoader.printSummary();

    // Step 2: Preprocess Data
    cout << "\n[Step 2] Preprocessing Data (Splitting, then applying training-set normalization)..." << endl;
    dataLoader.oneHotEncode();

    auto splitData = dataLoader.split(trainRatio, true);
    DataLoader trainLoader = splitData.first;
    DataLoader testLoader = splitData.second;

    trainLoader.normalize();
    testLoader.applyNormalization(trainLoader.getMinValues(), trainLoader.getMaxValues());

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
    stringstream archSS;
    archSS << inputSize << " inputs";
    for (size_t i = 0; i < hiddenNeurons.size(); ++i) {
        archSS << " -> " << hiddenNeurons[i] << " hidden (" << actName << ")";
    }
    archSS << " -> " << outputSize << " outputs (Sigmoid)";
    string archStr = archSS.str();
    cout << " Architecture: " << archStr << endl;

    NeuralNetwork nn;
    int prevDim = inputSize;
    for (size_t i = 0; i < hiddenNeurons.size(); ++i) {
        nn.addLayer(prevDim, hiddenNeurons[i], hiddenActivation, make_shared<He>());
        prevDim = hiddenNeurons[i];
    }
    nn.addLayer(prevDim, outputSize, make_shared<Sigmoid>(), make_shared<Xavier>());

    // Set User Selected Loss Function
    if (lossChoice == 2) {
        nn.setLoss(make_shared<BinaryCrossEntropy>());
    } else if (lossChoice == 3) {
        nn.setLoss(make_shared<CategoricalCrossEntropy>());
    } else {
        nn.setLoss(make_shared<MSE>());
    }

    nn.setOptimizer(optName, learningRate);
    nn.summary();

    // Step 4: Training
    cout << "[Step 4] Training Neural Network..." << endl;
    cout << " Hyperparameters: Epochs=" << epochs
         << ", LearningRate=" << learningRate
         << ", BatchSize=" << batchSize
         << ", Optimizer=" << optName
         << ", Loss=" << lossName << endl;

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

    vector<string> classNames(dataLoader.getLabelMapping().size());
    for (const auto& pair : dataLoader.getLabelMapping()) {
        if (pair.second >= 0 && pair.second < static_cast<int>(classNames.size())) {
            classNames[pair.second] = pair.first;
        }
    }
    Evaluator::printClassificationReport(testPreds, testLoader.getY(), classNames);

    // Step 7: Model Saving & Artifact Generation
    cout << "\n[Step 7] Saving Model, Matrices & Evaluation Results..." << endl;
    string modelDir = getUniqueModelDirectory(datasetPath, "saved_models");
    cout << " Target Directory: " << modelDir << "/" << endl;

    // 1. Save core model checkpoint
    string modelFile = modelDir + "/model.txt";
    if (nn.saveModel(modelFile)) {
        cout << " [Saved] Model Architecture & Weights -> " << modelFile << endl;
    }

    // 2. Save metadata (feature names, min/max values, label encodings)
    string metaFile = modelDir + "/metadata.txt";
    saveModelMetadata(metaFile, trainLoader, datasetPath);
    cout << " [Saved] Preprocessing Metadata       -> " << metaFile << endl;

    // 3. Save individual matrix CSVs for weights and biases
    const auto& layers = nn.getLayers();
    for (size_t i = 0; i < layers.size(); ++i) {
        string wFile = modelDir + "/layer_" + to_string(i + 1) + "_weights.csv";
        string bFile = modelDir + "/layer_" + to_string(i + 1) + "_biases.csv";
        saveMatrixToCSV(layers[i].getWeights(), wFile);
        saveMatrixToCSV(layers[i].getBiases(), bFile);
    }
    cout << " [Saved] Layer Weight & Bias CSVs     -> " << modelDir << "/layer_*_{weights,biases}.csv" << endl;

    // 4. Save human-readable weights and biases report
    string wbReport = modelDir + "/weights_and_biases.txt";
    saveWeightsAndBiasesReport(nn, wbReport);
    cout << " [Saved] Human-readable Weights Report -> " << wbReport << endl;

    // 5. Save comprehensive test results report
    string testReport = modelDir + "/test_results.txt";
    saveTestResultsReport(testReport, datasetPath, archStr, optName, lossName,
                          learningRate, epochs, batchSize, trainRatio,
                          trainLoader.getNumSamples(), testLoader.getNumSamples(),
                          acc, prec, rec, f1, cm, testPreds, testLoader.getY(), classNames);
    cout << " [Saved] Complete Evaluation Report   -> " << testReport << endl;

    cout << "\n==========================================================" << endl;
    cout << "              Demonstration Completed Successfully!       " << endl;
    cout << "==========================================================" << endl;
}

// --- Mode 2: Prediction Using Pre-trained Model ---

static void runPredictionPipeline() {
    cout << "\n--- Predict Output Using Pre-trained Model ---\n" << endl;

    // Discover available saved model directories
    vector<string> modelDirs = listModelDirectories("saved_models");
    string defaultModelDir = "saved_models/bedtime_screentime_sleep_debt_converted";
    if (!modelDirs.empty()) {
        cout << "Discovered Pre-trained Models:" << endl;
        for (size_t i = 0; i < modelDirs.size(); ++i) {
            cout << "   " << (i + 1) << ". " << modelDirs[i] << endl;
        }
        defaultModelDir = modelDirs[0];
    } else {
        cout << "No existing models found in saved_models/. Using default search path." << endl;
    }

    string selectedDir = promptString("\nEnter Model Folder or Path", defaultModelDir);

    // Determine model and metadata file paths
    string modelFile = selectedDir;
    string metaFile = selectedDir;
    if (directoryExists(selectedDir)) {
        modelFile = selectedDir + "/model.txt";
        metaFile = selectedDir + "/metadata.txt";
    } else if (fileExists(selectedDir)) {
        size_t lastSlash = selectedDir.find_last_of("/\\");
        string parent = (lastSlash != string::npos) ? selectedDir.substr(0, lastSlash) : ".";
        metaFile = parent + "/metadata.txt";
    }

    // Load Neural Network Model
    NeuralNetwork loadedNN;
    if (!loadedNN.loadModel(modelFile)) {
        cerr << "Error: Could not load model weights from " << modelFile << endl;
        return;
    }
    cout << "\n[Success] Pre-trained Neural Network Loaded Successfully from " << modelFile << "!" << endl;

    // Load Preprocessing Metadata
    ModelMetadata meta;
    bool hasMeta = loadModelMetadata(metaFile, meta);
    if (hasMeta) {
        cout << "[Success] Loaded Preprocessing Metadata from " << metaFile << endl;
    } else {
        cout << "[Warning] Metadata file not found at " << metaFile << ". Proceeding with raw inputs." << endl;
    }

    // Display Loaded Model Summary
    loadedNN.summary();

    cout << "Select Prediction Input Mode:\n"
         << "   1. Predict on a CSV Dataset / Test File\n"
         << "   2. Enter Feature Values Interactively (Single Sample)\n";
    int predMode = promptInt("Select (1-2)", 1);

    if (predMode == 1) {
        // Batch Prediction on CSV
        string defaultCSV = meta.datasetPath.empty() ? "data/bedtime_screentime_sleep_debt_converted.csv" : meta.datasetPath;
        string csvPath = promptString("\nEnter CSV Path for Prediction", defaultCSV);

        DataLoader predLoader;
        if (!predLoader.loadCSV(csvPath, -1, true, ',')) {
            cerr << "Error: Could not load CSV from " << csvPath << endl;
            return;
        }

        cout << "\nLoaded " << predLoader.getNumSamples() << " samples with "
             << predLoader.getNumFeatures() << " features from " << csvPath << endl;

        // Apply saved normalization scaling
        if (hasMeta && !meta.minValues.empty() && !meta.maxValues.empty()) {
            cout << "Applying loaded Min-Max normalization scaling to features..." << endl;
            predLoader.applyNormalization(meta.minValues, meta.maxValues);
        } else {
            predLoader.normalize();
        }

        Matrix preds = loadedNN.predict(predLoader.getX());
        vector<int> predClassIndices = Evaluator::getPredictedClasses(preds);

        cout << "\n================ Sample Prediction Outputs ================" << endl;
        int displayCount = min(15, predLoader.getNumSamples());
        cout << left << setw(8) << "Sample"
             << setw(16) << "Predicted Class"
             << setw(20) << "Class Name"
             << "Confidence" << endl;
        cout << string(56, '-') << endl;

        for (int s = 0; s < displayCount; ++s) {
            int cls = predClassIndices[s];
            string clsName = (meta.classNames.find(cls) != meta.classNames.end()) ? meta.classNames[cls] : "Unknown";
            double conf = preds(cls, s);

            cout << left << setw(8) << (s + 1)
                 << setw(16) << cls
                 << setw(20) << clsName
                 << fixed << setprecision(2) << (conf * 100.0) << "%" << endl;
        }
        cout << string(56, '-') << endl;

        // If target labels exist, compute evaluation metrics
        if (predLoader.getY().getRows() > 0 && predLoader.getY().getCols() == predLoader.getNumSamples()) {
            Matrix trueY = predLoader.getY();
            if (trueY.getRows() == 1 && preds.getRows() > 1) {
                predLoader.oneHotEncode(preds.getRows());
                trueY = predLoader.getY();
            }

            if (trueY.getRows() == preds.getRows()) {
                cout << "\n================ Evaluation Against Ground Truth ================" << endl;
                double acc = Evaluator::accuracy(preds, trueY);
                double prec = Evaluator::precision(preds, trueY, -1);
                double rec = Evaluator::recall(preds, trueY, -1);
                double f1 = Evaluator::f1Score(preds, trueY, -1);

                cout << fixed << setprecision(4);
                cout << " Accuracy : " << acc * 100.0 << "%" << endl;
                cout << " Precision: " << prec << endl;
                cout << " Recall   : " << rec << endl;
                cout << " F1-Score : " << f1 << endl;

                Matrix cm = Evaluator::confusionMatrix(preds, trueY);
                Evaluator::printConfusionMatrix(cm);

                vector<string> cNames;
                for (int i = 0; i < preds.getRows(); ++i) {
                    if (meta.classNames.find(i) != meta.classNames.end()) {
                        cNames.push_back(meta.classNames[i]);
                    } else {
                        cNames.push_back("Class " + to_string(i));
                    }
                }
                Evaluator::printClassificationReport(preds, trueY, cNames);
            }
        }

    } else {
        // Interactive Single Sample Prediction
        int numFeats = (hasMeta && meta.numFeatures > 0) ? meta.numFeatures :
                       (!loadedNN.getLayers().empty() ? loadedNN.getLayers()[0].getWeights().getCols() : 1);

        cout << "\nEnter Feature Values for Prediction (" << numFeats << " features expected):" << endl;
        vector<vector<double>> sampleData(numFeats, vector<double>(1, 0.0));

        for (int i = 0; i < numFeats; ++i) {
            string featLabel = (i < static_cast<int>(meta.featureNames.size())) ? meta.featureNames[i] : ("Feature " + to_string(i + 1));
            double defVal = 0.5;
            if (hasMeta && i < static_cast<int>(meta.minValues.size()) && i < static_cast<int>(meta.maxValues.size())) {
                defVal = (meta.minValues[i] + meta.maxValues[i]) / 2.0;
            }
            double rawVal = promptDouble("   - " + featLabel, defVal);

            // Apply min-max normalization
            if (hasMeta && i < static_cast<int>(meta.minValues.size()) && i < static_cast<int>(meta.maxValues.size())) {
                double range = meta.maxValues[i] - meta.minValues[i];
                if (range < 1e-12) range = 1.0;
                sampleData[i][0] = (rawVal - meta.minValues[i]) / range;
            } else {
                sampleData[i][0] = rawVal;
            }
        }

        Matrix inputSample(sampleData);
        Matrix outputProb = loadedNN.predict(inputSample);

        cout << "\n================ Single Sample Prediction Output ================" << endl;
        int predClass = 0;
        double maxProb = -1.0;

        for (int r = 0; r < outputProb.getRows(); ++r) {
            double prob = outputProb(r, 0);
            if (prob > maxProb) {
                maxProb = prob;
                predClass = r;
            }
            string cName = (meta.classNames.find(r) != meta.classNames.end()) ? meta.classNames[r] : ("Class " + to_string(r));
            cout << " Class [" << r << "] (" << left << setw(14) << cName << "): "
                 << fixed << setprecision(4) << prob
                 << " (" << setprecision(2) << (prob * 100.0) << "%)" << endl;
        }

        string bestName = (meta.classNames.find(predClass) != meta.classNames.end()) ? meta.classNames[predClass] : to_string(predClass);
        cout << "\n>>> PREDICTED TARGET CLASS: [" << predClass << "] " << bestName
             << " (Confidence: " << fixed << setprecision(2) << (maxProb * 100.0) << "%) <<<\n" << endl;
    }

    cout << "==========================================================" << endl;
    cout << "              Prediction Completed Successfully!          " << endl;
    cout << "==========================================================" << endl;
}

// --- Main Entry Point ---

int main(int argc, char* argv[]) {
    cout << "==========================================================" << endl;
    cout << "           MiniANN - Neural Network Demonstration         " << endl;
    cout << "==========================================================" << endl;

    if (argc > 1) {
        // CLI arguments provided -> run training directly
        runTrainingPipeline(argc, argv);
        return 0;
    }

    // Interactive Top-level Menu
    cout << "\nSelect Mode:\n"
         << "   1. Train on Dataset\n"
         << "   2. Predict Output Using Pre-trained Model\n";
    int mode = promptInt("Select (1-2)", 1);

    if (mode == 2) {
        runPredictionPipeline();
    } else {
        runTrainingPipeline(argc, argv);
    }

    return 0;
}
