#include "../include/DataLoader.h"
#include <fstream>
#include <sstream>
#include <iostream>
#include <iomanip>
#include <cmath>
#include <algorithm>
#include <random>
#include <chrono>

DataLoader::DataLoader() : numSamples(0), numFeatures(0), numTargets(0) {}

DataLoader::DataLoader(const Matrix& X, const Matrix& Y) 
    : X(X), Y(Y), numSamples(X.getCols()), numFeatures(X.getRows()), numTargets(Y.getRows()) {
    if (X.getCols() != Y.getCols()) {
        throw invalid_argument("DataLoader: X and Y must have the same number of columns (samples)");
    }
}

static string trim(const string& str) {
    size_t first = str.find_first_not_of(" \t\r\n");
    if (first == string::npos) return "";
    size_t last = str.find_last_not_of(" \t\r\n");
    return str.substr(first, (last - first + 1));
}

static bool isNumeric(const string& str) {
    if (str.empty()) return false;
    char* end = nullptr;
    strtod(str.c_str(), &end);
    return end != str.c_str() && (*end == '\0' || *end == '\r');
}

static string toLower(const string& s) {
    string res = s;
    transform(res.begin(), res.end(), res.begin(), ::tolower);
    return res;
}

bool DataLoader::loadCSV(const string& filename, int labelCol, bool hasHeader, char delimiter) {
    string actualFilename = filename;
    ifstream file(actualFilename);
    if (!file.is_open()) {
        // Try appending .csv
        if (actualFilename.find(".csv") == string::npos) {
            actualFilename += ".csv";
            file.open(actualFilename);
        }
    }
    if (!file.is_open()) {
        cerr << "[DataLoader] Error: Could not open file " << filename << endl;
        return false;
    }

    string line;
    vector<vector<string>> rawRows;
    featureNames.clear();
    targetName = "";
    labelMapping.clear();

    vector<string> headerCells;
    if (hasHeader && getline(file, line)) {
        stringstream ss(line);
        string cell;
        while (getline(ss, cell, delimiter)) {
            headerCells.push_back(trim(cell));
        }
    }

    while (getline(file, line)) {
        string trimmedLine = trim(line);
        if (trimmedLine.empty() || trimmedLine[0] == '#') continue;

        stringstream ss(trimmedLine);
        string cell;
        vector<string> row;
        while (getline(ss, cell, delimiter)) {
            row.push_back(trim(cell));
        }
        if (!row.empty()) {
            rawRows.push_back(row);
        }
    }
    file.close();

    if (rawRows.empty()) {
        cerr << "[DataLoader] Error: CSV file contains no data rows" << endl;
        return false;
    }

    const size_t MAX_CSV_ROWS = 25000;
    if (rawRows.size() > MAX_CSV_ROWS) {
        cout << "[DataLoader] Warning: CSV rows (" << rawRows.size() << ") exceed maximum limit of " << MAX_CSV_ROWS 
             << ". Randomly sampling " << MAX_CSV_ROWS << " rows." << endl;
        random_device rd;
        mt19937 g(rd());
        std::shuffle(rawRows.begin(), rawRows.end(), g);
        rawRows.resize(MAX_CSV_ROWS);
    }

    int totalCols = rawRows[0].size();
    int actualLabelCol = (labelCol < 0) ? (totalCols + labelCol) : labelCol;
    if (actualLabelCol < 0 || actualLabelCol >= totalCols) {
        cerr << "[DataLoader] Error: Invalid label column index: " << labelCol << endl;
        return false;
    }

    // Check if column 0 is an ID column (e.g. user_id, id, or values start with USR-)
    bool skipFirstCol = false;
    if (totalCols > 2 && actualLabelCol != 0) {
        if (!headerCells.empty()) {
            string h0 = toLower(headerCells[0]);
            if (h0 == "id" || h0 == "user_id" || h0 == "sample_id" || h0 == "index") {
                skipFirstCol = true;
            }
        }
        if (!skipFirstCol && rawRows.size() > 0) {
            string val0 = rawRows[0][0];
            if (val0.size() >= 3 && (val0.substr(0, 3) == "USR" || val0.substr(0, 3) == "usr" || val0.substr(0, 2) == "ID")) {
                skipFirstCol = true;
            }
        }
    }

    int startCol = skipFirstCol ? 1 : 0;
    numSamples = rawRows.size();
    numFeatures = (totalCols - (skipFirstCol ? 1 : 0)) - 1;
    numTargets = 1;

    if (!headerCells.empty()) {
        for (int c = startCol; c < totalCols; ++c) {
            if (c == actualLabelCol) {
                targetName = headerCells[c];
            } else {
                featureNames.push_back(headerCells[c]);
            }
        }
    }

    // Detect categorical feature columns and map string categories to integer values
    vector<map<string, double>> featureEncoders(totalCols);
    for (int c = startCol; c < totalCols; ++c) {
        if (c == actualLabelCol) continue;
        int nonNumericCount = 0;
        int checkCount = min(numSamples, 100);
        for (int s = 0; s < checkCount; ++s) {
            if (!isNumeric(rawRows[s][c])) nonNumericCount++;
        }
        if (nonNumericCount > 0) {
            double categoryIndex = 0.0;
            for (int s = 0; s < numSamples; ++s) {
                const string& val = rawRows[s][c];
                if (featureEncoders[c].find(val) == featureEncoders[c].end()) {
                    featureEncoders[c][val] = categoryIndex++;
                }
            }
        }
    }

    // Parse target and features
    int classCounter = 0;
    vector<vector<double>> xData(numFeatures, vector<double>(numSamples, 0.0));
    vector<vector<double>> yData(numTargets, vector<double>(numSamples, 0.0));

    for (int s = 0; s < numSamples; ++s) {
        const auto& row = rawRows[s];
        int featIdx = 0;

        for (int c = startCol; c < totalCols; ++c) {
            if (c == actualLabelCol) {
                string labelStr = (c < static_cast<int>(row.size())) ? row[c] : "";
                if (isNumeric(labelStr)) {
                    yData[0][s] = stod(labelStr);
                } else {
                    if (labelMapping.find(labelStr) == labelMapping.end()) {
                        labelMapping[labelStr] = classCounter++;
                    }
                    yData[0][s] = static_cast<double>(labelMapping[labelStr]);
                }
            } else {
                if (featIdx < numFeatures) {
                    string cellVal = (c < static_cast<int>(row.size())) ? row[c] : "0";
                    if (!featureEncoders[c].empty()) {
                        xData[featIdx][s] = featureEncoders[c][cellVal];
                    } else {
                        try {
                            xData[featIdx][s] = stod(cellVal);
                        } catch (...) {
                            xData[featIdx][s] = 0.0;
                        }
                    }
                    featIdx++;
                }
            }
        }
    }

    X = Matrix(xData);
    Y = Matrix(yData);

    return true;
}

bool DataLoader::saveCSV(const string& filename, char delimiter) const {
    ofstream file(filename);
    if (!file.is_open()) return false;

    if (!featureNames.empty()) {
        for (size_t i = 0; i < featureNames.size(); ++i) {
            file << featureNames[i] << delimiter;
        }
        file << (targetName.empty() ? "target" : targetName) << "\n";
    }

    for (int s = 0; s < numSamples; ++s) {
        for (int f = 0; f < numFeatures; ++f) {
            file << X(f, s) << delimiter;
        }
        for (int t = 0; t < numTargets; ++t) {
            file << Y(t, s);
            if (t + 1 < numTargets) file << delimiter;
        }
        file << "\n";
    }
    file.close();
    return true;
}

void DataLoader::shuffle(unsigned int seed) {
    if (numSamples <= 1) return;

    vector<int> indices(numSamples);
    for (int i = 0; i < numSamples; ++i) indices[i] = i;

    if (seed == 0) {
        random_device rd;
        mt19937 g(rd());
        std::shuffle(indices.begin(), indices.end(), g);
    } else {
        mt19937 g(seed);
        std::shuffle(indices.begin(), indices.end(), g);
    }

    vector<vector<double>> newXData(numFeatures, vector<double>(numSamples));
    vector<vector<double>> newYData(numTargets, vector<double>(numSamples));

    for (int j = 0; j < numSamples; ++j) {
        int origIdx = indices[j];
        for (int i = 0; i < numFeatures; ++i) {
            newXData[i][j] = X(i, origIdx);
        }
        for (int i = 0; i < numTargets; ++i) {
            newYData[i][j] = Y(i, origIdx);
        }
    }

    X = Matrix(newXData);
    Y = Matrix(newYData);
}

pair<DataLoader, DataLoader> DataLoader::split(double trainRatio, bool shuffleFirst) const {
    if (trainRatio <= 0.0 || trainRatio >= 1.0) {
        throw invalid_argument("DataLoader::split: trainRatio must be between 0.0 and 1.0");
    }

    DataLoader copyLoader = *this;
    if (shuffleFirst) {
        copyLoader.shuffle();
    }

    int trainCount = static_cast<int>(numSamples * trainRatio);
    if (trainCount < 1) trainCount = 1;
    if (trainCount >= numSamples) trainCount = numSamples - 1;
    int testCount = numSamples - trainCount;

    vector<vector<double>> trainX(numFeatures, vector<double>(trainCount));
    vector<vector<double>> trainY(numTargets, vector<double>(trainCount));
    vector<vector<double>> testX(numFeatures, vector<double>(testCount));
    vector<vector<double>> testY(numTargets, vector<double>(testCount));

    for (int j = 0; j < trainCount; ++j) {
        for (int i = 0; i < numFeatures; ++i) trainX[i][j] = copyLoader.X(i, j);
        for (int i = 0; i < numTargets; ++i) trainY[i][j] = copyLoader.Y(i, j);
    }

    for (int j = 0; j < testCount; ++j) {
        int srcIdx = trainCount + j;
        for (int i = 0; i < numFeatures; ++i) testX[i][j] = copyLoader.X(i, srcIdx);
        for (int i = 0; i < numTargets; ++i) testY[i][j] = copyLoader.Y(i, srcIdx);
    }

    Matrix mTrainX(trainX);
    Matrix mTrainY(trainY);
    Matrix mTestX(testX);
    Matrix mTestY(testY);

    DataLoader trainLoader(mTrainX, mTrainY);
    DataLoader testLoader(mTestX, mTestY);
    trainLoader.featureNames = featureNames;
    trainLoader.targetName = targetName;
    trainLoader.labelMapping = labelMapping;
    testLoader.featureNames = featureNames;
    testLoader.targetName = targetName;
    testLoader.labelMapping = labelMapping;

    return {trainLoader, testLoader};
}

void DataLoader::normalize() {
    if (numSamples == 0 || numFeatures == 0) return;

    for (int i = 0; i < numFeatures; ++i) {
        double minVal = X(i, 0);
        double maxVal = X(i, 0);
        for (int j = 1; j < numSamples; ++j) {
            double v = X(i, j);
            if (v < minVal) minVal = v;
            if (v > maxVal) maxVal = v;
        }
        double range = maxVal - minVal;
        if (range < 1e-12) range = 1.0;

        for (int j = 0; j < numSamples; ++j) {
            X(i, j) = (X(i, j) - minVal) / range;
        }
    }
}

void DataLoader::standardize() {
    if (numSamples == 0 || numFeatures == 0) return;

    for (int i = 0; i < numFeatures; ++i) {
        double sum = 0.0;
        for (int j = 0; j < numSamples; ++j) {
            sum += X(i, j);
        }
        double mean = sum / numSamples;

        double sumSq = 0.0;
        for (int j = 0; j < numSamples; ++j) {
            double diff = X(i, j) - mean;
            sumSq += diff * diff;
        }
        double stdDev = sqrt(sumSq / numSamples);
        if (stdDev < 1e-12) stdDev = 1.0;

        for (int j = 0; j < numSamples; ++j) {
            X(i, j) = (X(i, j) - mean) / stdDev;
        }
    }
}

void DataLoader::oneHotEncode(int numClasses) {
    if (numSamples == 0 || numTargets == 0) return;

    if (numClasses <= 0) {
        double maxLabel = 0.0;
        for (int j = 0; j < numSamples; ++j) {
            if (Y(0, j) > maxLabel) maxLabel = Y(0, j);
        }
        numClasses = static_cast<int>(maxLabel) + 1;
        if (numClasses < 2) numClasses = 2;
    }

    vector<vector<double>> oneHotData(numClasses, vector<double>(numSamples, 0.0));
    for (int j = 0; j < numSamples; ++j) {
        int classIdx = static_cast<int>(Y(0, j));
        if (classIdx >= 0 && classIdx < numClasses) {
            oneHotData[classIdx][j] = 1.0;
        }
    }

    Y = Matrix(oneHotData);
    numTargets = numClasses;
}

int DataLoader::getNumBatches(int batchSize) const {
    if (batchSize <= 0 || numSamples == 0) return 0;
    return (numSamples + batchSize - 1) / batchSize;
}

pair<Matrix, Matrix> DataLoader::getBatch(int batchIndex, int batchSize) const {
    if (batchSize <= 0 || numSamples == 0) {
        throw invalid_argument("DataLoader::getBatch: invalid batchSize");
    }

    int startSample = batchIndex * batchSize;
    if (startSample >= numSamples) {
        throw out_of_range("DataLoader::getBatch: batchIndex out of bounds");
    }

    int endSample = min(startSample + batchSize, numSamples);
    int currentBatchSize = endSample - startSample;

    vector<vector<double>> batchX(numFeatures, vector<double>(currentBatchSize));
    vector<vector<double>> batchY(numTargets, vector<double>(currentBatchSize));

    for (int j = 0; j < currentBatchSize; ++j) {
        int srcIdx = startSample + j;
        for (int i = 0; i < numFeatures; ++i) {
            batchX[i][j] = X(i, srcIdx);
        }
        for (int i = 0; i < numTargets; ++i) {
            batchY[i][j] = Y(i, srcIdx);
        }
    }

    return {Matrix(batchX), Matrix(batchY)};
}

vector<pair<Matrix, Matrix>> DataLoader::getBatches(int batchSize) const {
    int totalBatches = getNumBatches(batchSize);
    vector<pair<Matrix, Matrix>> batches;
    batches.reserve(totalBatches);
    for (int b = 0; b < totalBatches; ++b) {
        batches.push_back(getBatch(b, batchSize));
    }
    return batches;
}

void DataLoader::printSummary() const {
    cout << "================ Dataset Summary ================" << endl;
    cout << " Total Samples : " << numSamples << endl;
    cout << " Features Count: " << numFeatures << endl;
    cout << " Target Shape  : (" << numTargets << ", " << numSamples << ")" << endl;
    if (!featureNames.empty()) {
        cout << " Features (" << featureNames.size() << ")  : ";
        for (size_t i = 0; i < featureNames.size(); ++i) {
            cout << featureNames[i] << (i + 1 < featureNames.size() ? ", " : "");
        }
        cout << endl;
    }
    if (!labelMapping.empty()) {
        cout << " Label Encodings (" << labelMapping.size() << " classes):" << endl;
        for (const auto& pair : labelMapping) {
            cout << "   " << pair.first << " -> " << pair.second << endl;
        }
    }
    cout << "=================================================" << endl;
}

void DataLoader::printSample(int index) const {
    if (index < 0 || index >= numSamples) {
        cout << "[DataLoader] Invalid sample index: " << index << endl;
        return;
    }
    cout << "Sample #" << index << " Features: [";
    for (int i = 0; i < numFeatures; ++i) {
        cout << X(i, index) << (i + 1 < numFeatures ? ", " : "");
    }
    cout << "] Targets: [";
    for (int i = 0; i < numTargets; ++i) {
        cout << Y(i, index) << (i + 1 < numTargets ? ", " : "");
    }
    cout << "]" << endl;
}
