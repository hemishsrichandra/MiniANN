#pragma once

#include "Matrix.h"
#include <string>
#include <vector>
#include <utility>
#include <map>

using namespace std;

class DataLoader {
private:
    Matrix X; // Feature matrix: shape (numFeatures, numSamples)
    Matrix Y; // Target matrix: shape (numTargets, numSamples)
    int numSamples;
    int numFeatures;
    int numTargets;
    vector<string> featureNames;
    string targetName;
    map<string, int> labelMapping; // For categorical labels to integer classes

public:
    DataLoader();
    DataLoader(const Matrix& X, const Matrix& Y);

    // Load dataset from CSV file
    // labelCol: index of target column (-1 means the last column)
    // hasHeader: true if the first line is header with column names
    // delimiter: CSV delimiter character (default is ',')
    bool loadCSV(const string& filename, int labelCol = -1, bool hasHeader = true, char delimiter = ',');

    // Save dataset back to CSV file
    bool saveCSV(const string& filename, char delimiter = ',') const;

    // Data preprocessing
    void shuffle(unsigned int seed = 0);
    pair<DataLoader, DataLoader> split(double trainRatio = 0.8, bool shuffleFirst = true) const;
    void normalize();   // Min-Max feature scaling: (x - min) / (max - min)
    void standardize(); // Z-score standardization: (x - mean) / std
    void oneHotEncode(int numClasses = -1); // Convert 1-row labels to numClasses-row one-hot vectors

    // Batching methods
    int getNumBatches(int batchSize) const;
    pair<Matrix, Matrix> getBatch(int batchIndex, int batchSize) const;
    vector<pair<Matrix, Matrix>> getBatches(int batchSize) const;

    // Getters
    int getNumSamples() const { return numSamples; }
    int getNumFeatures() const { return numFeatures; }
    int getNumTargets() const { return numTargets; }
    const Matrix& getX() const { return X; }
    const Matrix& getY() const { return Y; }
    Matrix& getX() { return X; }
    Matrix& getY() { return Y; }
    const vector<string>& getFeatureNames() const { return featureNames; }
    const string& getTargetName() const { return targetName; }
    const map<string, int>& getLabelMapping() const { return labelMapping; }

    // Display info
    void printSummary() const;
    void printSample(int index) const;
};
