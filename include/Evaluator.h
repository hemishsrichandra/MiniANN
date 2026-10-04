#pragma once

#include "Matrix.h"
#include <vector>
#include <string>

using namespace std;

class Evaluator {
public:
    // Classification Metrics
    // Accuracy: proportion of correctly classified samples
    static double accuracy(const Matrix& predictions, const Matrix& targets);

    // Precision for a specific class (or macro-averaged if positiveClass = -1)
    static double precision(const Matrix& predictions, const Matrix& targets, int positiveClass = 1);

    // Recall for a specific class (or macro-averaged if positiveClass = -1)
    static double recall(const Matrix& predictions, const Matrix& targets, int positiveClass = 1);

    // F1-Score for a specific class (or macro-averaged if positiveClass = -1)
    static double f1Score(const Matrix& predictions, const Matrix& targets, int positiveClass = 1);

    // Confusion Matrix: rows = True Class, cols = Predicted Class
    static Matrix confusionMatrix(const Matrix& predictions, const Matrix& targets, int numClasses = -1);

    // Pretty printing
    static void printConfusionMatrix(const Matrix& cm, const vector<string>& classNames = {});
    static void printClassificationReport(const Matrix& predictions, const Matrix& targets, const vector<string>& classNames = {});

    // Regression Metrics
    static double meanSquaredError(const Matrix& predictions, const Matrix& targets);
    static double meanAbsoluteError(const Matrix& predictions, const Matrix& targets);
    static double r2Score(const Matrix& predictions, const Matrix& targets);

    // Helper functions
    static vector<int> getPredictedClasses(const Matrix& predictions);
    static vector<int> getTrueClasses(const Matrix& targets);
};
