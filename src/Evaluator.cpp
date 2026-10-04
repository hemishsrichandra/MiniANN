#include "Evaluator.h"
#include <iostream>
#include <iomanip>
#include <cmath>
#include <algorithm>
#include <set>

vector<int> Evaluator::getPredictedClasses(const Matrix& predictions) {
    int numOutputs = predictions.getRows();
    int numSamples = predictions.getCols();
    vector<int> predClasses(numSamples, 0);

    if (numOutputs == 1) {
        // Binary classification output (e.g. from Sigmoid)
        for (int j = 0; j < numSamples; ++j) {
            predClasses[j] = (predictions(0, j) >= 0.5) ? 1 : 0;
        }
    } else {
        // Multiclass classification (argmax along row dimension for each sample)
        for (int j = 0; j < numSamples; ++j) {
            int bestIdx = 0;
            double bestVal = predictions(0, j);
            for (int i = 1; i < numOutputs; ++i) {
                if (predictions(i, j) > bestVal) {
                    bestVal = predictions(i, j);
                    bestIdx = i;
                }
            }
            predClasses[j] = bestIdx;
        }
    }
    return predClasses;
}

vector<int> Evaluator::getTrueClasses(const Matrix& targets) {
    int numOutputs = targets.getRows();
    int numSamples = targets.getCols();
    vector<int> trueClasses(numSamples, 0);

    if (numOutputs == 1) {
        for (int j = 0; j < numSamples; ++j) {
            trueClasses[j] = static_cast<int>(round(targets(0, j)));
        }
    } else {
        // One-hot encoded target
        for (int j = 0; j < numSamples; ++j) {
            int bestIdx = 0;
            double bestVal = targets(0, j);
            for (int i = 1; i < numOutputs; ++i) {
                if (targets(i, j) > bestVal) {
                    bestVal = targets(i, j);
                    bestIdx = i;
                }
            }
            trueClasses[j] = bestIdx;
        }
    }
    return trueClasses;
}

double Evaluator::accuracy(const Matrix& predictions, const Matrix& targets) {
    int numSamples = predictions.getCols();
    if (numSamples == 0 || targets.getCols() != numSamples) return 0.0;

    vector<int> preds = getPredictedClasses(predictions);
    vector<int> trues = getTrueClasses(targets);

    int correct = 0;
    for (int j = 0; j < numSamples; ++j) {
        if (preds[j] == trues[j]) {
            correct++;
        }
    }
    return static_cast<double>(correct) / numSamples;
}

Matrix Evaluator::confusionMatrix(const Matrix& predictions, const Matrix& targets, int numClasses) {
    vector<int> preds = getPredictedClasses(predictions);
    vector<int> trues = getTrueClasses(targets);
    int numSamples = predictions.getCols();

    if (numClasses <= 0) {
        int maxClass = 0;
        for (int c : preds) if (c > maxClass) maxClass = c;
        for (int c : trues) if (c > maxClass) maxClass = c;
        numClasses = maxClass + 1;
        if (numClasses < 2) numClasses = 2;
    }

    Matrix cm(numClasses, numClasses, 0.0);
    for (int j = 0; j < numSamples; ++j) {
        int t = trues[j];
        int p = preds[j];
        if (t >= 0 && t < numClasses && p >= 0 && p < numClasses) {
            cm(t, p) = cm(t, p) + 1.0;
        }
    }
    return cm;
}

double Evaluator::precision(const Matrix& predictions, const Matrix& targets, int positiveClass) {
    Matrix cm = confusionMatrix(predictions, targets);
    int numClasses = cm.getRows();

    if (positiveClass >= 0) {
        if (positiveClass >= numClasses) return 0.0;
        double tp = cm(positiveClass, positiveClass);
        double fp = 0.0;
        for (int r = 0; r < numClasses; ++r) {
            if (r != positiveClass) fp += cm(r, positiveClass);
        }
        return (tp + fp > 0) ? (tp / (tp + fp)) : 0.0;
    } else {
        // Macro-averaged precision across all classes
        double totalPrecision = 0.0;
        int validClasses = 0;
        for (int c = 0; c < numClasses; ++c) {
            double tp = cm(c, c);
            double fp = 0.0;
            for (int r = 0; r < numClasses; ++r) {
                if (r != c) fp += cm(r, c);
            }
            if (tp + fp > 0) {
                totalPrecision += tp / (tp + fp);
                validClasses++;
            }
        }
        return (validClasses > 0) ? (totalPrecision / validClasses) : 0.0;
    }
}

double Evaluator::recall(const Matrix& predictions, const Matrix& targets, int positiveClass) {
    Matrix cm = confusionMatrix(predictions, targets);
    int numClasses = cm.getRows();

    if (positiveClass >= 0) {
        if (positiveClass >= numClasses) return 0.0;
        double tp = cm(positiveClass, positiveClass);
        double fn = 0.0;
        for (int c = 0; c < numClasses; ++c) {
            if (c != positiveClass) fn += cm(positiveClass, c);
        }
        return (tp + fn > 0) ? (tp / (tp + fn)) : 0.0;
    } else {
        // Macro-averaged recall across all classes
        double totalRecall = 0.0;
        int validClasses = 0;
        for (int c = 0; c < numClasses; ++c) {
            double tp = cm(c, c);
            double fn = 0.0;
            for (int col = 0; col < numClasses; ++col) {
                if (col != c) fn += cm(c, col);
            }
            if (tp + fn > 0) {
                totalRecall += tp / (tp + fn);
                validClasses++;
            }
        }
        return (validClasses > 0) ? (totalRecall / validClasses) : 0.0;
    }
}

double Evaluator::f1Score(const Matrix& predictions, const Matrix& targets, int positiveClass) {
    if (positiveClass >= 0) {
        double p = precision(predictions, targets, positiveClass);
        double r = recall(predictions, targets, positiveClass);
        return (p + r > 0) ? (2.0 * p * r / (p + r)) : 0.0;
    } else {
        Matrix cm = confusionMatrix(predictions, targets);
        int numClasses = cm.getRows();
        double totalF1 = 0.0;
        int validClasses = 0;
        for (int c = 0; c < numClasses; ++c) {
            double p = precision(predictions, targets, c);
            double r = recall(predictions, targets, c);
            if (p + r > 0) {
                totalF1 += 2.0 * p * r / (p + r);
                validClasses++;
            }
        }
        return (validClasses > 0) ? (totalF1 / validClasses) : 0.0;
    }
}

void Evaluator::printConfusionMatrix(const Matrix& cm, const vector<string>& classNames) {
    int numClasses = cm.getRows();
    cout << "\n================= Confusion Matrix =================" << endl;
    cout << setw(12) << "True \\ Pred";
    for (int j = 0; j < numClasses; ++j) {
        string name = (j < static_cast<int>(classNames.size())) ? classNames[j] : ("Class " + to_string(j));
        cout << setw(12) << name;
    }
    cout << endl;

    for (int i = 0; i < numClasses; ++i) {
        string name = (i < static_cast<int>(classNames.size())) ? classNames[i] : ("Class " + to_string(i));
        cout << setw(12) << name;
        for (int j = 0; j < numClasses; ++j) {
            cout << setw(12) << static_cast<int>(cm(i, j));
        }
        cout << endl;
    }
    cout << "====================================================" << endl;
}

void Evaluator::printClassificationReport(const Matrix& predictions, const Matrix& targets, const vector<string>& classNames) {
    Matrix cm = confusionMatrix(predictions, targets);
    int numClasses = cm.getRows();
    int totalSamples = predictions.getCols();

    cout << "\n================ Classification Report ================" << endl;
    cout << left << setw(16) << "Class"
         << right << setw(12) << "Precision"
         << setw(12) << "Recall"
         << setw(12) << "F1-Score"
         << setw(12) << "Support" << endl;
    cout << string(64, '-') << endl;

    double macroP = 0.0, macroR = 0.0, macroF1 = 0.0;
    double weightedP = 0.0, weightedR = 0.0, weightedF1 = 0.0;

    for (int c = 0; c < numClasses; ++c) {
        string name = (c < static_cast<int>(classNames.size())) ? classNames[c] : ("Class " + to_string(c));
        
        // Support is sum of row c in confusion matrix
        int support = 0;
        for (int j = 0; j < numClasses; ++j) support += static_cast<int>(cm(c, j));

        double p = precision(predictions, targets, c);
        double r = recall(predictions, targets, c);
        double f1 = (p + r > 0) ? (2.0 * p * r / (p + r)) : 0.0;

        macroP += p;
        macroR += r;
        macroF1 += f1;

        weightedP += p * support;
        weightedR += r * support;
        weightedF1 += f1 * support;

        cout << left << setw(16) << name
             << right << fixed << setprecision(4)
             << setw(12) << p
             << setw(12) << r
             << setw(12) << f1
             << setw(12) << support << endl;
    }

    macroP /= numClasses;
    macroR /= numClasses;
    macroF1 /= numClasses;

    if (totalSamples > 0) {
        weightedP /= totalSamples;
        weightedR /= totalSamples;
        weightedF1 /= totalSamples;
    }

    double acc = accuracy(predictions, targets);

    cout << string(64, '-') << endl;
    cout << left << setw(16) << "Accuracy"
         << right << setw(36) << fixed << setprecision(4) << acc
         << setw(12) << totalSamples << endl;
    cout << left << setw(16) << "Macro Avg"
         << right << fixed << setprecision(4)
         << setw(12) << macroP
         << setw(12) << macroR
         << setw(12) << macroF1
         << setw(12) << totalSamples << endl;
    cout << left << setw(16) << "Weighted Avg"
         << right << fixed << setprecision(4)
         << setw(12) << weightedP
         << setw(12) << weightedR
         << setw(12) << weightedF1
         << setw(12) << totalSamples << endl;
    cout << "=======================================================" << endl;
}

double Evaluator::meanSquaredError(const Matrix& predictions, const Matrix& targets) {
    int rows = predictions.getRows();
    int cols = predictions.getCols();
    if (rows == 0 || cols == 0) return 0.0;

    double sum = 0.0;
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            double diff = predictions(i, j) - targets(i, j);
            sum += diff * diff;
        }
    }
    return sum / (rows * cols);
}

double Evaluator::meanAbsoluteError(const Matrix& predictions, const Matrix& targets) {
    int rows = predictions.getRows();
    int cols = predictions.getCols();
    if (rows == 0 || cols == 0) return 0.0;

    double sum = 0.0;
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            sum += fabs(predictions(i, j) - targets(i, j));
        }
    }
    return sum / (rows * cols);
}

double Evaluator::r2Score(const Matrix& predictions, const Matrix& targets) {
    int rows = predictions.getRows();
    int cols = predictions.getCols();
    int total = rows * cols;
    if (total == 0) return 0.0;

    double meanTarget = 0.0;
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            meanTarget += targets(i, j);
        }
    }
    meanTarget /= total;

    double ssTot = 0.0;
    double ssRes = 0.0;
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            double diffTrue = targets(i, j) - meanTarget;
            double diffPred = targets(i, j) - predictions(i, j);
            ssTot += diffTrue * diffTrue;
            ssRes += diffPred * diffPred;
        }
    }
    if (ssTot < 1e-12) return 1.0;
    return 1.0 - (ssRes / ssTot);
}
