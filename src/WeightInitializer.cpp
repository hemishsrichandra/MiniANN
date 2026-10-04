#include "../include/WeightInitializer.h"
#include <random>
#include <cmath>

RandomUniform::RandomUniform(double min, double max) : min_val(min), max_val(max) {}

void RandomUniform::initialize(Matrix& weights, int fanIn, int fanOut) {
    random_device rd;
    mt19937 gen(rd());
    uniform_real_distribution<> dis(min_val, max_val);
    
    for (int i = 0; i < weights.getRows(); ++i) {
        for (int j = 0; j < weights.getCols(); ++j) {
            weights(i, j) = dis(gen);
        }
    }
}

void Xavier::initialize(Matrix& weights, int fanIn, int fanOut) {
    random_device rd;
    mt19937 gen(rd());
    double limit = sqrt(6.0 / (fanIn + fanOut));
    uniform_real_distribution<> dis(-limit, limit);
    
    for (int i = 0; i < weights.getRows(); ++i) {
        for (int j = 0; j < weights.getCols(); ++j) {
            weights(i, j) = dis(gen);
        }
    }
}

void He::initialize(Matrix& weights, int fanIn, int fanOut) {
    random_device rd;
    mt19937 gen(rd());
    double stddev = sqrt(2.0 / fanIn);
    normal_distribution<> dis(0.0, stddev);
    
    for (int i = 0; i < weights.getRows(); ++i) {
        for (int j = 0; j < weights.getCols(); ++j) {
            weights(i, j) = dis(gen);
        }
    }
}
