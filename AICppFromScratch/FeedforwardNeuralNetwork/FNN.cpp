/**
 * @file FNN.cpp
 * @author Wei Li (Ithaca) (weili_yzzca@163.com)
 * @brief Feedforward Neural Network implementation
 * @version 0.1
 * @date 2026-06-08
 * 
 * @copyright Copyright (c) 2026
 * 
 */

#include <chrono>
#include <cmath>
#include <iostream>
#include <random>
#include <stdexcept>
#include <vector>

namespace Activation {
inline double relu(double x)
{
    return std::max(0.0, x);
}

inline double relu_derivative(double x)
{
    return x > 0.0 ? 1.0 : 0.0;
}

inline double sigmoid(double x)
{
    return 1.0 / (1.0 + std::exp(-x));
}

inline double sigmoid_derivative(double x)
{
    return sigmoid(x) * (1.0 - sigmoid(x));
}
} // namespace Activation

class Matrix
{
private:
    std::vector<std::vector<double>> data;
    size_t                           rows;
    size_t                           cols;

public:
    Matrix(size_t r, size_t c)
        : rows(r)
        , cols(c)
    {
        data.resize(r, std::vector<double>(c, 0.0));
    }

    double &operator()(size_t r, size_t c)
    {
        return data[r][c];
    }

    const double &operator()(size_t r, size_t c) const
    {
        return data[r][c];
    }

    size_t getRows() const
    {
        return rows;
    }

    size_t getCols() const
    {
        return cols;
    }
};

class NeuralNetwork
{
private:
    std::vector<int>    layerSizes;
    Matrix              weights1, weights2, weights3;
    std::vector<double> bias1, bias2, bias3;
    std::mt19937        gen;

    void initializeWeights()
    {
        std::normal_distribution<> dist(0.0, 1.0);

        double scale1 = std::sqrt(2.0 / layerSizes[0]);
        double scale2 = std::sqrt(2.0 / layerSizes[1]);
        double scale3 = std::sqrt(2.0 / layerSizes[2]);

        // He initialization
        for (size_t i = 0; i < weights1.getRows(); ++i)
        {
            for (size_t j = 0; j < weights1.getCols(); ++j)
            {
                weights1(i, j) = dist(gen) * scale1;
            }
        }
        for (size_t i = 0; i < weights2.getRows(); ++i)
        {
            for (size_t j = 0; j < weights2.getCols(); ++j)
            {
                weights2(i, j) = dist(gen) * scale2;
            }
        }
        for (size_t i = 0; i < weights3.getRows(); ++i)
        {
            for (size_t j = 0; j < weights3.getCols(); ++j)
            {
                weights3(i, j) = dist(gen) * scale3;
            }
        }

        std::fill(bias1.begin(), bias1.end(), 0.0);
        std::fill(bias2.begin(), bias2.end(), 0.0);
        std::fill(bias3.begin(), bias3.end(), 0.0);
    }

public:
    NeuralNetwork(int inputSize, int hidden1Size, int hidden2Size, int outputSize)
        : layerSizes{inputSize, hidden1Size, hidden2Size, outputSize}
        , weights1(inputSize, hidden1Size)
        , weights2(hidden1Size, hidden2Size)
        , weights3(hidden2Size, outputSize)
        , bias1(hidden1Size)
        , bias2(hidden2Size)
        , bias3(outputSize)
        , gen(std::random_device{}())
    {
        if (inputSize <= 0 || hidden1Size <= 0 || hidden2Size <= 0 || outputSize <= 0)
        {
            throw std::invalid_argument("Layer sizes must be greater than zero");
        }

        initializeWeights();
    }

    void feedForward(const std::vector<double> &input);
    void backPropagate(const std::vector<double> &target);
    void train(const std::vector<std::vector<double>> &inputs, const std::vector<std::vector<double>> &targets,
               int epochs);
};

// ----------------------------------------
int main(int argc, const char *argv[])

{
    return 0;
}
