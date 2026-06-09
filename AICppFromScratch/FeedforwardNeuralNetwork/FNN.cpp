/**
 * @file FNN.cpp
 * @author Wei Li (Ithaca) (weili_yzzca@163.com)
 * @brief Feedforward Neural Network implementation
 * @version 0.1
 * @date 2026-06-08
 * 
 * @copyright Copyright (c) 2026
 * 
 * clang++ -std=c++17 FNN.cpp -o main.exe
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

    std::vector<double> forward(const std::vector<double> &input)
    {
        if (input.size() != layerSizes[0])
        {
            throw std::invalid_argument("Input size mismatch");
        }

        std::vector<double> hidden1(layerSizes[1]);
        for (int j = 0; j < layerSizes[1]; ++j)
        {
            double sum = bias1[j];
            for (int i = 0; i < layerSizes[0]; ++i)
            {
                sum += input[i] * weights1(i, j);
            }

            hidden1[j] = Activation::relu(sum);
        }

        std::vector<double> hidden2(layerSizes[2]);
        for (int j = 0; j < layerSizes[2]; ++j)
        {
            double sum = bias2[j];
            for (int i = 0; i < layerSizes[1]; ++i)
            {
                sum += input[i] * weights2(i, j);
            }

            hidden2[j] = Activation::relu(sum);
        }

        std::vector<double> output(layerSizes[3]);
        for (int j = 0; j < layerSizes[3]; ++j)
        {
            double sum = bias3[j];
            for (int i = 0; i < layerSizes[2]; ++i)
            {
                sum += input[i] * weights3(i, j);
            }

            output[j] = Activation::sigmoid(sum);
        }

        return output;
    }

    void train(const std::vector<std::vector<double>> &inputs, const std::vector<std::vector<double>> &targets,
               double learningRate, int epochs)
    {
        if (inputs.size() != targets.size() || inputs.empty())
        {
            throw std::invalid_argument("Input and target data size mismatch or empty");
        }

        for (int epoch = 0; epoch < epochs; ++epoch)
        {
            double totalError = 0.0;

            for (size_t k = 0; k < inputs.size(); k++)
            {
                std::vector<double> hidden1(layerSizes[1]);
                std::vector<double> hidden1Pre(layerSizes[1]);
                for (int j = 0; j < layerSizes[1]; j++)
                {
                    double sum = bias1[j];
                    for (int i = 0; i < layerSizes[0]; i++)
                    {
                        sum += inputs[k][i] * weights1(i, j);
                    }
                    hidden1Pre[j] = sum;
                    hidden1[j]    = Activation::relu(sum);
                }

                std::vector<double> hidden2(layerSizes[2]);
                std::vector<double> hidden2Pre(layerSizes[2]);
                for (int j = 0; j < layerSizes[2]; j++)
                {
                    double sum = bias2[j];
                    for (int i = 0; i < layerSizes[1]; i++)
                    {
                        sum += hidden1[i] * weights2(i, j);
                    }
                    hidden2Pre[j] = sum;
                    hidden2[j]    = Activation::relu(sum);
                }

                std::vector<double> output(layerSizes[3]);
                std::vector<double> outputPre(layerSizes[3]);
                for (int j = 0; j < layerSizes[3]; j++)
                {
                    double sum = bias3[j];
                    for (int i = 0; i < layerSizes[2]; i++)
                    {
                        sum += hidden2[i] * weights3(i, j);
                    }
                    outputPre[j] = sum;
                    output[j]    = Activation::sigmoid(sum);
                }

                for (int j = 0; j < layerSizes[3]; j++)
                {
                    double error = targets[k][j] - output[j];
                    totalError += error * error;
                }

                std::vector<double> outputGradients(layerSizes[3]);
                for (int j = 0; j < layerSizes[3]; j++)
                {
                    outputGradients[j] = (output[j] - targets[k][j]) * Activation::sigmoid_derivative(outputPre[j]);
                }

                std::vector<double> hidden2Gradients(layerSizes[2]);
                for (int i = 0; i < layerSizes[2]; i++)
                {
                    double error = 0;
                    for (int j = 0; j < layerSizes[3]; j++)
                    {
                        error += outputGradients[j] * weights3(i, j);
                    }

                    hidden2Gradients[i] = error * Activation::relu_derivative(hidden2Pre[i]);
                }

                std::vector<double> hidden1Gradients(layerSizes[1]);
                for (int i = 0; i < layerSizes[1]; i++)
                {
                    double error = 0;
                    for (int j = 0; j < layerSizes[2]; j++)
                    {
                        error += hidden2Gradients[j] * weights2(i, j);
                    }

                    hidden1Gradients[i] = error * Activation::relu_derivative(hidden1Pre[i]);
                }

                for (int i = 0; i < layerSizes[2]; i++)
                {
                    for (int j = 0; j < layerSizes[3]; j++)
                    {
                        weights3(i, j) -= learningRate * outputGradients[j] * hidden2[i];
                    }
                }

                for (int j = 0; j < layerSizes[3]; j++)
                {
                    bias3[j] -= learningRate * outputGradients[j];
                }

                for (int i = 0; i < layerSizes[1]; i++)
                {
                    for (int j = 0; j < layerSizes[2]; j++)
                    {
                        weights2(i, j) -= learningRate * hidden2Gradients[j] * hidden1[i];
                    }
                }

                for (int j = 0; j < layerSizes[2]; j++)
                {
                    bias2[j] -= learningRate * hidden2Gradients[j];
                }

                for (int i = 0; i < layerSizes[0]; i++)
                {
                    for (int j = 0; j < layerSizes[1]; j++)
                    {
                        weights1(i, j) -= learningRate * hidden1Gradients[j] * inputs[k][i];
                    }
                }

                for (int j = 0; j < layerSizes[1]; j++)
                {
                    bias1[j] -= learningRate * hidden1Gradients[j];
                }
            }

            if (epoch % 100 == 0)
            {
                std::cout << "EPOCH : " << epoch << "  MSE : " << totalError / inputs.size() << "\n";
            }
        }
    }
};

// ----------------------------------------
int main(int argc, const char *argv[])
{
    try
    {
        NeuralNetwork nn(2, 8, 4, 1);

        std::mt19937                     gen(std::random_device{}());
        std::uniform_real_distribution<> dist(-2.0, 2.0);
        const int                        numSamples = 1000;
        std::vector<std::vector<double>> inputs(numSamples);
        std::vector<std::vector<double>> targets(numSamples);
        for (int i = 0; i < numSamples; i++)
        {
            double x        = dist(gen);
            double y        = dist(gen);
            inputs[i]       = {x, y};
            double distance = sqrt(x * x + y * y);
            targets[i]      = {distance < 1.0 ? 1.0 : 0.0};
        }

        auto start = std::chrono::high_resolution_clock::now();
        nn.train(inputs, targets, 0.01, 1000);
        auto end = std::chrono::high_resolution_clock::now();
        std::cout << "Training Time : " << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count()
                  << " ms\n";
        std::vector<std::vector<double>> testPoints = {
            {0.0, 0.0},
            {1.0, 1.0},
            {0.5, 0.5},
            {2.0, 0.0}
        };

        std::cout << "\n Test Results (1 = inside , 0 = outside) : \n";
        for (const auto &point : testPoints)
        {
            auto   output = nn.forward(point);
            double actual = std::sqrt(point[0] * point[0] + point[1] * point[1]) < 1.0 ? 1.0 : 0.0;
            std::cout << " point ( " << point[0] << "," << point[1] << " ) :" << output[0] << " ( actual : " << actual
                      << " , error : " << std::abs(output[0] - actual) << " ) \n";
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << " ERROR : " << e.what() << std::endl;
        return 1;
    }

    return 0;
}
