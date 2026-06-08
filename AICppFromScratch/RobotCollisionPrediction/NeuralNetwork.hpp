/**
 * @file NeuralNetwork.hpp
 * @author Wei Li (Ithaca) (weili_yzzca@163.com)
 * @brief Neural Network implementation
 * @version 0.1
 * @date 2026-06-07
 * 
 * @copyright Copyright (c) 2026
 * 
 */
#pragma once

#include "Layer.hpp"
#include "Logger.hpp"
#include "Matrix.hpp"

#include <chrono>
#include <cmath>
#include <sstream>
#include <stdexcept>
#include <vector>

namespace Ithaca {

class NeuralNetwork
{
private:
    std::vector<Layer>  layers;
    std::vector<Matrix> weights;

    void connect_layers()
    {
        for (size_t i = 0; i < layers.size() - 1; ++i)
        {
            weights.emplace_back(layers[i].size, layers[i + 1].size);
        }
    }

    void init_weights()
    {
        for (auto &ws : weights)
        {
            ws.fillRandom();
            // scale them using He. initialization
            ws *= std::sqrt(2.0 / ws.getRows());
        }
    }

public:
    NeuralNetwork(const std::vector<Layer> &netwrok_layers)
        : layers(netwrok_layers)
    {
        connect_layers();
        init_weights();
    }

    std::vector<double> forward(const std::vector<double> &input)
    {
        if (input.size() != layers[0].size)
        {
            throw std::runtime_error("Input size does not match");
        }

        layers[0].a = input;

        for (size_t l = 0; l < layers.size() - 1; ++l)
        {
            // weights.size() = layers.size() - 1
            for (size_t j = 0; j < weights[l].getCols(); ++j)
            {
                double sum = layers[l + 1].bias[j];

                for (size_t i = 0; i < weights[l].getRows(); ++i)
                {
                    sum += weights[l](i, j) * layers[l].a[i];
                }

                layers[l + 1].z[j] = sum;
                layers[l + 1].a[j] = layers[l + 1].applyActivation(sum);
            }
        }

        return layers.back().a;
    }

    std::vector<double> predict(const std::vector<double> &input)
    {
        return forward(input);
    }

    void train(const std::vector<std::vector<double>> &inputs, const std::vector<std::vector<double>> &targets,
               double learning_rate, size_t epochs, size_t batch_size = 1, bool verbose = true)
    {
        if (inputs.size() != targets.size())
        {
            throw std::runtime_error("Input and Target sizes don't match !");
        }
        if (learning_rate <= 0.0 || epochs <= 0 || batch_size <= 0)
        {
            throw std::runtime_error("Learning rate, epochs, and batch size must be positive.");
        }

        size_t dataset_size = inputs.size();
        auto   start        = std::chrono::high_resolution_clock::now();
        double totalError   = 0.0;
        double base_lr      = learning_rate;
        double decay_rate   = 0.996; // decay by 0.4% each epoch
        double min_lr       = 1e-4;  // prevent it from vanishing
        if (verbose)
        {
            std::stringstream data;
            data << "---------------------training info-------------------\n";
            data << "learning_rate = " << learning_rate << "\t learning rate decay rate = " << (1.0 - decay_rate) * 100
                 << "%"
                 << "\t EPOCHS = " << epochs << "\n";

            data << "dataset size = " << dataset_size << "\t batch size = " << batch_size << "\n";
            data << "\n--------------------training ... ------------------\n";
            std::cout << data.str();
            Ithaca::log(data.str());
        }

        for (int epoch = 0; epoch < epochs; ++epoch)
        {
            learning_rate = std::max(min_lr, base_lr * std::pow(decay_rate, epoch));
            totalError    = 0.0;

            for (size_t batch = 0; batch < dataset_size; batch += batch_size)
            {
                size_t actual_batch_size = std::min(batch_size, (dataset_size - batch));

                std::vector<Matrix> weight_batch_gradients;
                for (size_t i = 0; i < weights.size(); ++i)
                {
                    weight_batch_gradients.emplace_back(weights[i].getRows(), weights[i].getCols());
                }
                std::vector<std::vector<double>> bias_batch_gradients(layers.size() - 1);
                for (size_t i = 0; i < bias_batch_gradients.size(); ++i)
                {
                    bias_batch_gradients[i].resize(layers[i + 1].size, 0.0);
                }

                for (size_t k = batch; k < (batch + actual_batch_size); ++k)
                {
                    forward(inputs[k]);
                    //compute Gradients
                    //compute the outputGradients
                    Layer       &outputLayer = layers.back();
                    const double epsilon     = 1e-7;
                    for (size_t i = 0; i < outputLayer.size; ++i)
                    {
                        double y_true = targets[k][i];
                        double y_pred = outputLayer.a[i];

                        // Clamp y_pred to avoid log(0)
                        y_pred = std::min(std::max(y_pred, epsilon), 1.0 - epsilon);

                        // Binary cross-entropy loss
                        //L=−(ylog( y ^ ​ )+(1−y)log(1− y ^ ​ ))
                        totalError += -(y_true * std::log(y_pred) + (1.0 - y_true) * std::log(1.0 - y_pred));

                        // Gradient: derivative of BCE w/ sigmoid output
                        outputLayer.gradient[i] = y_pred - y_true;
                    }
                    //compute other layers Gradients
                    for (int l = (static_cast<int>(layers.size()) - 2); l > 0; --l)
                    {
                        for (size_t i = 0; i < weights[l].getRows(); ++i)
                        {
                            double error = 0.0;
                            for (size_t j = 0; j < weights[l].getCols(); ++j)
                            {
                                error += layers[l + 1].gradient[j] * weights[l](i, j);
                            }
                            layers[l].gradient[i] = error * layers[l].applyActivationDerivative(layers[l].z[i]);
                        }
                    }
                    //Accumulate gradients
                    for (int l = (static_cast<int>(layers.size()) - 2); l >= 0; --l)
                    {
                        const std::vector<double> &activations = (l == 0) ? inputs[k] : layers[l].a;

                        for (size_t i = 0; i < weight_batch_gradients[l].getRows(); ++i)
                        {
                            for (size_t j = 0; j < weight_batch_gradients[l].getCols(); ++j)
                            {
                                weight_batch_gradients[l](i, j) += layers[l + 1].gradient[j] * activations[i];
                            }
                        }

                        for (size_t i = 0; i < layers[l + 1].size; ++i)
                        {
                            //bias_batch_gradients size is (layer - 1 )
                            //(EX: for 4 layers it is 3 it means for l: 2 to 0 bias_batch_gradients[l])
                            bias_batch_gradients[l][i] += layers[l + 1].gradient[i];
                        }
                    }

                    //end of batch loop
                }

                //update the weights and biases
                for (int l = (static_cast<int>(layers.size()) - 2); l >= 0; --l)
                {
                    for (size_t i = 0; i < weights[l].getRows(); ++i)
                    {
                        for (size_t j = 0; j < weights[l].getCols(); ++j)
                        {
                            weights[l](i, j) -= learning_rate * (weight_batch_gradients[l](i, j) / actual_batch_size);
                        }
                    }

                    for (size_t i = 0; i < layers[l + 1].size; ++i)
                    {
                        layers[l + 1].bias[i] -= learning_rate * (bias_batch_gradients[l][i] / actual_batch_size);
                    }
                }
            }
            if (verbose && epoch % 100 == 0)
            {
                std::stringstream data;

                data << "[" << (100 * epoch / epochs) << "%] EPOCH : " << epoch
                     << " | BCE: " << totalError / inputs.size() << "\n";
                std::cout << data.str();
                Ithaca::log(data.str());
            }
        }
        auto end = std::chrono::high_resolution_clock::now();
        if (verbose)
        {
            std::stringstream data;
            data << "-------------------training done---------------------\n";
            data << "Final BCE : " << totalError / inputs.size() << "\n";
            data << "Training Time : " << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count()
                 << " ms\n";
            std::cout << data.str();
            Ithaca::log(data.str());
        }
    }

    void saveModel(const std::string &filename = "model.csv") const
    {
        std::ofstream file(filename);
        if (!file.is_open())
        {
            std::cerr << "Unable to open file: " << filename << std::endl;
            return;
        }

        file << "type,layer,row,col,value\n"; // header

        for (size_t l = 1; l < layers.size(); ++l)
        {
            for (size_t i = 0; i < weights[l - 1].getRows(); ++i)
            {
                for (size_t j = 0; j < weights[l - 1].getCols(); ++j)
                {
                    file << "weight," << l << "," << i << "," << j << "," << weights[l - 1](i, j) << "\n";
                }
            }

            for (size_t i = 0; i < layers[l].bias.size(); ++i)
            {
                file << "bias," << l << "," << i << ",0," << layers[l].bias[i] << "\n";
            }
        }

        file.close();
        std::cout << "Model saved to " << filename << std::endl;
    }

    void loadModel(const std::string &filename = "model.csv")
    {
        std::ifstream file(filename);
        if (!file.is_open())
        {
            std::cerr << "Unable to open model file: " << filename << std::endl;
            return;
        }

        std::string line;
        std::getline(file, line); // Skip header

        while (std::getline(file, line))
        {
            std::stringstream ss(line);
            std::string       type, layerStr, rowStr, colStr, valueStr;

            std::getline(ss, type, ',');
            std::getline(ss, layerStr, ',');
            std::getline(ss, rowStr, ',');
            std::getline(ss, colStr, ',');
            std::getline(ss, valueStr, ',');

            int    layer = std::stoi(layerStr);
            int    row   = std::stoi(rowStr);
            int    col   = std::stoi(colStr);
            double value = std::stod(valueStr);

            if (type == "weight")
            {
                weights[layer - 1](row, col) = value;
            }
            else if (type == "bias")
            {
                layers[layer].bias[row] = value;
            }
        }

        file.close();
        std::cout << "Model loaded from " << filename << std::endl;
    }
};

} // namespace Ithaca