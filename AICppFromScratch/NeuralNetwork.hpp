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
#include "Matrix.hpp"

#include <chrono>
#include <cmath>
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
};

} // namespace Ithaca