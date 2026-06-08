/**
 * @file Layer.hpp
 * @author Wei Li (Ithaca) (weili_yzzca@163.com)
 * @brief Layer implementation for neural network
 * @version 0.1
 * @date 2026-06-07
 * 
 * @copyright Copyright (c) 2026
 * 
 */
#pragma once

#include <cmath>
#include <functional>
#include <stdexcept>
#include <vector>

namespace Ithaca {
using ActivationFunction = std::function<double(double)>;

namespace Activation {
inline double sigmoid(double x)
{
    return 1.0 / (1.0 + std::exp(-x));
}

inline double sigmoidDerivative(double x)
{
    double sig = sigmoid(x);
    return sig * (1.0 - sig);
}

inline double relu(double x)
{
    // return std::max(0.0, x);
    return (x > 0.0) ? x : 0.0;
}

inline double reluDerivative(double x)
{
    return (x > 0.0) ? 1.0 : 0.0;
}
} // namespace Activation

enum class ActivationType
{
    None,
    Sigmoid,
    Relu,
};

inline std::pair<ActivationFunction, ActivationFunction> getActivationPair(ActivationType type)
{
    using namespace Activation;
    switch (type)
    {
    case ActivationType::Sigmoid:
        return {sigmoid, sigmoidDerivative};
    case ActivationType::Relu:
        return {relu, reluDerivative};
    case ActivationType::None:
    default:
        return {ActivationFunction{}, ActivationFunction{}};
    }
}

struct Layer
{
private:
    ActivationFunction activation;
    ActivationFunction activation_derivative;

public:
    int                 layer_index;
    int                 size;
    std::vector<double> z; // before activation
    std::vector<double> a; // after activation
    std::vector<double> bias;
    std::vector<double> gradient;

    Layer(int index, int size, ActivationType act_type)
        : layer_index(index)
        , size(size)
        , z(size, 0.0)
        , a(size, 0.0)
    {
        if (size <= 0)
        {
            throw std::invalid_argument("Layer size must be greater than zero");
        }

        // initialize bias for non-input layers
        if (0 != index)
        {
            gradient              = std::vector<double>(size, 0.0);
            bias                  = std::vector<double>(size, 0.0);
            activation            = getActivationPair(act_type).first;
            activation_derivative = getActivationPair(act_type).second;
        }
    }

    double applyActivation(double x) const
    {
        if (!activation)
        {
            throw std::runtime_error("This Layer Activation function not set");
        }

        return activation(x);
    }

    double applyActivationDerivative(double x) const
    {
        if (!activation_derivative)
        {
            throw std::runtime_error("This Layer Activation Derivative function not set");
        }

        return activation_derivative(x);
    }

    bool hasActivation() const
    {
        return static_cast<bool>(activation);
    }

    bool hasDerivative() const
    {
        return static_cast<bool>(activation_derivative);
    }
};
} // namespace Ithaca
