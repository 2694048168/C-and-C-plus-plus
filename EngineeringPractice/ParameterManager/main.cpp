/**
 * @file main.cpp
 * @author Wei Li (Ithaca) (weili_yzzcq@163.com)
 * @brief 
 * @version 0.1
 * @date 2025-08-27
 * 
 * @copyright Copyright (c) 2025
 * 
 * g++ main.cpp -std=c++20
 * clang++ main.cpp -std=c++20
 * 
 * 
 */

#include "ParameterManager.hpp"

#include <iostream>

// ------------------------------------
int main(int argc, const char **argv)
{
    // 设置基本类型参数
    ParameterManager parameterManager;
    parameterManager.SetParam("model_name", std::string("res-net"));
    parameterManager.SetParam("learning_rate", 0.001);
    parameterManager.SetParam("batch_size", 64);

    // 设置嵌套字典参数
    ParameterManager optimizerParameter;
    optimizerParameter.SetParam("type", std::string("Adam"));
    optimizerParameter.SetParam("weight_decay", 1e-5);
    parameterManager.SetParam("optimizer_config", optimizerParameter);

    // 或者使用点号语法('.')设置嵌套参数
    parameterManager.SetParam("dataset.name", std::string("image-net"));
    parameterManager.SetParam("dataset.num_classes", 1000);

    // 获取参数
    try
    {
        std::string model_name = parameterManager.GetParam<std::string>("model_name");
        double      lr         = parameterManager.GetParam<double>("learning_rate");
        int         batch_size = parameterManager.GetParam<int>("batch_size");

        std::cout << "Model: " << model_name << std::endl;
        std::cout << "Learning rate: " << lr << std::endl;
        std::cout << "Batch size: " << batch_size << std::endl;

        // 获取嵌套参数
        auto        optimizer      = parameterManager.GetParam<ParameterManager>("optimizer_config");
        std::string optimizer_type = optimizer.GetParam<std::string>("type");
        double      weight_decay   = optimizer.GetParam<double>("weight_decay");

        std::cout << "Optimizer: " << optimizer_type << std::endl;
        std::cout << "Weight decay: " << weight_decay << std::endl;

        // 使用点号语法获取嵌套参数
        std::string dataset_name = parameterManager.GetParam<std::string>("dataset.name");
        int         num_classes  = parameterManager.GetParam<int>("dataset.num_classes");

        std::cout << "Dataset: " << dataset_name << std::endl;
        std::cout << "Num classes: " << num_classes << std::endl;
    }
    catch (const std::exception &ex)
    {
        std::cerr << "---->Error: " << ex.what() << std::endl;
    }

    // system("pause");
    return 0;
}
