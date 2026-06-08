/**
 * @file main.cpp
 * @author Wei Li (Ithaca) (weili_yzzca@163.com)
 * @brief Build-Neural-Network-AI-for-Robot-Collision-Prediction
 * @version 0.1
 * @date 2026-06-08
 * 
 * @copyright Copyright (c) 2026
 * 
 * g++ -std=c++17 main.cpp -o main.exe
 * clang++ -std=c++17 main.cpp -o main.exe
 * 
 */

#include "CSVLoader.hpp"
#include "Layer.hpp"
#include "Logger.hpp"
#include "NeuralNetwork.hpp"

#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <vector>

// ------------------------------------
int main(int argc, const char *argv[])
{
    try
    {
        std::vector<Ithaca::Layer> layers;

        layers.emplace_back(0, 24, Ithaca::ActivationType::None);
        layers.emplace_back(1, 12, Ithaca::ActivationType::Relu);
        layers.emplace_back(2, 1, Ithaca::ActivationType::Sigmoid);

        Ithaca::NeuralNetwork nn(layers);

        std::vector<std::vector<double>> training_features;
        std::vector<std::vector<double>> training_labels;
        std::vector<int>                 training_ids;

        std::vector<std::vector<double>> test_features;
        std::vector<std::vector<double>> test_labels;
        std::vector<int>                 test_ids;

        std::string load_model;

        if (Ithaca::CSV::loadAndSplitSensorData("sensor_readings_24.csv", training_features, training_labels,
                                                training_ids, test_features, test_labels, test_ids, 0.8))
        {
            std::cout << "Do you want to load the model (y/n): ";
            std::cin >> load_model;
            if (load_model == "y" || load_model == "Y")
            {
                // load model
                std::cout << "Loading the model...\n";
                nn.loadModel();
            }
            else
            {
                load_model = "n";
                nn.train(training_features, training_labels, 0.029, 1300, 8);
            }
        }
        else
        {
            std::cerr << "Failed to load sensor data !\n";
            return 1;
        }

        std::cout << "-------------------Predictions--------------------\n";
        Ithaca::log("-------------------Predictions--------------------\n");
        for (size_t i = 0; i < test_features.size(); ++i)
        {
            std::stringstream data;
            data << "-------------------Prediction[" << i << "]--------------------\n";
            data << "Row ID     : " << test_ids[i] << "\n";

            data << "Sensor Data: ";
            for (double val : test_features[i])
            {
                data << std::fixed << std::setprecision(2) << val << " ";
            }
            data << "\n";

            std::vector<double> prediction = nn.predict(test_features[i]);

            data << "Prediction : " << std::fixed << std::setprecision(4) << prediction[0] << "\n";
            data << "Actual     : " << test_labels[i][0] << "\n";

            std::cout << data.str();
            Ithaca::log(data.str());
        }

        int tp = 0, tn = 0, fp = 0, fn = 0;

        for (size_t i = 0; i < test_features.size(); ++i)
        {
            double pred      = nn.predict(test_features[i])[0];
            int    predicted = pred >= 0.5 ? 1 : 0;
            int    actual    = static_cast<int>(test_labels[i][0]);

            if (predicted == 1 && actual == 1)
                tp++;
            else if (predicted == 0 && actual == 0)
                tn++;
            else if (predicted == 1 && actual == 0)
                fp++;
            else if (predicted == 0 && actual == 1)
                fn++;
        }

        int    total     = tp + tn + fp + fn;
        double accuracy  = static_cast<double>(tp + tn) / total;
        double precision = (tp + fp) == 0 ? 0.0 : static_cast<double>(tp) / (tp + fp);
        double recall    = (tp + fn) == 0 ? 0.0 : static_cast<double>(tp) / (tp + fn);
        double f1_score  = (precision + recall) == 0 ? 0.0 : 2.0 * (precision * recall) / (precision + recall);

        std::stringstream data;
        // Display results
        data << "-------------------Metrics---------------------\n";
        data << "\nConfusion Matrix:\n";
        data << "TP: " << tp << " | FP: " << fp << "\n";
        data << "FN: " << fn << " | TN: " << tn << "\n";

        data << std::fixed << std::setprecision(4);
        data << "\nAccuracy : " << accuracy * 100 << "%\n";
        data << "Precision: " << precision * 100 << "%\n";
        data << "Recall   : " << recall * 100 << "%\n";
        data << "F1 Score : " << f1_score * 100 << "%\n";
        std::cout << data.str();
        Ithaca::log(data.str());

        if (load_model == "n")
        {
            std::string save_model;
            std::cout << "Do you want to save the model (y/n): ";
            std::cin >> save_model;
            if (save_model == "y" || save_model == "Y")
            {
                // Save model
                std::cout << "Saving the model...\n";
                nn.saveModel();
            }
            else
            {
                std::cout << "Model not saved.\n";
            }
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << " ERROR : " << e.what() << std::endl;
        return 1;
    }

    return 0;
}
