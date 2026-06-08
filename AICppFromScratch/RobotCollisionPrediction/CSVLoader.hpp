/**
 * @file CSVLoader.hpp
 * @author Wei Li (Ithaca) (weili_yzzca@163.com)
 * @brief 
 * @version 0.1
 * @date 2026-06-08
 * 
 * @copyright Copyright (c) 2026
 * 
 */

#pragma once

#include "Logger.hpp"

#include <algorithm> // for std::shuffle
#include <fstream>
#include <functional>
#include <iomanip>
#include <iostream>
#include <numeric> // for std::iota
#include <random>  // for std::mt19937, std::random_device
#include <sstream>
#include <string>
#include <unordered_set>
#include <vector>

namespace Ithaca::CSV {

// Treat labels not equal to "Move-Forward" as collisions
// Move-Forward - Slight-Right-Turn - Sharp-Right-Turn - Move-Forward
bool isCollisionLabel(const std::string &label)
{
    static std::unordered_set<std::string> noCollisionLabels = {"Move-Forward"};
    return noCollisionLabels.count(label) == 0;
}

// Parse a single CSV line into features and label
bool parseLine(const std::string &line, std::vector<double> &features, int &label,
               const std::function<int(const std::string &)> &labelMapper)
{
    std::istringstream  ss(line);
    std::string         token;
    std::vector<double> tempFeatures;

    // Expecting 24 sensor values
    for (int i = 0; i < 24; ++i)
    {
        if (!std::getline(ss, token, ','))
        {
            return false;
        }
        try
        {
            tempFeatures.push_back(std::stod(token));
        }
        catch (...)
        {
            return false;
        }
    }

    // Read final label
    if (!std::getline(ss, token))
    {
        return false;
    }
    label = labelMapper(token);

    features = std::move(tempFeatures);
    return true;
}

void printClassBalance(const std::vector<std::vector<double>> &features, const std::vector<std::vector<double>> &labels)
{
    size_t positive = 0, negative = 0;
    for (const auto &label : labels)
    {
        if (!label.empty() && label[0] >= 0.5)
        {
            positive++;
        }
        else
        {
            negative++;
        }
    }

    size_t total       = labels.size();
    double pos_percent = 100.0 * positive / total;
    double neg_percent = 100.0 * negative / total;

    std::stringstream data;
    data << "----------- Class Balance:  -----------\n";
    data << "Total samples: " << total << "\n";
    data << "Positive (Collision = 1): " << positive << " (" << pos_percent << "%)\n";
    data << "Negative (No Collision = 0): " << negative << " (" << neg_percent << "%)\n";
    data << "---------------------------------------------------\n";

    std::cout << data.str();
    Ithaca::log(data.str());
}

// Full definition with labelMapper
bool loadSensorData(const std::string &filename, std::vector<std::vector<double>> &features,
                    std::vector<std::vector<double>> &labels, std::vector<int> &ids,
                    const std::function<int(const std::string &)> &labelMapper)
{
    std::ifstream file(filename);
    if (!file.is_open())
    {
        std::cerr << "Failed to open file: " << filename << "\n";
        return false;
    }

    std::string line;
    int         line_id = 0;

    while (std::getline(file, line))
    {
        std::vector<double> sample;
        int                 label;
        if (parseLine(line, sample, label, labelMapper))
        {
            features.push_back(sample);
            labels.push_back({static_cast<double>(label)}); // convert to vector
            ids.push_back(line_id++);
        }
    }

    return !features.empty();
}

// Splits into train/test sets
bool loadAndSplitSensorData(
    const std::string &filename, std::vector<std::vector<double>> &training_features,
    std::vector<std::vector<double>> &training_labels, std::vector<int> &training_ids,
    std::vector<std::vector<double>> &test_features, std::vector<std::vector<double>> &test_labels,
    std::vector<int> &test_ids, double train_ratio = 0.8,
    const std::function<int(const std::string &)> &labelMapper
    = [](const std::string &l) { return isCollisionLabel(l) ? 1 : 0; })
{
    std::vector<std::vector<double>> all_features;
    std::vector<std::vector<double>> all_labels;
    std::vector<int>                 all_ids;

    if (!loadSensorData(filename, all_features, all_labels, all_ids, labelMapper))
    {
        return false;
    }

    std::vector<size_t> indices(all_features.size());
    std::iota(indices.begin(), indices.end(), 0);
    std::mt19937 gen(std::random_device{}());
    std::shuffle(indices.begin(), indices.end(), gen);

    std::vector<std::vector<double>> features_shuffled;
    std::vector<std::vector<double>> labels_shuffled;
    std::vector<int>                 ids_shuffled;

    features_shuffled.reserve(all_features.size());
    labels_shuffled.reserve(all_labels.size());
    ids_shuffled.reserve(all_ids.size());

    for (size_t i : indices)
    {
        features_shuffled.push_back(std::move(all_features[i]));
        labels_shuffled.push_back(std::move(all_labels[i]));
        ids_shuffled.push_back(all_ids[i]);
    }

    printClassBalance(features_shuffled, labels_shuffled);

    size_t train_size = static_cast<size_t>(train_ratio * features_shuffled.size());

    training_features.assign(features_shuffled.begin(), features_shuffled.begin() + train_size);
    training_labels.assign(labels_shuffled.begin(), labels_shuffled.begin() + train_size);
    training_ids.assign(ids_shuffled.begin(), ids_shuffled.begin() + train_size);

    test_features.assign(features_shuffled.begin() + train_size, features_shuffled.end());
    test_labels.assign(labels_shuffled.begin() + train_size, labels_shuffled.end());
    test_ids.assign(ids_shuffled.begin() + train_size, ids_shuffled.end());

    std::stringstream data;

    data << "Dataset split: " << train_size << " training samples, " << (features_shuffled.size() - train_size)
         << " test samples.\n";
    data << "-----------------training set-------------------" << std::endl;
    data << "Features: " << training_features.size() << " samples, "
         << (training_features.empty() ? 0 : training_features[0].size()) << " features each\n";
    data << "Labels: " << training_labels.size() << "\n";
    data << "IDs: " << training_ids.size() << "\n";
    data << "-----------------test set-------------------" << std::endl;
    data << "Features: " << test_features.size() << " samples, "
         << (test_features.empty() ? 0 : test_features[0].size()) << " features each\n";
    data << "Labels: " << test_labels.size() << "\n";
    data << "IDs: " << test_ids.size() << "\n";
    data << "------------------------------------" << std::endl;

    std::cout << data.str();
    Ithaca::log(data.str());

    return true;
}

} // namespace Ithaca::CSV