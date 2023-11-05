/**
 * @file csv_file.cpp
 * @author Wei Li (weili_yzzcq@163.com)
 * @brief
 * @version 0.1
 * @date 2023-11-05
 *
 * @copyright Copyright (c) 2023
 *
 */

#include <array>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

void write_csv(std::string filename, std::string col_name, std::vector<float> values)
{
    // Make a CSV file with one column of float values
    // filename - the name of the file
    // colname - the name of the one and only column
    // vals - an integer vector of values

    std::ofstream csv_file(filename, std::ios::out | std::ios::app);

    csv_file << col_name << '\n';

    for (const auto &elem : values) csv_file << elem << '\n';

    csv_file.close();
    std::cout << "CSV file write successfully\n";
}

void write_csv(std::string filename, std::vector<std::pair<std::string, std::vector<float>>> dataset)
{
    // Make a CSV file with one or more columns of float values
    // Each column of data is represented by the pair <column name, column data>
    //   as std::pair<std::string, std::vector<float>>
    // The dataset is represented as a vector of these columns
    // Note that all columns should be the same size

    std::ofstream csv_file;

    if (!std::filesystem::exists(filename))
    {
        // Create an output filestream object
        csv_file.open(filename, std::ios::out | std::ios::app);

        // Send column names to the stream
        for (int j = 0; j < dataset.size(); ++j)
        {
            csv_file << dataset.at(j).first;
            if (j != dataset.size() - 1)
                csv_file << ","; // No comma at end of line
        }
        csv_file << "\n";
    }
    else
    {
        csv_file.open(filename, std::ios::out | std::ios::app);
    }

    // Send data to the stream
    for (int i = 0; i < dataset.at(0).second.size(); ++i)
    {
        for (int j = 0; j < dataset.size(); ++j)
        {
            csv_file << dataset.at(j).second.at(i);
            if (j != dataset.size() - 1)
                csv_file << ","; // No comma at end of line
        }
        csv_file << "\n";
    }

    // Close the file
    csv_file.close();
    std::cout << "CSV file write successfully\n";
}

constexpr size_t SIZE_COL = 3;

struct DataResults
{
    float width;
    float height;
    float total;
};

void write_csv(const std::string filename, const std::vector<std::string> &col_names, const DataResults &values)
{
    // Make a CSV file with one or more columns of float values
    // Each column of data is represented by the pair <column name, column data>
    //   as std::pair<std::string, std::vector<float>>
    // The dataset is represented as a vector of these columns
    // Note that all columns should be the same size

    std::ofstream csv_file;

    if (!std::filesystem::exists(filename))
    {
        // Create an output filestream object
        csv_file.open(filename, std::ios::out | std::ios::app);

        // Send column names to the stream
        for (int j = 0; j < col_names.size(); ++j)
        {
            csv_file << col_names[j];
            if (j != col_names.size() - 1)
                csv_file << ","; // No comma at end of line
        }
        csv_file << "\n";
    }
    else
    {
        csv_file.open(filename, std::ios::out | std::ios::app);
    }

    csv_file << values.width << ',' << values.height << ',' << values.total << "\n";

    // Close the file
    csv_file.close();
    std::cout << "CSV file write successfully\n";
}

//template<typename Container>
//void write_csv(const std::string filename, const Container& col_names, const DataResults& values) {
void write_csv(const std::string filename, const std::array<std::string, SIZE_COL> &col_names,
               const DataResults &values)
{
    // Make a CSV file with one or more columns of float values
    // Each column of data is represented by the pair <column name, column data>
    //   as std::pair<std::string, std::vector<float>>
    // The dataset is represented as a vector of these columns
    // Note that all columns should be the same size

    std::ofstream csv_file;

    if (!std::filesystem::exists(filename))
    {
        // Create an output filestream object
        csv_file.open(filename, std::ios::out | std::ios::app);

        // Send column names to the stream
        for (int j = 0; j < col_names.size(); ++j)
        {
            csv_file << col_names[j];
            if (j != col_names.size() - 1)
                csv_file << ","; // No comma at end of line
        }
        csv_file << "\n";
    }
    else
    {
        csv_file.open(filename, std::ios::out | std::ios::app);
    }

    csv_file << values.width << ',' << values.height << ',' << values.total << "\n";

    // Close the file
    csv_file.close();
    std::cout << "CSV file write successfully\n";
}

std::vector<std::pair<std::string, std::vector<float>>> read_csv(std::string filename)
{
    // Reads a CSV file into a vector of <string, vector<float>> pairs where
    // each pair represents <column name, column values>

    // Create a vector of <string, int vector> pairs to store the result
    std::vector<std::pair<std::string, std::vector<float>>> result;

    // Create an input file-stream
    std::ifstream csv_file(filename);

    // Make sure the file is open
    if (!csv_file.is_open())
        throw std::runtime_error("Could not open file");

    // Helper vars
    std::string line, colName;

    // Read the column names
    if (csv_file.good())
    {
        // Extract the first line in the file
        std::getline(csv_file, line);

        // Create a stringstream from line
        std::stringstream ss(line);

        // Extract each column name
        while (std::getline(ss, colName, ','))
        {
            // Initialize and add <colName, int vector> pairs to result
            result.push_back({colName, std::vector<float>{}});
        }
    }

    float val;
    // Read data, line by line
    while (std::getline(csv_file, line))
    {
        // Create a stringstream of the current line
        std::stringstream ss(line);

        // Keep track of the current column index
        int colIdx = 0;

        // Extract each integer
        while (ss >> val)
        {
            // Add the current integer to the 'colIdx' column's values vector
            result.at(colIdx).second.push_back(val);

            // If the next token is a comma, ignore it and move on
            if (ss.peek() == ',')
                ss.ignore();

            // Increment the column index
            colIdx++;
        }
    }

    // Close file
    csv_file.close();

    return result;
}

// ------------------------------
int main(int argc, char **argv)
{
    // Make a vector of length 100 filled with 1s
    // std::vector<float> vec(100, 3.1415926f);
    // Write the vector to CSV
    // write_csv("测量结果.csv", "宽度", vec);

    // Make three vectors, each of length 100 filled with 1s, 2s, and 3s
    std::vector<float> vec1(20, 1.23f);
    std::vector<float> vec2(20, 2.45f);
    std::vector<float> vec3(20, 3.14f);

    // Wrap into a vector
    // std::vector<std::pair<std::string, std::vector<float>>> vals = {
    //     {"宽度", vec1},
    //     {"高度", vec2},
    //     {"间距", vec3}
    // };
    std::vector<std::pair<std::string, std::vector<float>>> vals = {
        {  "width", vec1},
        { "height", vec2},
        {"spacing", vec3}
    };

    // Write the vector to CSV
    // const std::string filename2 = "2多列测量结果.csv";
    const std::string filename2 = "2_measureResults.csv";
    write_csv(filename2, vals);
    write_csv(filename2, vals);
    write_csv(filename2, vals);
    write_csv(filename2, vals);
    write_csv(filename2, vals);

    DataResults result{3.12f, 5.34f, 23.1f};

    // const std::string filename3 = "3多列结构体测量结果.csv";
    const std::string filename3 = "3_measureResults.csv";

    //std::vector<std::string> names{ "宽度", "高度", "总宽" };
    std::array<std::string, SIZE_COL> names_arr{"width", "height", "total"};
    write_csv(filename3, names_arr, result);
    write_csv(filename3, names_arr, result);
    write_csv(filename3, names_arr, result);
    write_csv(filename3, names_arr, result);

    std::vector<std::string> names{"width", "height", "total"};
    write_csv(filename3, names, result);
    write_csv(filename3, names, result);
    write_csv(filename3, names, result);
    write_csv(filename3, names, result);

    //-----------------------------------------------
    auto val_result = read_csv(filename2);
    // for (const auto &elem : val_result)
    // TODO there is an litter bug
    for (unsigned long long i = 0; i < val_result[0].second.size(); ++i)
    {
        std::cout << val_result[i].first << ": " << val_result[i].second[i] << "\n";
    }

    return 0;
}
