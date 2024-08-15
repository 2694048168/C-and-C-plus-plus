#include "UniversallyUniqueIdentifier.hpp"

#include <cstddef>
#include <iostream>
#include <thread>
#include <vector>

// =====================================
int main(int argc, const char **argv)
{
    std::vector<std::thread> thread_vec;

    auto generate_UUID = []()
    {
        std::cout << "========================\n";
        int random_UUID = getUUID_random();
        std::cout << "The UUID via random: " << random_UUID << '\n';

        std::string timestamp_UUID = UniqueIDGenerator::generate();
        std::cout << "The UUID via timestamp: " << timestamp_UUID << '\n';
        std::cout << "========================\n";
    };

    for (size_t idx{0}; idx < std::thread::hardware_concurrency(); ++idx)
    {
        thread_vec.push_back(std::thread(generate_UUID));
    }

    for (auto &task : thread_vec)
    {
        if (task.joinable())
            task.join();
    }

    return 0;
}
