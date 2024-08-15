#include "UniversallyUniqueIdentifier.hpp"

#include <chrono>
#include <iomanip>
#include <random>
#include <sstream>

int getUUID_random()
{
    std::random_device rd;

    std::mt19937 gen(rd());

    std::uniform_int_distribution<> dis(10000, 99999); // 定义随机数范围

    // 生成一个随机数
    return dis(gen);
}

// int UniqueIDGenerator::counter = 0;
std::atomic<int> UniqueIDGenerator::counter = 0;

std::string UniqueIDGenerator::generate()
{
    // 获取当前时间戳
    auto now = std::chrono::system_clock::now();

    auto now_c = std::chrono::system_clock::to_time_t(now);

    std::stringstream ss;
    ss << std::put_time(std::localtime(&now_c), "%Y%m%d%H%M%S");

    // 增加计数器
    counter++;

    // 组合时间戳和计数器生成唯一值
    ss << std::setw(3) << std::setfill('0') << counter;
    return ss.str();
}
