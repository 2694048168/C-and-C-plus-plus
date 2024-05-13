/**
 * @file 02_streamManipulators.cpp
 * @author Wei Li (weili_yzzcq@163.com)
 * @brief 
 * @version 0.1
 * @date 2024-05-13
 * 
 * @copyright Copyright (c) 2024
 * 
 */

#include <deque>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

/**
 * @brief Manipulators 操纵符
 * 操纵符(manipulator)是修改流解释输入或格式化输出方式的特殊对象, 操纵符的存在是为了执行多种流更改.
 * 1. std::ws 修改 istream 以跳过空格;
 * 2. std::flush 将任何缓冲的输出直接清空到 ostream;
 * 3. std::ends 发送一个空字节到 ostream;
 * 4. std::endl 就像 std::flush 一样, 但是它会在刷新之前发送一个换行符;
 * 
 * ?stdlib 在 ＜ios＞ 头文件中提供了许多其他操纵符
 * 
 * ====用户自定义类型 User-Defined Types
 * *通过实现某些非成员函数使用户自定义类型与流一起使用
 */
template<typename T>
std::ostream &operator<<(std::ostream &os, std::vector<T> vec)
{
    os << "Size: " << vec.size() << "\nCapacity: " << vec.capacity() << "\nElements:\n";
    for (const auto &element : vec)
    {
        os << "\t" << element << "\n";
    }
    return os;
}

template<typename T>
std::istream &operator>>(std::istream &is, std::deque<T> &t)
{
    T element;
    while (is >> element)
    {
        t.emplace_back(std::move(element));
    }
    return is;
}

// -----------------------------------
int main(int argc, const char **argv)
{
    // 可以确定 ostream 是用文本(boolalpha)还是数字(noboolalpha)表示布尔值;
    // 是使用八进制(oct)、十进制(dec)还是十六进制(hex)表示整数值;
    // 是将浮点数表示为十进制记数法(fixed), 还是科学记数法(scientific);
    // *只需使用 operator＜＜ 将这些操纵符中的一个传递给ostream, 所有后续插入的都将以该操纵符类型输出
    // 使用 setw 操纵符设置流的宽度参数;
    // 对于浮点输出, setprecision 将设置数字的精度
    // ＜iomanip＞ 头文件中可用的操纵符的程序, 操纵符如何执行类似于各种 printf 格式说明符的功能
    std::cout << "Gotham needs its " << std::boolalpha << true << " hero.";
    std::cout << "\nMark it " << std::noboolalpha << false << "!";

    std::cout << "\nThere are " << 69 << "," << std::oct << 105 << " leaves in here.";
    std::cout << "\nYabba " << std::hex << 3669732608 << "!";

    std::cout << "\nAvogadro's number: " << std::scientific << 6.0221415e-23;
    std::cout << "\nthe Hogwarts platform: " << std::fixed << std::setprecision(2) << 9.750123;
    std::cout << "\nAlways eliminate " << 3735929054;

    std::cout << std::setw(4) << "\n" << 0x1 << "\n" << 0x10 << "\n" << 0x100 << "\n" << 0x1000 << std::endl;

    // TODO: https://en.cppreference.com/w/cpp/io/manip

    const std::vector<std::string> characters{"Bobby Shaftoe", "Lawrence Waterhouse", "Gunter Bischoff",
                                              "Earl Comstock"};

    std::cout << characters << std::endl;

    const std::vector<bool> bits{true, false, true, false};
    std::cout << std::boolalpha << bits << std::endl;

    std::cout << "Give me numbers: ";
    std::deque<int> numbers;
    std::cin >> numbers;
    int sum{};
    std::cout << "Cumulative sum:\n";
    for (const auto &element : numbers)
    {
        sum += element;
        std::cout << sum << "\n";
    }

    return 0;
}
