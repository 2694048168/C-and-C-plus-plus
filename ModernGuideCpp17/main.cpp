/**
 * @file main.cpp
 * @author weili (weili_yzzcq@163.com)
 * @brief
 * @version 0.1
 * @date 2023-05-22
 *
 * @copyright Copyright (c) 2023
 *
 */

#include <iostream>
#include <map>
#include <string>
#include <string_view>

struct MyStruct
{
    int         num = 0;
    std::string msg{};
};

MyStruct getStruct()
{
    return MyStruct{42, "hello"};
}

// void print_map(std::string_view comment, const std::map<std::string, int> &m)
void print_map(const char *comment, const std::map<std::string, int> &m)
{
    std::cout << comment;
    // iterate using C++17 facilities
    for (const auto &[key, value] : m)
    {
        std::cout << '[' << key << "] = " << value << "; ";
    }

    /*
    // C++11 alternative:
    for (const auto &n : m)
    {
        std::cout << n.first << " = " << n.second << "; ";
    }

    // C++98 alternative
    for (std::map<std::string, int>::const_iterator it = m.begin(); it != m.end(); it++)
    {
        std::cout << it->first << " = " << it->second << "; ";
    }
    */

    std::cout << '\n';
}

/**
 * @brief main function
 * 
 * @param argc 
 * @param argv 
 * @return int 
 */
int main(int argc, const char **argv)
{
    // C++11引入的统一初始化(uniform initialization)
    // 列表初始化(list initialization)的初始化方式有优势
    int i{42};

    std::string s{"hello cpp"};

    int   num{};
    float numf{};
    bool  initbool{};
    int  *ptr{};
    // std::cout << "init value default: " << num << "\n";
    // std::cout << "init value default: " << numf << "\n";
    // std::cout << "init value default: " << initbool << "\n";
    // std::cout << "init value default: " << ptr << "\n";

    // C++17 结构化绑定
    MyStruct ms;
    auto [u, v] = ms;
    // std::cout << "u and v: " << u << ", " << v << std::endl;
    // auto [u, v] {ms};

    auto [id, val] = getStruct();
    if (id > 30)
    {
        std::cout << "the id value: " << id << std::endl;
    }
    std::cout << "the value: " << val << std::endl;

    // Create a map of three (string, int) pairs
    std::map<std::string, int> m{
        {"CPU", 10},
        {"GPU", 15},
        {"RAM", 20}
    };

    print_map("1) Initial map: ", m);

    m["CPU"] = 25; // update an existing value
    m["SSD"] = 30; // insert a new value
    print_map("2) Updated map: ", m);

    // using operator[] with non-existent key always performs an insert
    std::cout << "3) m[UPS] = " << m["UPS"] << '\n';
    print_map("4) Updated map: ", m);

    m.clear();
    std::cout << std::boolalpha << "8) Map is empty: " << m.empty() << '\n';

    return 0;
}