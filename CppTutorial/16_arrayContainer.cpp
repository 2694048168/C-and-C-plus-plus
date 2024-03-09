/**
 * @file 16_arrayContainer.cpp
 * @author Wei Li (weili_yzzcq@163.com)
 * @brief 现代C++编程学习容器之数组
 * @version 0.1
 * @date 2024-03-08
 * 
 * @copyright Copyright (c) 2024
 * 
 */

#include <array>
#include <cstddef>
#include <iostream>
#include <string>

struct Person
{
    std::string  name;
    unsigned int age;
    bool         gender;

    friend std::ostream &operator<<(std::ostream &output, const Person &person)
    {
        output << "The Name: " << person.name << " Age: " << person.age
               << " and Gender: " << (person.gender ? "male" : "female") << std::endl;
        return output;
    }
};

template<typename T>
void printContainer(const T &container)
{
    for (const auto &elem : container)
    {
        std::cout << elem << ' ';
    }
    std::cout << std::endl;
}

// ===================================
int main(int argc, const char **argv)
{
    const size_t SizeArray = 12;

    std::array<int, SizeArray> arr;

    for (size_t idx = 0; idx < SizeArray; ++idx)
    {
        arr[idx] = idx + SizeArray;
    }

    std::cout << "============ the int array elem ============\n";
    printContainer(arr);

    std::array<Person, 3> arr_person;
    arr_person[0] = {"wei li", 24, true};
    arr_person[1] = {"li wei", 42, false};
    arr_person[2] = {"ming li", 44, true};

    std::cout << "============ the Person array elem ============\n";
    printContainer(arr_person);

    return 0;
}

// ===================================
// compile and link via Clang or GCC
// clang++ .\16_arrayContainer.cpp -std=c++23
// g++ .\16_arrayContainer.cpp -std=c++23
