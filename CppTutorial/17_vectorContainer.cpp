/**
 * @file 17_vectorContainer.cpp
 * @author Wei Li (weili_yzzcq@163.com)
 * @brief 现代C++编程学习容器之 std::vector
 * @version 0.1
 * @date 2024-03-08
 * 
 * @copyright Copyright (c) 2024
 * 
 */

#include <iostream>
#include <vector>

template<typename T>
void printContainer(const T &container)
{
    for (const auto &elem : container)
    {
        std::cout << elem << ' ';
    }
    std::cout << std::endl;
}

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

// ===================================
int main(int argc, const char **argv)
{
    std::vector<int> vec;

    if (vec.empty())
        for (size_t idx = 0; idx < 12; ++idx)
        {
            vec.push_back(idx + 42);
        }

    std::cout << "============ the int vector elem ============\n";
    printContainer(vec);

    std::vector<Person> vec_person;
    vec_person.push_back({"wei li", 24, true});
    vec_person.push_back({"li wei", 42, false});
    vec_person.push_back({"ming li", 44, true});

    std::cout << "============ the Person array elem ============\n";
    printContainer(vec_person);

    return 0;
}

// ===================================
// compile and link via Clang or GCC
// clang++ .\17_vectorContainer.cpp -std=c++23
// g++ .\17_vectorContainer.cpp -std=c++23
