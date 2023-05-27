/**
 * @file vector_stl.cpp
 * @author Wei Li (weili_yzzcq@163.com)
 * @brief 
 * @version 0.1
 * @date 2023-05-26
 * 
 * @copyright Copyright (c) 2023
 * 
 * the 'const' and 'reverse' to iter in vector container.
 * 
 */

#include <cstdio>
#include <iostream>
#include <vector>

template<typename T>
bool print_container(const T &container, const char* msg)
{
    if (container.empty())
    {
        std::cout << "the container is empty, please check.\n" << std::endl;
        return false;
    }

    std::printf("%s: ", msg);
    for (const auto elem : container)
    {
        std::cout << elem << " ";
    }
    std::printf("\n");

    return true;
}

/**
 * @brief the vector container in C++ STL.
 * 
 * @param argc 
 * @param argv 
 * @return int 
 */
int main(int argc, const char **argv)
{
    // -------------------------------
    std::vector<int> vec1;
    for (size_t i = 1; i < 6; ++i)
    {
        vec1.push_back(i);
    }

    std::printf("from cbegin to cend: ");
    for (auto iter = vec1.cbegin(); iter != vec1.cend(); ++iter)
    {
        std::cout << *iter << " ";
    }

    std::printf("\nfrom crbegin to crend: ");
    for (auto iter = vec1.crbegin(); iter != vec1.crend(); ++iter)
    {
        std::cout << *iter << " ";
    }

    // -------------------------------
    std::cout << "\n\nSize : " << vec1.size();
    std::cout << "\nCapacity : " << vec1.capacity();
    std::cout << "\nMax_Size : " << vec1.max_size();

    vec1.resize(4);
    std::cout << "\nSize(after resize) : " << vec1.size();
    std::cout << "\nCapacity(after resize) : " << vec1.capacity();

    if (vec1.empty() == false)
        std::printf("\nVector is not empty");
    else
        std::printf("\nVector is empty");

    vec1.shrink_to_fit();
    std::cout << "\nCapacity(after shrink_to_fit) : " << vec1.capacity();
    std::cout << "\nVector elements are: ";
    for (auto it = vec1.cbegin(); it != vec1.cend(); ++it)
    {
        std::cout << *it << " ";
    }

    // large-scale data for elements in vector via reserve 主动分配内存
    std::vector<int> vec2;
    vec2.reserve(10000);
    for (size_t i = 1; i < 100; ++i)
    {
        vec2.push_back(i);
    }
    std::cout << "\n\nSize : " << vec2.size();
    std::cout << "\nCapacity : " << vec2.capacity();

    // -------------------------------
    std::vector<int> vec3;
    for (unsigned int i = 0; i < 10; ++i)
    {
        vec3.push_back(i);
    }

    std::cout << "\n\nReference operator [position] : vec3[2] = " << vec3[2];
    std::cout << "\nat : vec3.at(4) = " << vec3.at(4);
    std::cout << "\nfront() : vec3.front() = " << vec3.front();
    std::cout << "\nback() : vec3.back() = " << vec3.back();

    // pointer to the first element
    int *pos = vec3.data();
    std::cout << "\nThe first element is " << *pos;
    std::cout << "\nThe second element is " << *(pos + 1);
    std::cout << "\nThe second element is " << *(++pos);

    // -------------------------------
    std::vector<int> vec4;
    vec4.assign(5, 42);

    auto print = [&vec4](std::string_view const msg)
    {
        std::cout << msg << ":";
        for (const auto elem : vec4)
        {
            std::cout << elem << " ";
        }
    };
    print("\n\nthe element of vec4");

    vec4.push_back(15);
    print("\nafter push_back of vector");

    vec4.pop_back();
    print("\nafter pop_back of vector");

    vec4.insert(vec4.begin(), 5);
    print("\nafter insert at head of vector");
    vec4.insert(vec4.end(), 5);
    print("\nafter insert at end of vector");

    vec4.erase(vec4.begin());
    print("\nafter erase at head of vector");

    vec4.emplace(vec4.cbegin(), 66);
    print("\nafter emplace at head of vector");
    vec4.emplace_back(66);
    print("\nafter emplace at end of vector");

    vec4.clear();
    print("\nafter clear of vector");
    std::cout << "\nand the size is: " << vec4.size();

    // -------------------------------
    // two vector to perform swap
    std::vector<int> v1, v2;
    v1.push_back(1);
    v1.push_back(2);
    v2.push_back(3);
    v2.push_back(4);

    std::printf("\n\nbefore swap\n");
    print_container(v1, "vector 1");
    print_container(v2, "vector 2");

    v1.swap(v2);

    std::printf("\nafter swap\n");
    print_container(v1, "vector 1");
    print_container(v2, "vector 2");

    return 0;
}