/**
 * @file nested_class.cpp
 * @author Wei Li (weili_yzzcq@163.com)
 * @date 2023-05-14
 * @version 0.1.1
 *
 * @copyright Copyright (c) 2023
 *
 * @brief Nested type and Nested Classes
 * @attention
 *
 */

#include <iostream>
#include <string>

class Storage
{
public:
    class Fruit
    {
        std::string name;
        int weight;

    public:
        Fruit(std::string name = "", int weight = 0) : name(name), weight(weight) {}
        std::string getInfo() { return name + ", weight " + std::to_string(weight) + "kg."; }
    };

private:
    Fruit fruit;

public:
    Storage(Fruit f)
    {
        this->fruit = f;
    }
    void print()
    {
        std::cout << fruit.getInfo() << std::endl;
    }
};

/**
 * @brief main function
 */
int main(int argc, char const *argv[])
{
    Storage::Fruit apple("apple", 100);
    Storage mystorage(apple);
    mystorage.print();

    return 0;
}

/** Build(compile and link) commands via command-line.
 *
 * $ clang++ nested_class.cpp
 * $ clang++ nested_class.cpp -std=c++17
 * $ ./a.exe # on Windows
 * $ ./a.out # on Linux or Mac
 *
 */