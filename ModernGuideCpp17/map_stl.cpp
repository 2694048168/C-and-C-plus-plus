/**
 * @file map_stl.cpp
 * @author Wei Li (weili_yzzcq@163.com)
 * @brief 
 * @version 0.1
 * @date 2023-05-27
 * 
 * @copyright Copyright (c) 2023
 * 
 */

#include <iostream>
#include <map>
#include <string>
#include <utility>

template<typename T>
bool print_container(const T &container, const char *msg)
{
    if (container.empty())
    {
        std::cout << "the container is empty, please check.\n" << std::endl;
        return false;
    }

    std::printf("%s: ", msg);
    for (const auto &[key, value] : container)
    {
        std::cout << "{" << key << ", " << value << "} ";
    }
    std::printf("\n");

    return true;
}

/**
 * @brief Maps and multimap are associative containers in C++ STL.
 * 
 * @param argc 
 * @param argv 
 * @return int 
 */
int main(int argc, const char **argv)
{
    // -------------------------------------
    std::map<int, std::string> map_1;
    map_1[0] = "map";
    map_1[2] = "two";
    map_1[1] = "one";
    map_1[3] = "name";

    std::map<int, std::string>::iterator iter = map_1.begin();
    while (iter != map_1.end())
    {
        std::cout << "key: " << iter->first << ", value: " << iter->second << " ";
        ++iter;
    }
    std::cout << "\n";

    // C++17 supported the bind operator
    for (const auto &[key, value] : map_1)
    {
        std::cout << "(key: " << key << ", value: " << value << ") ";
    }
    std::cout << "\n";

    std::cout << "the size of map: " << map_1.size() << std::endl;

    // -------------------------------------
    std::cout << "-------------------------" << std::endl;
    std::map<int, int> map_2;
    map_2.insert(std::pair<int, int>(24, 42));
    map_2.insert(std::pair<int, int>(22, 12));
    map_2.insert(std::pair<int, int>(21, 22));
    map_2.insert(std::pair<int, int>(14, 62));
    map_2[3] = 78;

    print_container(map_2, "the elements of map: ");

    if (map_1.count(2) > 0)
    {
        std::cout << "Key '2' is in the map" << std::endl;
    }
    else
    {
        std::cout << "Key '2' is not in the map" << std::endl;
    }
    std::cout << "Key: 1, Value: " << map_1[1] << std::endl;
    std::cout << "Key: 0, Value: " << map_1[0] << std::endl;

    std::cout << "map.at function: " << map_2.at(22) << std::endl;

    // -------------------------------------
    std::cout << "-------------------------" << std::endl;
    std::map<std::string, int> map_3;
    map_3["one"]   = 1;
    map_3["two"]   = 2;
    map_3["three"] = 3;

    if (map_3.find("two") != map_3.end())
    {
        std::cout << "the key 'two' found in the map." << std::endl;
    }
    else
    {
        std::cout << "the key 'two' NOT found in the map." << std::endl;
    }

    if (map_3.find("four") != map_3.end())
    {
        std::cout << "the key 'four' found in the map." << std::endl;
    }
    else
    {
        std::cout << "the key 'four' NOT found in the map." << std::endl;
    }

    // --------------- multimap ---------------
    std::cout << "-------------------------" << std::endl;
    std::multimap<std::string, int> map_4;
    map_4.insert(std::pair<std::string, int>("one", 1));
    map_4.insert(std::pair<std::string, int>("three", 3));
    map_4.insert(std::pair<std::string, int>("one", 11));
    map_4.insert(std::pair<std::string, int>("two", 2));
    map_4.insert(std::pair<std::string, int>("one", 1));
    map_4.insert(std::pair<std::string, int>("three", 33));

    print_container(map_4, "the elements of multimap");

    return 0;
}