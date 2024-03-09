/**
 * @file 19_mapContainer.cpp
 * @author Wei Li (weili_yzzcq@163.com)
 * @brief 现代C++编程学习容器之 std::map
 * @version 0.1
 * @date 2024-03-09
 * 
 * @copyright Copyright (c) 2024
 * 
 */

#include <iostream>
#include <map>
#include <string>

struct CameraInfo
{
    std::string cameraName;
    std::string serialNumber;
};

// ===================================
int main(int argc, const char **argv)
{
    std::map<std::string, std::string> cameraMap;

    CameraInfo camera0 = {"CCD0", "A78239"};
    CameraInfo camera1 = {"CCD1", "A12236"};
    CameraInfo camera2 = {"CCD2", "A78569"};
    CameraInfo camera3 = {"CCD3", "A78134"};

    cameraMap[camera0.cameraName] = camera0.serialNumber;
    cameraMap[camera1.cameraName] = camera1.serialNumber;
    cameraMap[camera2.cameraName] = camera2.serialNumber;
    cameraMap[camera3.cameraName] = camera3.serialNumber;

    // 遍历一个map
    for (const auto &elem : cameraMap)
    {
        std::cout << "The CCD name: " << elem.first;
        std::cout << " and the Serial Number: " << elem.second << std::endl;
    }

    // 根据 key, 判断是否存在
    std::string find_key = "CCD0";
    // std::string find_key = "CCD";
    auto        iter = cameraMap.find(find_key);
    if (cameraMap.end() == iter)
        std::cout << "the key: " << find_key << " is NOT in the map\n";
    else
    {
        std::cout << "the key: " << find_key << " is in the map\n";

        std::cout << "The CCD name: " << iter->first;
        std::cout << " and the Serial Number: " << iter->second << std::endl;
    }

    // 根据指定 key 删除 map 中元素
    // 迭代器刪除
    iter = cameraMap.find("CCD0");
    cameraMap.erase(iter);

    // 用关键字刪除
    std::string key = "CCD2";
    int         res = cameraMap.erase(key); //如果刪除了會返回1，否則返回0
    if (0 == res)
        std::cout << "Map erase " << key << " is NOT successfully\n";
    else
        std::cout << "Map erase " << key << " is successfully\n";

    // 用迭代器范围刪除 : 把整个map清空
    // cameraMap.erase(cameraMap.begin(), cameraMap.end());
    cameraMap.clear();
    if (cameraMap.empty())
        std::cout << "Map clear " << key << " is successfully\n";
    else
        std::cout << "Map erase " << key << " is NOT successfully\n";

    return 0;
}

// ===================================
// compile and link via Clang or GCC
// clang++ .\19_mapContainer.cpp -std=c++23
// g++ .\19_mapContainer.cpp -std=c++23
