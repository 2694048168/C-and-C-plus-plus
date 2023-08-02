/**
 * @file 7_13_3_parameter_pass.cpp
 * @author Wei Li (weili_yzzcq@163.com)
 * @brief 
 * @version 0.1
 * @date 2023-08-02
 * 
 * @copyright Copyright (c) 2023
 * 
 */

#include <iostream>

struct Box
{
    char  maker[40]{"wei li"};
    float height{0.f};
    float width{0.f};
    float length{0.f};
    float volume{0.f};
};

void print_info(const Box box);
// void print_info(const Box &box);
// void print_info(const Box *box);

// void set_volume(Box box);
// void set_volume(Box &box);
void set_volume(Box *box);

/**
 * @brief 编写C++程序, 将结构体进行参数传递: pass by value/address/reference
 * 
 * @param argc 
 * @param argv 
 * @return int 
 */
int main(int argc, const char **argv)
{
    Box box{"wei li", 12.3f, 23.4f, 21.0f};

    print_info(box);

    set_volume(&box);

    print_info(box);

    return 0;
}

void print_info(const Box box)
{
    std::cout << "======== The information of Box structure ========\n";
    std::cout << box.height << " ";
    std::cout << box.width << " ";
    std::cout << box.length << " ";
    std::cout << box.volume << "\n";
    std::cout << "==================================================\n";
}

void set_volume(Box *box)
{
    box->volume = box->height * box->width * box->length;
}
