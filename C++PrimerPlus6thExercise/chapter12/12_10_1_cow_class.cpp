/**
 * @file 12_10_1_cow_class.cpp
 * @author Wei Li (weili_yzzcq@163.com)
 * @brief 
 * @version 0.1
 * @date 2023-08-10
 * 
 * @copyright Copyright (c) 2023
 * 
 */

// #include <cstring>
#include <iostream>
#include <string>

class Cow
{
private:
    // char   name[20] = "";
    // char   name[20];
    std::string name;
    // char  *hobby    = nullptr;
    // char  *hobby;
    std::string hobby;
    // double weight   = 0.0;
    double      weight;

public:
    Cow() = delete;

    // Cow() = default;

    Cow(const char *nm, const char *ho, double wt)
    {
        // strcpy_s(name, strlen(nm) + 1, nm);
        // strcpy_s(hobby, strlen(ho) + 1, ho);

        name   = nm;
        hobby  = ho;
        weight = wt;
    }

    // constructor
    Cow(const Cow &c)
    {
        // strcpy_s(this->name, strlen(c.name) + 1, c.name);
        // strcpy_s(this->hobby, strlen(c.hobby) + 1, c.hobby);

        this->name   = c.name;
        this->hobby  = c.hobby;
        this->weight = c.weight;
    }

    ~Cow()
    {
        std::cout << "==== this is deconstructor.\n";
    }

    // Copy assignment function(shallow copy or deep copy)
    Cow &operator=(const Cow &c)
    {
        // strcpy_s(this->name, strlen(c.name) + 1, c.name);
        // strcpy_s(this->hobby, strlen(c.hobby) + 1, c.hobby);

        this->name   = c.name;
        this->hobby  = c.hobby;
        this->weight = c.weight;

        return *this;
    }

    void ShowCow() const // display all cow data
    {
        std::cout << "The name of Cow: " << name << "\n";
        std::cout << "The hobby of Cow: " << hobby << "\n";
        std::cout << "The weight of Cow: " << weight << "\n";
        std::cout << "======================================\n";
    }
};

/**
 * @brief 编写C++程序, 声明并实现类, 并对类的使用做简单程序演示
 * 使用 C-style 字符串需要考虑内存拷贝和非法操作等一系列问题
 * 建议使用封装好的 std::string 类, 简单便捷安全
 * @param argc 
 * @param argv 
 * @return int 
 */
int main(int argc, const char **argv)
{
    Cow obj1_cow("", "", 0.0);
    obj1_cow.ShowCow();

    Cow obj2_cow("wei li", "reading", 66);
    obj2_cow.ShowCow();

    Cow obj3_cow(obj2_cow);
    obj3_cow.ShowCow();

    obj1_cow = obj3_cow;
    obj1_cow.ShowCow();

    return 0;
}