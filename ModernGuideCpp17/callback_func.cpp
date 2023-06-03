/**
 * @file callback_func.cpp
 * @author Wei Li (weili_yzzcq@163.com)
 * @brief 
 * @version 0.1
 * @date 2023-06-03
 * 
 * @copyright Copyright (c) 2023
 * 
 */

#include <iostream>
#include <string>

// 返回值是右值
int add_integer(int var1, int var2)
{
    return var1 + var2;
}

char encrypt(const char &param)
{
    return param - 13;
}

char decrypt(const char &param)
{
    return param + 13;
}

void modify(std::string &str, char (*callback_func)(const char &))
{
    for (size_t i = 0; i < str.size(); ++i)
    {
        str[i] = callback_func(str[i]);
    }
}

/**
 * @brief 函数指针 | 左值 | 右值
 * 
 * - 函数只是存在于程序的内存映射中某处的一段代码
 * - 当然可以获取函数的地址将其存储在函数指针中
 * - 通过函数指针可以完成很多操作，如回调(callback)
 * 
 * - 左值，左边的值
 *      - 左值 是可以获取地址并在以后使用的
 *      - 左值 是可以获取地址, 不需要进行 value copy
 * - 右值，右边的值
 *      - 右值 本质上是临时或暂时的值,无法获取地址
 *      - 右值 无法获取地址, 需要进行 value copy
 *      - 右值 用 && 方式进行存储, 避免进行 value copy, 性能好
 * 
 * @param argc 
 * @param argv 
 * @return int 
 */
int main(int argc, const char **argv)
{
    // -------------------------------
    std::cout << "-------------------------------\n";
    std::string msg{"weili"};
    std::cout << "the original: " << msg << "\n";

    // char - 13 to encrypt
    modify(msg, encrypt);
    std::cout << "the encrypt: " << msg << "\n";

    // char + 13 to decrypt
    modify(msg, decrypt);
    std::cout << "the decrypt: " << msg << "\n";

    std::cout << "-------------------------------" << std::endl;
    // -------------------------------
    int var1{1};
    int var2{2};
    int var3{3};
    // 都是左值
    std::cout << "the address: " << &var1 << "\n";
    std::cout << "the address: " << &var2 << "\n";
    std::cout << "the address: " << &var3 << "\n";

    int *ptr{&var3};
    std::cout << "the value: " << *ptr << "\n";

    int var4 = (var1 + var2); /* (var1 + var2) 地址不能获取(右值), value copy */
    // int *ptr_ = &(var1 + var2);
    std::cout << "the result: " << var4 << "\n";

    int &&result = (var1 + var2); /* 避免 value copy */
    std::cout << "the result: " << result << "\n";

    int result2 = add_integer(var3, var2); /* value copy */
    std::cout << "the result: " << result2 << "\n";

    int &&result3 = add_integer(var3, var2); /* move? */
    std::cout << "the result: " << result3 << "\n";

    std::cout << "-------------------------------" << std::endl;

    return 0;
}