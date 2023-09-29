/**
 * @file cplusplus_macros.c
 * @author Wei Li (weili_yzzcq@163.com)
 * @brief 
 * @version 0.1
 * @date 2023-09-29
 * 
 * @copyright Copyright (c) 2023
 * 
 * clang cplusplus_macros.c -std=c18
 * clang++ cplusplus_macros.c -std=c++17
 * 
 */

#ifdef __cplusplus 
    #include <iostream>
#else
    #include <stdio.h>
#endif // __cplusplus


// 仔细想一想, C语言标准和C++语言标准, 针对变量/函数的命名的修饰
// C++ 新增了命名空间和函数重载功能的缘故, 需要对函数签名进行一定规则的修饰
#ifdef __cplusplus
extern "C" {
#endif    

void add(int val1, int val2)
{
    printf("the sum of two integer: %d", (val1 + val2));
}

#ifdef __cplusplus
}
#endif

/**
 * @brief Standard Predefined Macros
 * 
 * @param argc 
 * @param argv 
 * @return int 
 */
int main(int argc, const char** argv) 
{
#ifdef __cplusplus
    std::cout << "Hello C++ world\n";
#else
    printf("Hello C world\n");
#endif // __cplusplus

    int val1 = 42;
    int val2 = 24;
    add(val1, val2);

    return 0;
}
