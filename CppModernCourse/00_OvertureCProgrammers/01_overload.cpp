/**
 * @file 01_overload.cpp
 * @author Wei Li (weili_yzzcq@163.com)
 * @brief 
 * @version 0.1
 * @date 2024-04-15
 * 
 * @copyright Copyright (c) 2024
 * 
 */

/**
 * @brief 将整数类似转换为C语言风格的字符串,
 * C语言中, 每一个函数必须有唯一的名称(函数签名 signature),
 * C++语言中, 参数列表不同时, 函数名可以共享名称, 即函数重载(本质函数签名唯一),
 * C++语言中, 只需要记住一个函数名称即可, 编译器会确定调用具体哪一个函数, 
 * 
 * @param value 
 * @param str 
 * @param base 
 * @return char* 
 */
char *itoa(int value, char *str, int base);
char *Itoa(long value, char *buffer, int base);
char *uItoa(unsigned long value, char *buffer, int base);

char *toa(int value, char *buffer, int base)
{
    return nullptr;
}

char *toa(long value, char *buffer, int base)
{
    return nullptr;
}

char *toa(unsigned long value, char *buffer, int base)
{
    return nullptr;
}

// -----------------------------------
int main(int argc, const char **argv)
{
    char          buff[10];
    int           a = 1;
    long          b = 2;
    unsigned long c = 3;

    toa(a, buff, 10);
    toa(b, buff, 10);
    toa(c, buff, 10);

    return 0;
}
