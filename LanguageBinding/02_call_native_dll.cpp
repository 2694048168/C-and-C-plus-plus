/**
 * @file 02_call_native_dll.cpp
 * @author Wei Li (weili_yzzcq@163.com)
 * @brief 
 * @version 0.1
 * @date 2024-10-05
 * 
 * @copyright Copyright (c) 2024
 * 
 */

#include <stdarg.h>

#define DLL_IMPORT
#include "NativeDll/Native.h"
#pragma comment(lib, "./bin/NativeDll.lib")

void test(int length, ...)
{
    va_list ap;
    va_start(ap, length);

    int    num1 = va_arg(ap, int);
    int    num2 = va_arg(ap, int);
    double num3 = va_arg(ap, double);
    va_end(ap);
}

// =================================
int main(int argc, char *argv[])
{
    test(3, 123, 456, 12.3);

    return 0;
}
