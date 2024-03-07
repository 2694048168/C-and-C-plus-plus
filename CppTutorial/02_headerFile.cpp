#include "02_headerFile.hpp"

int addNumber(int num1, int num2)
{
    return num1 + num2;
}

int subNumber(int num1, int num2)
{
    return num1 - num2;
}

int mulNumber(int num1, int num2)
{
    return num1 * num2;
}

int divNumber(int num1, int num2)
{
    if (0 == num2)
        return 0;
    return num1 / num2;
}
