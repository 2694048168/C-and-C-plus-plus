/**
 * @file 7_13_7_array_function.cpp
 * @author Wei Li (weili_yzzcq@163.com)
 * @brief 
 * @version 0.1
 * @date 2023-08-03
 * 
 * @copyright Copyright (c) 2023
 * 
 */

#include <iostream>

const unsigned int Max_Size = 5;

// function prototypes or signature
// int  fill_array(double ar[], int limit);
/**1. 指针函数：带指针的函数，即本质是一个函数。函数返回类型是某一类型的指针:
 *    类型标识符 *函数名(参数表) ----> int *f(x，y);
 * 2. 函数指针：指向函数(首地址)的指针变量，即本质是一个指针变量。
 *  函数指针说的就是一个指针，但这个指针指向的函数，不是普通的基本数据类型或者类对象。
 *  指向函数的指针包含了函数的地址，可以通过它来调用函数。
 *  声明格式：类型说明符 (*函数名)(参数) ----> int (*func)(int a, int b);
 *  其实这里不能称为函数名，应该叫做指针的变量名。这个特殊的指针指向一个返回整型值的函数。
 *  指针的声明必须和它指向函数的声明保持一致。指针名和指针运算符外面的括号改变了默认的运算符优先级。
 *  如果没有圆括号，就变成了一个返回整型指针的函数的原型声明。
 *
 * 3. 函数指针的返回值也可以是指针。
---------------------------------------------------------------- */
double *fill_array_ptr(double *start_array, double *end_array);

// void show_array(const double ar[], int n);
void show_array_ptr(const double *start_array, const double *end_array);

// void revalue(double r, double ar[], int n);
void revalue_ptr(const double r, double *start_array, double *end_array);

/**
 * @brief 编写C++程序, 利用函数封装对数组的操作
 * 
 * @param argc 
 * @param argv 
 * @return int 
 */
int main(int argc, const char **argv)
{
    double properties[Max_Size];

    // int size = fill_array(properties, Max_Size);
    double *last_position = fill_array_ptr(properties, &properties[Max_Size]);

    // show_array(properties, size);
    show_array_ptr(properties, last_position);

    // if (last_position != nullptr)
    if (last_position)
    {
        std::cout << "Enter revaluation factor: ";
        double factor;
        while (!(std::cin >> factor)) // bad input
        {
            std::cin.clear();
            while (std::cin.get() != '\n') continue;
            std::cout << "Bad input; Please enter a number: ";
        }

        // revalue(factor, properties, size);
        revalue_ptr(factor, properties, last_position);
        // show_array(properties, size);
        show_array_ptr(properties, last_position);
    }
    std::cout << "Done.\n";
    std::cin.get();
    std::cin.get();

    return 0;
}

int fill_array(double ar[], int limit)
{
    using namespace std;
    double temp;
    int    i;
    for (i = 0; i < limit; i++)
    {
        cout << "Enter value #" << (i + 1) << ": ";
        cin >> temp;
        if (!cin) // bad input
        {
            cin.clear();
            while (cin.get() != '\n') continue;
            cout << "Bad input; input process terminated.\n";
            break;
        }
        else if (temp < 0) // signal to terminate
            break;
        ar[i] = temp;
    }
    return i;
}

double *fill_array_ptr(double *start_array, double *end_array)
{
    unsigned idx = 1;
    double   temp;
    while (start_array != end_array)
    {
        std::cout << "Enter value #" << idx << ": ";
        std::cin >> temp;

        if (!std::cin)
        {
            std::cin.clear();
            while (std::cin.get() != '\n') continue;
            std::cout << "Bad input; input process terminated.\n";
            break;
        }
        else if (temp < 0) // signal to terminate
        {
            break;
        }
        else
        {
            *start_array = temp;
        }

        ++start_array;
        ++idx;
    }

    return start_array;
    // return end_array;
}

// the following function can use, but not alter,
// the array whose address is ar
void show_array(const double ar[], int n)
{
    using namespace std;
    for (int i = 0; i < n; i++)
    {
        cout << "Property #" << (i + 1) << ": $";
        cout << ar[i] << endl;
    }
}

void show_array_ptr(const double *start_array, const double *end_array)
{
    unsigned idx = 1;
    while (start_array != end_array)
    {
        std::cout << "Property #" << idx << ": $" << *start_array << "\n";

        ++start_array;
        ++idx;
    }
}

// multiplies each element of ar[] by r
void revalue(double r, double ar[], int n)
{
    for (int i = 0; i < n; i++) ar[i] *= r;
}

// multiplies each element of ar[] by r
void revalue_ptr(const double r, double *start_array, double *end_array)
{
    while (start_array != end_array)
    {
        (*start_array) *= r;
        ++start_array;
    }
}