/* exercise 3-43、3-44、3-45
** 练习3.43: 编写 3 个不同版本的程序，令其均能输出 ia 的元素。
** 版本 1 使用范围for 语句管理迭代过程;
** 版本 2和版本 3都使用普通的for 语句，其中版本2要求用下标运算符，版本3要求用指针
** 此外，在所有 3个版本的程序中都要直接写出数据类型，不能使用类型别名、auto关键字或decltype关键字
** 
** 练习3.44: 改写上一个练习中的程序，使用类型别名来代替循环控制变量的类型
**
** 练习3.45: 再一次改写程序，这次使用auto关键字
*/

#include <iostream>
#include <vector>

// solution 3-43
void solution_43()
{
    int arr[3][4] = 
    { 
        { 0, 1, 2, 3 },
        { 4, 5, 6, 7 },
        { 8, 9, 10, 11 }
    };

    // range for
    for (const int(&row)[4] : arr)
    {
        for (int col : row)
        {
            std::cout << col << " ";
        }
    }
    std::cout << std::endl;

    // for loop with using index
    for (size_t i = 0; i != 3; ++i)
    {
        for (size_t j = 0; j != 4; ++j)
        {
            std::cout << arr[i][j] << " ";
        }
    }
    std::cout << std::endl;

    // for loop with using pointer
    for (int(*row)[4] = arr; row != arr + 3; ++row)
    {
        for (int *col = *row; col != *row + 4; ++col)
        {
            std::cout << *col << " ";
        }
    }
    std::cout << std::endl;
}

void solution_44()
{
    int ia[3][4] = { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11 };

    // for range with using type alias
    using int_array = int[4];
    for (int_array& p : ia)
    {
        for (int q : p)
        {
            std::cout << q << " ";
        }
    }
    std::cout << std::endl;

    // ordinary for loop using subscripts
    for (size_t i = 0; i != 3; ++i)
    {
        for (size_t j = 0; j != 4; ++j)
        {
            std::cout << ia[i][j] << " ";
        }
    }
    std::cout << std::endl;

    // using pointers and using type alias
    for (int_array* p = ia; p != ia + 3; ++p)
    {
        for (int *q = *p; q != *p + 4; ++q)
        {
            std::cout << *q << " ";
        }
    }
    std::cout << std::endl;
}

void solution_45()
{
    int ia[3][4] = { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11 };

    // a range for to manage the iteration with using auto
    for (auto& p : ia)
    {
        for (int q : p)
        {
            std::cout << q << " ";
        }
    }
    std::cout << std::endl;

    // ordinary for loop using subscripts with auto
    for (size_t i = 0; i != 3; ++i)
    {
        for (size_t j = 0; j != 4; ++j)
        {
            std::cout << ia[i][j] << " ";
        }
    }
    std::cout << std::endl;

    // using pointers.
    for (auto p = ia; p != ia + 3; ++p)
    {
        for (int *q = *p; q != *p + 4; ++q)
        {
            std::cout << *q << " ";
        }
    }
    std::cout << std::endl;
}


int main()
{
    // solutin 3-43
    solution_43();

    // solutin 3-44
    solution_44();

    // solutin 3-45
    solution_45();
    
    return 0;
}

/* 编译命令操作流程
** 0、打开终端 terminal，VSCode使用 Ctrl+shift+`; Ubuntu使用 Ctrl+Alt+T
** 1、进入当前源文件路径，cd src; cd chapter03
** 2、编译源代码文件，g++ --version; g++ -o exercise exercise3-43-44-45.cpp
** 3、运行生成的可执行程序，exercise; Ubuntu使用 ./exercise
** 4、删除生成的可执行程序，rm -rf exercise.exe; Ubuntu使用 rm -rf exercise
*/
