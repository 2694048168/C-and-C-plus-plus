/* exercise 3-36
** 练习3.36: 编写程序，比较两个数组是否相等
** 再写程序，比较两个 vector 对象是否相等
** solution: size equal and each element equal
*/

#include <iostream>
#include <vector>

// ptr_beg point to begin of the array, ptr_end point to end of the array.
bool compare(int* const ptr_beg1, int* const ptr_end1, int* const ptr_beg2, int* const ptr_end2)
{
    // size equal ?
    if ((ptr_end1 - ptr_beg1) != (ptr_end2 - ptr_beg2))
    {
        return false;
    }
        
    else
    {
        for (int* i = ptr_beg1, *j = ptr_beg2; (i != ptr_end1) && (j != ptr_end2); ++i, ++j)
        {
            // each element equal ?
            if (*i != *j)
            {
                return false;
            }
        }
    }

    return true;
}

int main()
{
    // solutin 3-36
    // array
    int arr1[3] = { 0, 1, 2 };
    int arr2[3] = { 0, 2, 4 };

    if (compare(std::begin(arr1), std::end(arr1), std::begin(arr2), std::end(arr2)))
        std::cout << "The two arrays are equal." << std::endl;
    else
        std::cout << "The two arrays are not equal." << std::endl;

    std::cout << "===============================" << std::endl;

    // vector
    std::vector<int> vec1 = { 0, 1, 2 };
    std::vector<int> vec2 = { 0, 1, 2 };

    if (vec1 == vec2)
        std::cout << "The two vectors are equal." << std::endl;
    else
        std::cout << "The two vectors are not equal." << std::endl;

    return 0;
}

/* 编译命令操作流程
** 0、打开终端 terminal，VSCode使用 Ctrl+shift+`; Ubuntu使用 Ctrl+Alt+T
** 1、进入当前源文件路径，cd src; cd chapter03
** 2、编译源代码文件，g++ --version; g++ -o exercise exercise3-36.cpp
** 3、运行生成的可执行程序，exercise; Ubuntu使用 ./exercise
** 4、随意输入一组数字，例如：1 2 3 4 5 6 7 8 9 ，键入 EOF 即可结束输入
** 5、删除生成的可执行程序，rm -rf exercise.exe; Ubuntu使用 rm -rf exercise
*/
