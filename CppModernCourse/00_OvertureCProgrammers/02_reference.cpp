/**
 * @file 02_reference.cpp
 * @author Wei Li (weili_yzzcq@163.com)
 * @brief 
 * @version 0.1
 * @date 2024-04-15
 * 
 * @copyright Copyright (c) 2024
 * 
 */

#include <iostream>
#include <string>

struct HolmesIV
{
    bool        is_sentient;
    int         sense_of_humor_rating;
    std::string name;
};

/**
 * @brief 指针是C语言的重要特性, 能够通过传递数据地址而不是实际数据来有效处理大量的数据,
 * C++的引用是指针处理的改进, 安全特性更强, 防止空指针解引用和无意的指针再赋值, 
 * reference 使用 & 进行声明; pointer 使用 * 进行声明;
 * reference 使用 . 点运算符; pointer 使用 -> 箭头运算符 与成员进行交互;
 * reference and pointer 实现原理等同, 都是零开销的抽象概念, 编译器生成差不多的指令代码,
 * 编译时期, reference 比 原始pointer 更安全, 不需要进行为空的检查,  
 * reference 保证为非空(不保证有效), 同时不能被重定位(引用初始化后不能指向另一个内存地址),
 * 引用(reference)只是带有额外安全预防措施和语法糖的指针, 当把引用放在等号(assignment)的
 * 左边时, 实际上就是把被引用的值设置为等号右边的值.
 * 
 */
void mannie_service(HolmesIV *mike)
{
    if (nullptr == mike)
        return;

    mike->is_sentient           = true;
    mike->sense_of_humor_rating = 42;
    mike->name                  = "mike";
}

void mannie_service(HolmesIV &mike)
{
    mike.is_sentient           = true;
    mike.sense_of_humor_rating = 42;
    mike.name                  = "mike";
}

// !reference is not nullptr, but not effective
// HolmesIV &not_dinkum()
// {
//     HolmesIV mike;
//     return mike;
// }

// -----------------------------------
int main(int argc, const char **argv)
{
    // ======== reference ========
    int  val     = 42;
    int &val_ref = val;
    int  num     = 100;
    val_ref      = num;

    std::cout << "\n======== reference ========\n";
    std::cout << "val = " << val << " and num = " << num;
    std::cout << " and val_ref = " << val_ref << '\n';

    // ======== pointer ========
    int  a     = 42;
    int *a_ptr = &a;
    int  b     = 100;
    *a_ptr     = b;

    std::cout << "\n======== pointer ========\n";
    std::cout << "a = " << a << " and b = " << b;
    std::cout << " and *a_ptr = " << *a_ptr << '\n';

    return 0;
}
