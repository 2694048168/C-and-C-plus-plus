/**
 * @file 03_auto.cpp
 * @author Wei Li (weili_yzzcq@163.com)
 * @brief 
 * @version 0.1
 * @date 2024-04-15
 * 
 * @copyright Copyright (c) 2024
 * 
 */

#include <cstdlib>
#include <iostream>
#include <typeinfo>

/**
 * @brief C语言经常需要多次重复变量的类型,  
 * C++可以使用 auto 关键字来表达变量的类型, 编译器自动根据变量信息进行推导其类型,
 * 
 * 
 * NOTE: auto 可以添加 const, volatile, & and * 限定符 
 * 
 */
struct HolmesIV
{
    bool is_sentient;
    int  sense_of_humor_rating;
};

HolmesIV *make_mike(int sense_of_humor)
{
    auto mike = (HolmesIV *)malloc(sizeof(HolmesIV));

    mike->is_sentient           = true;
    mike->sense_of_humor_rating = sense_of_humor;
    return mike;
}

// -----------------------------------
int main(int argc, const char **argv)
{
    int  val = 42;
    auto num = 24;
    std::cout << "the type of val = " << typeid(val).name() << '\n';
    std::cout << "the type of num = " << typeid(num).name() << '\n';

    // ========= auto 关键字便于阅读, 修改类型后容易进行代码调整
    auto mike = make_mike(42);
    std::cout << "mike sense of humor  = " << mike->sense_of_humor_rating << '\n';

    if (nullptr != mike)
    {
        free(mike);
        mike = nullptr;
    }

    return 0;
}
