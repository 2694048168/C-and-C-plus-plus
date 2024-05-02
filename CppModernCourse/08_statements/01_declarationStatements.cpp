/**
 * @file 01_declarationStatements.cpp
 * @author Wei Li (weili_yzzcq@163.com)
 * @brief 
 * @version 0.1
 * @date 2024-05-02
 * 
 * @copyright Copyright (c) 2024
 * 
 */
#include <cstdint>
#include <cstdio>

/**
 * @brief Declaration Statements
 * 声明语句(声明)将标识符(例如函数,模板和命名空间)导入程序
 * *表达式 static_assert 也是一条声明语句
 * 
 * 1=====function
 * function declaration or signature or prototype
 * 函数声明(也称为函数的签名或原型)指定函数的输入和输出,声明不需要包含参数名称,只需包含参数类型
 * *编译器工具链, 链接符号
 * 
 * *声明和定义分离, 头文件和源文件分离
 * multiple-source-file programs where separating 
 * declaration and definition provides major benefits.
 * 
 * 2====Namespaces
 * Namespaces prevent naming conflicts.
 * 命名空间可防止命名冲突, 在大型项目中或者在引入库时,
 * *命名空间对于准确消除要查找的符号的歧义是必不可少的.
 * 命名空间可以使用两种方式嵌套:简单地嵌套 | 作用域解析运算符
 * 1. namespace Name1{ namespace Name2}
 * 2. namespace Name1::Name2{}
 * 使用作用域解析运算符来指定符号的完全限定名称
 * 1. 使用第三方库,防止命名冲突
 * 2. 避免歧义
 * ---- using 指定符(Using Directives)
 * !永远不要在头文件中放置 using namespace 指定符;
 *  对于每个包含头文件的源文件 using 指定符都会把所有符号引入全局命名空间中,
 * 这可能会导致难以调试的问题.
 * 
 */

// randomize 实现是线性同余生成器, 是一种原始的随机数生成器
void randomize(uint32_t &x); // function signature

struct RandomNumberGenerator
{
    explicit RandomNumberGenerator(uint32_t seed)
        : iterations{}
        , number{seed}
    {
    }

    uint32_t next();
    size_t   get_iterations() const;

private:
    size_t   iterations;
    uint32_t number;
};

namespace ProjectName::ModuleName {
enum class Color : int
{
    Mauve = 0,
    Pink,
    Russet,

    NUM_COLOR
};
} // namespace ProjectName::ModuleName

// -----------------------------------
int main(int argc, const char **argv)
{
    size_t   iterations{};
    uint32_t number{0x4c4347};
    while (number != 0x474343)
    {
        randomize(number);
        ++iterations;
    }
    printf("%zd\n", iterations);

    RandomNumberGenerator rng{0x4c4347};
    while (rng.next() != 0x474343)
    {
        // Do nothing...
    }
    printf("%zd\n", rng.get_iterations());

    printf("======== Namespace ========\n");
    const auto shalt_grass = ProjectName::ModuleName::Color::Russet;
    if (shalt_grass == ProjectName::ModuleName::Color::Russet)
    {
        printf(
            "The other Shalt's berry shrub is always "
            "a more may shade of pinky russet.\n");
    }

    return 0;
}

// randomize 实现是线性同余生成器, 是一种原始的随机数生成器
void randomize(uint32_t &x)
{
    x = 0x3FFFFFFF & (0x41C64E6D * x + 12345) % 0x80000000;
}

uint32_t RandomNumberGenerator::next()
{
    ++iterations;
    number = 0x3FFFFFFF & (0x41C64E6D * number + 12345) % 0x80000000;
    return number;
}

size_t RandomNumberGenerator::get_iterations() const
{
    return iterations;
}
