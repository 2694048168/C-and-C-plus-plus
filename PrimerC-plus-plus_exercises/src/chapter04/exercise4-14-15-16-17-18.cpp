/* exercise 4-14、4-15、4-16、4-17、4-18
** 练习4.14: 执行下述if语句后将发生什么情况?
** if(42=i)//...
** if(i=42)//...
** solution: = 运算符是右结合，而且左边对象一定是可以修改的左值
** 第一个返回结果 42=i ，False，跳过if 的语句块，执行下面代码
** 第二个返回结果 i=42，True，执行if语句块
**
** 练习4.15:下面的赋值是非法的，为什么?应该如何修改?
** double dval; int ival; int *pi;
** dval = ival = pi = 0;
** solution：= 是右结合，double and int 类型不能赋值地址或者指针
** pi = 0; ival = *pi; dval = *pi
**
** 练习4.16: 尽管下面的语句合法，但它们实际执行的行为可能和预期并不一样,为什么?应该如何修改?
** (a) if (p = getPtr() != 0)
** (b) if (i = 1024)
** solution: 函数调用符() 优于 不等于!= , 优于赋值运算符=，同时= 是右结合
** (a) if ((p = getPtr()) != 0)
** (b) if (i)
**
** 练习4.17:说明前置递增运算符和后置递增运算符的区别。
** solution: 建议:除非必须，否则不用递增递减运算符的后置版本!!!
** 有C语言背景的读者可能对优先使用前置版本递增运算符有所疑问,其实原因非常简单:
** 前置版本的递增运算符避免了不必要的工作，它把值加 1 后直接返回改变了的运算对象。
** 与之相比，后置版本需要将原始值存储下来以便于返回这个未修改的内容。
** 如果不需要修改前的值，那么后置版本的操作就是一种浪费
** 对于整数和指针类型来说，编译器可能对这种额外的工作进行一定的优化;
** 但是对于相对复杂的迭代器类型，这种额外的工作就消耗巨大了。
** 建议养成使用前置版本的习惯，这样不仅不需要担心性能的问题，而且更重要的是写出的代码会更符合编程的初衷。
**
** 练习4.18:如果第132页那个输出vector对象元素的while循环使用前置递增运算符，将得到什么结果?
** solution: 
*/

#include <iostream>
#include <vector>

int main()
{
    // solution page132
    std::vector<int> v{1, 2, 3, 4, 5, 6, 7, -1, 8, 9};
    auto pbeg = v.begin();
    // 输出元素直至遇到第一个负值为止
    while (pbeg != v.end() && *pbeg >= 0)
    {
        // 输出当前值并将pbeg向前移动一个元素
        std::cout << *pbeg++ << std::endl;
    }

    std::cout << "===============" << std::endl;

    // solution 4-18
    std::vector<int> ivec{1, 2, 3, 4, 5, 6, 7, -1, 8, 9};
    auto pbegin = ivec.begin();
    // 输出元素直至遇到第一个负值为止
    while (pbegin != ivec.end() && *pbegin >= 0)
    {
        // 输出当前值并将pbeg向前移动一个元素
        std::cout << *++pbegin << std::endl;
    }
    
    return 0;
}

/* 编译命令操作流程
** 0、打开终端 terminal，VSCode使用 Ctrl+shift+`; Ubuntu使用 Ctrl+Alt+T
** 1、进入当前源文件路径，cd src; cd chapter04
** 2、编译源代码文件，g++ --version; g++ -o exercise exercise4-14-15-16-17-18.cpp
** 3、运行生成的可执行程序，exercise; Ubuntu使用 ./exercise
** 4、删除生成的可执行程序，rm -rf exercise.exe; Ubuntu使用 rm -rf exercise
*/
