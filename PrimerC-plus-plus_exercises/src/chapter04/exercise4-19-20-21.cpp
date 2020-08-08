/* exercise 4-19、4-20、4-21
** 练习4.19:假设ptr的类型是指向int的指针、vec的类型是vector<int>、ival的类型是int,
** 说明下面的表达式是何含义?如果有表达式不正确，为什么?应该如何修改?
** (a) ptr != 0 && *ptr++
** (b) ival++ && ival
** (c) vec[ival++] <= vec[ival]
** solution：
** (a) 判断指针是否为空，同时指针移动一个元素后解引用的值是否为0
** (b) int类型值自增后是否为0，同时本身值是否为0
** (c) 判断vector对应索引的值的大小
** summary：运算符的优先级很重要，建议使用括号 (), 显式指定运算优先级
**
** 练习4.20:假设iter的类型是vector<string>::iterator,说明下面的表达式是否合法。
** 如果合法，表达式的含义是什么?如果不合法，错在何处?
** (a) *iter++;
** (b) (*iter)++;
** (c) *iter.empty()
** (d) iter->empty();
** (e) ++*iter;
** (f) iter++->empty();
** solution：
** (a) 迭代器自增后解引用
** (b) 迭代器解引用后自增
** (c) iter.empty(), 判断迭代器是否为空
** (d) 判断迭代器的值是否为空
** (e) *++iter，自增后解引用
** (f) 迭代器自增后检测是否为空
** summary：迭代器很重要，特别注意指针、迭代器、解引用等这些重要概念
**
** 练习4.21:编写程序，使用条件运算符从vector<int>中找到哪些元素的值是奇数，然后将这些奇数值翻倍。
**
*/

#include <iostream>
#include <vector>

int main()
{
    // solution 4-21
    std::vector<int> ivec{ 1, 2, 3, 4, 5, 6, 7, 8, 9 };
    // for range
    for (auto i : ivec)
    {
        std::cout << ((i & 0x1) ? i * 2 : i) << " ";
    }
    std::cout << std::endl;
    
    return 0;
}

/* 编译命令操作流程
** 0、打开终端 terminal，VSCode使用 Ctrl+shift+`; Ubuntu使用 Ctrl+Alt+T
** 1、进入当前源文件路径，cd src; cd chapter04
** 2、编译源代码文件，g++ --version; g++ -o exercise exercise4-19-20-21.cpp
** 3、运行生成的可执行程序，exercise; Ubuntu使用 ./exercise
** 4、删除生成的可执行程序，rm -rf exercise.exe; Ubuntu使用 rm -rf exercise
*/
