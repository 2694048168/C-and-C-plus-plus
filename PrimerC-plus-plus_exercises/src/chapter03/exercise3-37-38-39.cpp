/* exercise 3-37、3-38、3-39
** 练习3.37:下面的程序是何含义，程序的输出结果是什么?
** const char ca[] = {'h', 'e', 'l', 'l', 'o'};
** const char *Cp = ca;
** while (*cp) 
** {
**     cout << *Cp << endl;
**     ++cp;
** }
** solution: 遍历一个常量字符数组的元素，h e l l o ,  (
** while 循环结束条件有问题, 会出现一些未知的字符，指针使用注意！！！
**
** 练习3.38: 在本节中提到，将两个指针相加不但是非法的，而且也没什么意义。
** 请问为什么两个指针相加没什么意义?
** solution：指针是用来代表内存地址的。
** 指针的数值是该地址相对于最低位地址也就是0位地址的偏移量，也可称之为坐标。
** 坐标相加得到的新值是没什么意义的，坐标相减则是距离，坐标加距离则是新坐标，后两者是有意义的
** 
** 练习3.39: 编写程序，比较两个string对象
** 再编写一段程序，比较两个 C 风格字符串的内容
**
*/

#include <iostream>
#include <string>
#include <cstring>

int main()
{
    // solutin 3-37
    const char ca[] = {'h', 'e', 'l', 'l', 'o'};
    const char *cp = ca;
    while (*cp)
    {
        std::cout << *cp << std::endl;
        ++cp;
    }

    // solution 3-39
    // use string.
    std::string s1("Mooophy"), s2("Pezy");
    if (s1 == s2)
        std::cout << "same string." << std::endl;
    else if (s1 > s2)
        std::cout << "Mooophy > Pezy" << std::endl;
    else
        std::cout << "Mooophy < Pezy" << std::endl;

    std::cout << "=============================" << std::endl;

    // use C-Style character strings.
    const char* cs1 = "Wangyue";
    const char* cs2 = "Pezy";
    // strcmp include the header file cstring
    auto result = strcmp(cs1, cs2);
    if (result == 0)
        std::cout << "same string." << std::endl;
    else if (result < 0)
        std::cout << "Wangyue < Pezy" << std::endl;
    else
        std::cout << "Wangyue > Pezy" << std::endl;
    
    return 0;
}

/* 编译命令操作流程
** 0、打开终端 terminal，VSCode使用 Ctrl+shift+`; Ubuntu使用 Ctrl+Alt+T
** 1、进入当前源文件路径，cd src; cd chapter03
** 2、编译源代码文件，g++ --version; g++ -o exercise exercise3-37-38-39.cpp
** 3、运行生成的可执行程序，exercise; Ubuntu使用 ./exercise
** 4、随意输入一组数字，例如：1 2 3 4 5 6 7 8 9 ，键入 EOF 即可结束输入
** 5、删除生成的可执行程序，rm -rf exercise.exe; Ubuntu使用 rm -rf exercise
*/
