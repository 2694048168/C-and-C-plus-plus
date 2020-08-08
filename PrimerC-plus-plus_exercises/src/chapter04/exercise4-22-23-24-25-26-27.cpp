/* exercise 4-22、4-23、4-24、4-25、4-26、4-27
** 练习4.22:本节的示例程序将成绩划分成high pass、pass 和fail三种，扩展该程序使其
** 进一步将60分到75分之间的成绩设定为lowpass。要求程序包含两个版本:一个版本
** 只使用条件运算符;另外一个版本使用1个或多个if语句。哪个版本的程序更容易理解呢?为什么? 
** solution: if 逻辑更加清晰
**
** 练习4.23: 因为运算符的优先级问题，下面这条表达式无法通过编译。
** 根据运算符优先级，指出它的问题在哪里?应该如何修改?
** string s = "word";
** stringpl = s + s[s.size() - 1] == 's' ? "" : "s";
** solution: 加法+ 优于 等号== ==优于条件运算符？：
** stringpl = s + s[s.size() - 1] == ('s' ? "" : "s");
**
** 练习4.24:本节的示例程序将成绩划分成highpass、pass和fail三种，
** 它的依据是条件运算符满足右结合律。假如条件运算符满足的是左结合律，求值过程将是怎样的?
** solution: 各种运算符的优先级和结合律，了解很有必要，建议使用括号增加优先级
**
** 练习4.25:如果一台机器上int占32位、char占8位，用的是Latin-1 字符集，
** 其中字符'q'的二进制形式是01110001，那么表达式'q'<<6的值是什么?
** solution：计算机基础 二进制计算
**
** 练习4.26:在本节关于测验成绩的例子中，如果使用unsigned int作为quizl的类型会发生什么情况?
** solution：类型的表示范围有所差异
**
** 练习4.27: 下列表达式的结果是什么?
** unsigned long ull = 3, u12 = 7;
** (a) ull & ul2
** (b) ul1 | ul2
** (c) ul1 && ul2
** (d) ul1 || ul2
** solution：逻辑运算 和 位操作运算 是不同的运算
**
*/

#include <iostream>

int main()
{
    // solution 4-22
     for (unsigned g; std::cin >> g; )
    {
        // conditional operators
        auto result = g > 90 ? "high pass" : g < 60 ? "fail" : g < 75 ? "low pass" : "pass";
        std::cout << result << std::endl;

        // if statements
        if (g > 90)         std::cout << "high pass";
        else if (g < 60)    std::cout << "fail";
        else if (g < 75)    std::cout << "low pass";
        else                std::cout << "pass";
        std::cout << std::endl;
    }
    
    return 0;
}

/* 编译命令操作流程
** 0、打开终端 terminal，VSCode使用 Ctrl+shift+`; Ubuntu使用 Ctrl+Alt+T
** 1、进入当前源文件路径，cd src; cd chapter04
** 2、编译源代码文件，g++ --version; g++ -o exercise exercise4-22-23-24-25-26-27.cpp
** 3、运行生成的可执行程序，exercise; Ubuntu使用 ./exercise
** 4、enter the score for 0-100
** 5、删除生成的可执行程序，rm -rf exercise.exe; Ubuntu使用 rm -rf exercise
*/
