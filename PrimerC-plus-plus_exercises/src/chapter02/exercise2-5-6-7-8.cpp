/* exercise 2-5、2-6、2-7、2-8
** 练习2.5:指出下述字面值的数据类型并说明每一组内几种字面值的区别:
** (a) 'a'，L'a', "a", L"a"
** (b) 10，10u， 10L， 10uL， 012, 0xC
** (c) 3.14， 3.14f, 3.14L
** (d) 10，10u， 10.， 10e-2
** solution:
** (a) 'a' char类型字符字面值；L'a' wchar_t类型宽字符字符字面值；
**     "a" const char类型字符串字面值；L"a" wchar_t类型宽字符字符串字面值；
** 字符和字符串区别, 字符串就是字符序列 + '\0' 作为结束符
** ASCII字符集称为窄字符，不能存放世界上所有语言所有文字
** Unicode字符集称为宽字符，可以存放世界上所有语言所有文字
** (b) 10 int类型整数字面值；10u 代表最小匹配类型 unsigned 整数字面值；
**     10L long类型整数字面值；10uL，最小匹配类型unsigned long整数字面值；
**     012 八进制表示的整数；0xC 十六进制表示的整数
** (c) 3.14 double类型浮点数字面值；3.14f float类型浮点数字面值；
**     3.14L long double类型浮点数字面值；
** (d) 10 int类型整数字面值；10u unsigned int类型整数字面值；
**     10. double类型浮点数字面值；10e-2 科学计数法表示的浮点数double类型字面值；
** 
** 练习2.6:下面两组定义是否有区别，如果有，请叙述之:
** int month=9，day=7;
** int month = 09，day = 07;
** solution：
** 有区别，第一组是十进制数；第二组是八进制数；还有 0x 开头的十六进制数
** 09 是一个无效的八进制数
**
** 练习2.7:下述字面值表示何种含义? 它们各自的数据类型是什么?
** (a) "Who goes with F\145rgus?\012"
** (b) 3.14e1L
** (c) 1024f
** (d) 3.14L
** solution：
** (a) const char[] 字符数组 [Who goes with Fergus?\0]
** \145 和 \012 表示泛化的转义字符(使用十进制数字\1，八进制数字\0，十六进制数字\x)
** (b) 3.14e1L long double类型浮点数 科学计数法表示的31.399999999999999L
** (c) 3.14L long double类型浮点数 3.1400000000000001L

** 练习2.8:请利用转义序列编写一段程序， 要求先输出2M，然后转到新一行
** 修改程序使其先输出2，然后输出制表符，再输出M，最后转到新一行
*/

#include <iostream>

int main()
{
    std::cout << "2M\n" << "new row" << std::endl;

    std::cout << 2 << "\t" << "new tab " << 'M' << "\n" << "new row" << std::endl;

    return 0;
}

/* 编译命令操作流程
** 0、打开终端 terminal，VSCode使用 Ctrl+shift+`; Ubuntu使用 Ctrl+Alt+T
** 1、进入当前源文件路径，cd src; cd chapter02
** 2、编译源代码文件，g++ --version; g++ -o exercise exercise2-5-6-7-8.cpp
** 3、编译源代码文件并指定标准版本，g++ --version; g++ -std=c++11 -o exercise exercise2-5-6-7-8.cpp
** 4、运行生成的可执行程序，exercise; Ubuntu使用 ./exercise
** 5、删除生成的可执行程序，rm -rf exercise.exe; Ubuntu使用 rm -rf exercise
*/
