/* exercise 3-9、3-10、3-11
** 练习3.9: 下面的程序有何作用?它合法吗?如果不合法，为什么?
** string s;
** cout << s[0] << endl;
** solution: 输出字符串的第一个字符。合法，空串第一个字符，空的。
**
** 练习3.10: 编写一段程序，读入一个包含标点符号的字符串,将标点符号去除后输出字符串剩余的部分
**
** 练习3.11:下面的范围for语句合法吗?如果合法，c的类型是什么?
** const string s = "Keep out!";
** for(auto &c : s){ }
** solution: 合法，但是不能对 c 和 s 做任何重新赋值操作，因为 const 常量
** c 是 const char 类型的引用
**
*/

#include <iostream>
#include <cctype>

int main()
{
    // solutin 3-10
    std::string str;
    // test sequence stsring : weili,./';?:"weili 
    std::cout << " Please enter a string with punctuation : " ;
    std::cin >> str;
    std::cout << "the input string is : " << str << std::endl;
    for (auto i : str)
    {
        // 判断字符是否是标点符号
        // ispunct include the cctype header file
        if (!ispunct(i))
        {
            // 下标访问符 [ ] 序列类型是 std::string::size_type
            std::cout << i;
        }
    }
    std::cout << std::endl;
    
    return 0;
}

/* 编译命令操作流程
** 0、打开终端 terminal，VSCode使用 Ctrl+shift+`; Ubuntu使用 Ctrl+Alt+T
** 1、进入当前源文件路径，cd src; cd chapter03
** 2、编译源代码文件，g++ --version; g++ -o exercise exercise3-9-10-11.cpp
** 3、运行生成的可执行程序，exercise; Ubuntu使用 ./exercise
** 4、随意带有标点符号的字符串，weili,./';?:"weili 
** 5、删除生成的可执行程序，rm -rf exercise.exe; Ubuntu使用 rm -rf exercise
*/
