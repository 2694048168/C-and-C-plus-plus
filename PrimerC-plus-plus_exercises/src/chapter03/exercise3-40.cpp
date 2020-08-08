/* exercise 3-40
** 练习3.40:编写一段程序，定义两个字符数组并用字符串字面值初始化它们;
** 接着再定义一个字符数组存放前两个数组连接后的结果。
** 使用 strcpy 和 strcat 把前两个数组的内容拷贝到第三个数组中
**
*/

#include <iostream>
#include <cstring>

int main()
{
    // solutin 3-40
    // \0 储存字符空间不要忘了
    const char cstr1[]="Hello";
    const char cstr2[]="world!";
    // constexpr size_t new_size = strlen(cstr1) + strlen(" ") + strlen(cstr2) + 1;
    char cstr3[strlen(cstr1) + strlen(" ") + strlen(cstr2) + 1];
    
    // C-style string copy and cat
    strcpy(cstr3, cstr1);
    strcat(cstr3, " ");
    strcat(cstr3, cstr2);
    
    std::cout << cstr3 << std::endl;
    
    return 0;
}

/* 编译命令操作流程
** 0、打开终端 terminal，VSCode使用 Ctrl+shift+`; Ubuntu使用 Ctrl+Alt+T
** 1、进入当前源文件路径，cd src; cd chapter03
** 2、编译源代码文件，g++ --version; g++ -o exercise exercise3-40.cpp
** 3、运行生成的可执行程序，exercise; Ubuntu使用 ./exercise
** 4、删除生成的可执行程序，rm -rf exercise.exe; Ubuntu使用 rm -rf exercise
*/
