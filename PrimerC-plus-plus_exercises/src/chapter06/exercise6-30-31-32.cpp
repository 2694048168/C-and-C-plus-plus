/* exercise 6-30、6-31、6-32
** 练习6.30：编译第 200 页的 str_subrange 函数，看看编译器是如何处理函数中的错误的
**
** 练习6.31：什么情况下返回的引用无效？什么情况下返回常量的引用无效？
** solution: 函数返回结果的过程和接受参数的过程类似，如果返回的是值，则创建一个唯一命名那个的临时对象，
**          并把返回的值拷贝给这个临时对象，如果返回的是引用，则该引用是他所引用对象的别名，不会真正拷贝对象
** 如果引用的对象在函数之前已经存在，则返回该引用是有效的，
** 如果引用索引的是函数的局部变量，则随着函数结束局部变量也失效了，此时返回的引用无效
**
** 练习6.32：下面的函数合法吗？如果合法，说明其功能；如果不合法，修改其中的错误并解释原因。
** int &get (int *arry, int index) { return arry[index]; }
** int main() {
    int ia[10];
    for (int i = 0; i != 10; ++i)
        get(ia, i) = i;
}
** solution：
*/

#include <iostream>

//因为含有不正确的返回值，所以这段代码无法通过编译
bool str_subrange(const std::string &str1, const std::string &str2)
{
    //大小相同:此时用普通的相等性判断结果作为返回值
    if (str1.size() == str2.size())
        return str1 == str2;
    // 正确: ==运算符返回布尔值
    //得到较短string对象的大小，条件运算符参见第4.7节(134页)
    auto size = (str1.size() < str2.size())
                    ? str1.size()
                    : str2.size();
    //检查两个string对象的对应字符是否相等，以较短的字符串长度为限
    for (decltype(size) i = 0; i != size; ++i)
    {
        if (str1[i] != str2[i])
            return; // 错误#1:没有返回值，编译器将报告这一错误
        //错误#2:控制流可能尚未返回任何值就结束了函数的执行
        // 编译器可能检查不 出这一错误
    }
}

int main(int argc, char *argv[])
{
    std::string str1{"weili"}, str2{"yzzcq"};

    str_subrange(str1, str2);

    // solution:
    // In function 'bool str_subrange(const string&, const string&)':
    // error: return-statement with no value, in function returning 'bool' [-fpermissive]
    // return; // ? 

    return 0;
}

/* 编译命令操作流程
** 0、打开终端 terminal，VSCode使用 Ctrl+shift+`; Ubuntu使用 Ctrl+Alt+T
** 1、进入当前源文件路径，cd src; cd chapter06
** 2、编译源代码文件，g++ --version; g++ -o exercise exercise6-30-31-32.cpp
** 3、运行生成的可执行程序，exercise; Ubuntu使用 ./exercise
** 4、删除生成的可执行程序，rm -rf exercise.exe; Ubuntu使用 rm -rf exercise
*/
