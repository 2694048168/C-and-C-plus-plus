/* exercise 7-43、7-44、7-45、7-46
** 练习7.43: 假定有一个名为NoDefault 的类，它有一个接受int的构造函数，但是没有默认构造函数。
** 定义类C, C 有一个NoDefault 类型的成员，定义C 的默认构造函数。
**
** 练习7.44: 下面这条声明合法吗？如果不，为什么？
** vector<NoDefault> vec(10);
** solution:
** 这条语句的意思是初始化一个vector，其中包含10个NoDefault类（由其自身默认初始化）。
** 而该类型不存在默认构造函数，无法成功构造因此不合法。
**
** 练习7.45: 如果在上一个练习中定义的vector 的元素类型是C，则声明合法吗？为什么？
** solution:
** 合法，C可以默认初始化。
**
** 练习7.46: 下面哪些论断是不正确的？为什么？
( a ） 一个类必须至少提供一个构造函数。
( b ）默认构造函数是参数列表为空的构造函数。
( c ） 如果对于类来说不存在有意义的默认值，则类不应该提供默认构造函数。
( d ）如果类没有定义默认构造函数，则编译器将为其生成一个并把每个数据成员初始化成相应类型的默认值。
**
** solution:
（a）错误。见（d）
（b）错误。默认构造函数也可以有参数，为对应的成员提供默认初始化。
（c）错误。一般情况下，类需要一个默认构造函数来初始化其成员。
（d）错误。只要类内有了构造函数，编译器就不会再为其生成构造函数。
**
*/

// solution 7-43
#include <vector> 

class NoDefault 
{
public:
    NoDefault(int i) { }
};

class C 
{
public:
    C() : def(0) { } // define the constructor of C.
private:
    NoDefault def;
};


int main(int argc, char **argv)
{
    C c;
    
    std::vector<C> vec(10); 
    
    return 0;
}


/* 编译命令操作流程
** 0、打开终端 terminal，VSCode使用 Ctrl+shift+`; Ubuntu使用 Ctrl+Alt+T
** 1、进入当前源文件路径，cd src; cd chapter07
** 2、编译源代码文件，g++ --version; g++ -o exercise exercise7-43-44-45-46.cpp
** 3、运行生成的可执行程序，exercise; Ubuntu使用 ./exercise
** 4、删除生成的可执行程序，rm -rf exercise.exe; Ubuntu使用 rm -rf exercise
*/
