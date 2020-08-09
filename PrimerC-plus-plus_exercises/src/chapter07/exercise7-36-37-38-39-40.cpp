/* exercise 7-36、7-37、7-38、7-39、7-40
** 练习7.36: 下面的初始值是错误的，请找出问题所在并尝试修改它。
struct X
{
    X (int i, int j): base(i), rem(base % j) {}
    int rem, base;
};
**
** 根据类定义内出现的顺序，rem将被优先初始化，而此时base未被初始化，因此错误。改为：
struct X
{
    X (int i, int j): base(i), rem(base % j) {}
    int base， rem;
};
**
** 练习7.37: 使用本节提供的Sales_data类，确定初始化下面的变量时分别使用了哪个构造函数，
** 然后罗列出每个对象所有的数据成员的值。
**
** 练习7.38: 有些情况下我们希望提供cin作为接受istream& 参数的构造函数的默认实参，请声明这样的构造函数。
** solution:
** Sales_data(std::istream &is = std::cin) { read(is, *this)}
** //该构造函数已有了默认构造函数的功能，为避免二义性，因将原构造函数删除
**
** 练习7.39: 如果接受string的构造函数和接受 istream&的构造函数都使用默认实参，
** 这种行为合法吗？如果不，为什么？
** solution:
** 不合法。此时使用构造函数且不输入参数，则两个构造函数都可以为其进行默认构造，
** 将无法选择使用哪个函数，引起二义性。
**
** 练习7.40: 从下面的抽象概念中选择一个（或者你自己指定一个 ，
** 思考这样的类需要哪些数据成员， 提供一组合理的构造函数并阐明这样做的原因
(a) Book           (b) Data           (c) Employee
(d) Vehicle        (e) Object        (f) Tree
** 
** solution:
**
*/

// solution 7-40
class Tree
{
private:
	std::string Name;
	unsigned Age = 0;
	double Height = 0;
public:
	Tree() = default;
	Tree(const std::string &n, unsigned a, double h)
        : Name(n), Age(a), Height(h)
};


// solution：7-37
#include "exercise7-21.hpp"

Sales_data first_item(std::cin);

int main(int argc, char **argv)
{
    // solution：7-37
    Sales_data next;    // Sales_data(std::string s = "")  bookNo="",cnt=0,rev=0.0
    // Sales_data(std::string s = "") bookNo="9-999-99999-9",cnt=0,rev=0.0
    Sales_data last("9-999-99999-9");
    
    return 0;
}


/* 编译命令操作流程
** 0、打开终端 terminal，VSCode使用 Ctrl+shift+`; Ubuntu使用 Ctrl+Alt+T
** 1、进入当前源文件路径，cd src; cd chapter07
** 2、编译源代码文件，g++ --version; g++ -o exercise exercise7-36-37-38-39-40.cpp
** 3、运行生成的可执行程序，exercise; Ubuntu使用 ./exercise
** 4、删除生成的可执行程序，rm -rf exercise.exe; Ubuntu使用 rm -rf exercise
*/
