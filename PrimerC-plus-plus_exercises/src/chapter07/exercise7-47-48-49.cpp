/* exercise 7-47、7-48、7-49
** 练习7.47: 说明接受一个string 参数的Sales_data构造函数是否应该是explicit的，并解释这样做的优缺点。
** solution:
** 应该是explicit的，这样可以防止编译器自动的把一个string对象转换成Sales_data对象，
** 这样可能会导致意想不到的后果。
** 使用explicit的优点是避免因隐式转换而带来的意想不到的错误，
** 缺点是用户的确需要这样的类类型转换时，不得不使用略显繁琐的方式来实现。
** 
** 练习7.48: 假定Sales_data 的构造函数不是explicit 的，则下边定义将执行什么样的操作？
** 如果Sales_data 的构造函数是explicit 的，又会发生什么呢？
string null_isbn("9-999-99999-9");
Sales_data item1(null_isbn);
Sales_data item2("9-999-99999-9");

不是explicit。

Sales_data item1(null_isbn);//定义了一个Sales_data对象，该对象利用null_isbn转换得到的临时对象通过构造函数进行初始化
Sales_data item2("9-999-99999-9");//定义了一个Sales_data对象，该对象使用字符串字面值转换得到的临时对象通过构造函数进行初始化
是explicit同上。

** 练习7.49: 对于combine 函数的三种不同声明，当我们调用i.combine(s) 时分别发生什么情况？其中 i 是一个 Sales_data，而 s 是一个string对象。
(a)Sales_data & combine(Sales_data);
(b)Sales_data & combine(Sales_data &);
(c)Sales_data & combine(const Sales_data &) const;

** solution:
（a）正确。s隐式调用了Sales_data的构造函数，生成临时对象并传递给combine的形参。
（b）错误。因为combine成员函数的形参是非常量引用，但是s自动创建的Sales_data临时对象无法传递给combine所需的非常量引用。（PS：隐式转换生成的无名的临时对象是const的）
（c）编译无法通过。因为我们把combine成员函数声明成了常量成员函数，所以该函数无法修改数据成员的值。
**
*/


int main(int argc, char **argv)
{
    
    return 0;
}


/* 编译命令操作流程
** 0、打开终端 terminal，VSCode使用 Ctrl+shift+`; Ubuntu使用 Ctrl+Alt+T
** 1、进入当前源文件路径，cd src; cd chapter07
** 2、编译源代码文件，g++ --version; g++ -o exercise exercise7-47-48-49.cpp
** 3、运行生成的可执行程序，exercise; Ubuntu使用 ./exercise
** 4、删除生成的可执行程序，rm -rf exercise.exe; Ubuntu使用 rm -rf exercise
*/
