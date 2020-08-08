/* exercise 2-32、2-33、2-34、2-35
** 练习2.32: 下列代码是否合法？非法，修改正确
** int null = 0, *p = null;
**
** 练习2.33:利用本节定义的变量，判断下列语句的运行结果
** a = 42; b = 42; C = 42;
** d = 42; e = 42; g = 42;
**
** 练习2.34:基于上一个练习中的变量和语句编写一段程序，输出赋值前后变量的内容，你刚才的推断正确吗?
** 如果不对，请反复研读本节的示例直到你明白错在何处为止
**
** 练习2.35: 判断下列定义推断出的类型是什么，然后编写程序进行验证。
** const int i = 42;
** auto j = i;
** const auto &k = i;
** auto *p = &i;
** const auto j2 = i, &k2 = i;
**
*/

#include <iostream>

int main()
{
    // solution 2-32
    // 非法，int类型不能初始化int*类型的实体
    // int null = 0, *p = null;
    int null = 0, *p = &null;

    // solutin 2-33 2-34
    int i = 0, &r = i;
    auto a = r;
    // auto 一般会忽略 top-level const, low-level const 保留
    const int ci = i, &cr  = ci;
    auto b = ci;
    auto c = cr;
    auto d = &i;
    auto e = &ci;
    const auto f = ci;
    auto &g = ci;
    const auto &j = 42;

    a = 42;   // 合法，a 为 r 引用，r 为 i 引用，故此，a=r=i=42
    b = 42;   // 合法，b 为 const int 常量引用，top-level const，
    c = 42;   // 合法
    // d = 42;   // 非法，类型不匹配
    // e = 42;   // 非法，类型不匹配
    // g = 42;   // 非法，g 不能修改

    // solution 2-35
    // VSCode 中，将鼠标放在变量上，编译器会自动识别到其类型
    // 注意 auto 会忽略 top-level const 
    const int i = 42;
    auto j = i;
    const auto &k = i;
    auto *p = &i;
    const auto j2 = i, &k2 = i;

    return 0;
}

/* 编译命令操作流程
** 0、打开终端 terminal，VSCode使用 Ctrl+shift+`; Ubuntu使用 Ctrl+Alt+T
** 1、进入当前源文件路径，cd src; cd chapter02
** 2、编译源代码文件，g++ --version; g++ -o exercise exercise2-32-33-34-35.cpp
** 3、运行生成的可执行程序，exercise; Ubuntu使用 ./exercise
** 4、删除生成的可执行程序，rm -rf exercise.exe; Ubuntu使用 rm -rf exercise
*/
