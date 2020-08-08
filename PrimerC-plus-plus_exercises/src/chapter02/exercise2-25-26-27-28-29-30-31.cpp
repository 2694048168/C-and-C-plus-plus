/* exercise 2-25、2-26、2-27、2-28、2-29、2-30、2-31
** 练习2.25: 说明下列变量的类型和值
** (a) int* ip, i, &r = i; 
** (b) int i, *ip = 0; 
** (c) int* ip， ip2;
** solution：修饰符
** (a) ip 是一个 int 类型的指针变量，未初始化；
**     i 是一个 int 类型变量，未初始化；r 是一个 int 类型的引用，初始化为 i
** (b) i 是一个 int 类型变量，未初始化；ip 是一个int类型的指针变量，初始化为 NULL=0=nullptr
** (c) ip 是一个int类型的指针变量，未初始化；ip2 是一个int类型变量，未初始化
**
** 练习2.26: 说明下列那些句子是合法的？不合法的，为什么？
** (a) const int buf;
** (b) int cnt = 0;
** (c) const int sz = cnt;
** (d) ++cnt; ++sz;
** solution：const 限定符
** (a) 非法，const 限制 buf 为常量，其值不能修改，必须初始化；
** (b) 合法，声明并初始化 int类型变量 cnt 为 0；
** (c) 合法，使用一个对象为 const 限制的 sz int类型常量进行初始化；
** (d) 合法，int类型变量自增运算；const int 类型变量不能进行自增运算，本身值不能进行修改；
**
** 练习2.27:下面的哪些初始化是合法的?请说明原因
** (a) int i = -1, &r = 0;
** (b) int *const p2 = &i2;
** (c) const int i = -1，&r = 0; 
** (d) const int *const p3 = &i2;
** (e) const int *p1 = &i2;
** (f) const int &const r2;
** (g) const int i2 = i, &r = i;
** solution:
** (a) int 类型变量 i 初始化合法；r int类型引用初始化非法，非常量的引用初始化必须为可修改左值；
** (b) 声明一个int 类型的指针变量const限制常量不能修改指针指向的值，用另一个对象的地址可以初始化；
** (c) 顶层const 限定两个变量都为常量，int类型常量 i 初始化合法；常量引用 r 初始化合法；
** (d) 合法，用一个对象的地址为一个指针常量初始化合法
** (f) 非法，常量必须初始化
** (g) 合法
** summary：const 修饰指针
** const 修饰指针，称之为常量指针，特点是指针的指向可以修改，指针指向的值不能修改
** const int *p = &a; 这就是常量指针
** const 修饰常量，称之为指针常量，特点是指针的指向不能修改，指针指向的值可以修改
** int * const p = &a; 这就是指针常量
** const 既修饰指针，也修饰常量，特点是指针的指向不能修改，指针指向的值也不能修改
** const int * const p = &a; 这就是两者都修饰
**
** 练习2.28:说明下面的这些定义是什么意思，挑出其中不合法的
** (a) int i, *const cp;
** (b) int *p1, *const p2;
** (c) const int ic, &r = ic;
** (d) const int *const p3;
** (e) const int *p;
** solution：
** (a) 声明一个int类型变量i，合法；声明一个指针常量cp，必须初始化常量，非法；
** (b) 声明一个int类型指针变量p1，合法；声明一个指针常量p2，必须初始化常量，非法；
** (C) 声明一个常量ic，必须初始化常量，非法；声明并初始化引用，合法；
** (d) const 修饰指针和常量，常量必须初始化，非法；
** (e) 声明一个常量指针，合法
**
** 练习2.29:假设已有上一个练习中定义的那些变量，则下面的哪些语句是合法的?请说明原因
** (a) i = ic;
** (b) p1 = p3;
** (C) p1 = &ic;
** (d) p3 = &ic;
** (e) p2 = p1;
** (f) ic = *p3;
** solution:
** (a) 合法，ic是一个常量，其值不能修改，但是可以参与赋值运算
** (b) 非法，赋值运算必须类型匹配，int * 和 const int * 类型不匹配
** (c) 非法，类型不匹配
** (d) 非法，p3 被const修饰指针和常量，其指向和指向的值都不能修改
** (e) 非法，p2 是一个指针常量，其指向的值不能修改
** (f) 非法，ic 是一个常量，值不能修改
** 
** 练习2.30: 对于下面的这些语句，请说明对象被声明成了顶层const还是底层const?
** const int v2 = 0;
** int v1 = v2;
** int *p1 = &v1，&r1 = v1;
** const int *p2 = &v2， *const p3 = &i, &r2 = v2;
**
** 练习2.31: 假设已有上一个练习中所做的那些声明，则下面的哪些语句是合法的?
** 请说明顶层const和底层const在每个例子中有何体现。
** r1 = v2;
** p1 = p2;
** p2 = p1;
** p1 = p3;
** p2 = p3;
** 
** summary：
** 顶层const(top-level const)，即就是const修饰指针本身是一个常量，常量指针
** 底层const(low-level const)，即就是const修饰指针指向的对象是一个常量，指针常量
** 既可以是被顶层const修饰的，也同时被顶层const修饰的
*/

#include <iostream>

int main()
{
    // solution 2-30
    const int v2 = 0;  // low-level const
    int v1 = v2;
    int *p1 = &v1, &r1 = v1;  
    // top-level const, top-levle and low-level cansot, top-level const
    const int *p2 = &v2, *const p3 = &i, &r2 = v2;  

    // solution 2-31
    r1 = v2;
    p1 = p2;  // 类型不匹配
    p2 = p1;
    p1 = p3;  // 类型不匹配
    p2 = p3;

    return 0;
}

/* 编译命令操作流程
** 0、打开终端 terminal，VSCode使用 Ctrl+shift+`; Ubuntu使用 Ctrl+Alt+T
** 1、进入当前源文件路径，cd src; cd chapter02
** 2、编译源代码文件，g++ --version; g++ -o exercise exercise2-25-26-27-28-29-30-31.cpp
** 3、运行生成的可执行程序，exercise; Ubuntu使用 ./exercise
** 4、删除生成的可执行程序，rm -rf exercise.exe; Ubuntu使用 rm -rf exercise
*/
