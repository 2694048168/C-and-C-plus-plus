/**
 * @file 00_storageDuration.cpp
 * @author Wei Li (weili_yzzcq@163.com)
 * @brief 
 * @version 0.1
 * @date 2024-04-26
 * 
 * @copyright Copyright (c) 2024
 * 
 */

#include <cstdio>

/**
 * @brief 对象的存储期
 * 对象是一段存储空间, 它有类型和值, 当声明一个变量时, 其实就创建了一个对象.
 * 
 * 分配、释放和生命周期 Allocation, Deallocation, and Lifetime
 * 每个对象都需要存储空间, 为对象保留存储空间的过程叫作 分配(allocation),
 * 当使用完对象释放对象的存储空间的过程叫作 释放(deallocation)后,
 * 对象的存储期(storage duration)从对象被分配存储空间时开始, 到对象被释放时结束.
 * 对象的生命周期是一个运行时属性, 受对象的存储期的约束.
 * 对象的生命周期从构造函数运行完成时开始, 在调用析构函数之前结束:
 * 1) 对象的存储期开始, 并分配存储空间.
 * 2) 对象的构造函数被调用.
 * 3) 对象的生命周期开始.
 * 4) 在程序中使用该对象.
 * 5) 对象的生命周期结束.
 * 6) 对象的析构函数被调用.
 * 7) 对象的存储期结束, 存储空间被释放.
 * 
 * 内存管理
 * 计算机编程语言, 那么有可能使用过自动内存管理功能, 或者垃圾收集器GC.
 * 在运行期间, 程序会创建对象, 垃圾收集器会定期确定哪些对象不再被程序所需要, 并安全地将它们释放.
 * 这种方法使程序员不用管理对象的生命周期, 但也需要付出一些代价, 包括运行时的性能损失, 
 * 而且需要一些强大的编程技术, 如确定性资源管理.
 * C++采取的是一种更有效的方法, 这样做的代价是, C++程序员必须对存储期有深入的了解.
 * 控制对象生命周期是编程者的责任, 而不是垃圾收集器的责任.
 * 
 */

/**
* @brief 自动存储期
* 自动对象 在代码块的开头被分配, 而在结尾处会释放.
* 代码块就是自动对象的 作用域, 自动对象具有自动存储期.
* 注意, 函数的参数是自动对象, 尽管从符号上来看它们出现在函数体之外.
* 函数调用时候参数列表都会被分配, 在函数返回之前, 这些变量会被释放.
* 因为在函数之外不能访问这些变量, 所以自动变量也被称为 局部变量.
* 
*/
void power_up_rat_thing(int nuclear_isotopes)
{
    int waste_heat = 0;
    printf("local variables: %d, and param: %d\n", waste_heat, nuclear_isotopes);
}

/**
 * @brief 态存储期
 * 静态对象是用 static 或 extern 关键字来声明的,
 * 在声明函数的同一范围内声明静态变量这一范围即全局作用域(或命名空间作用域),
 * 全局作用域的静态对象具有静态存储期, 在程序启动时分配, 在程序停止时释放.
 * 
 * 在全局作用域内用 static 关键字声明的,
 * 在全局作用域内声明的另一个作用是可以从编译单元的任何函数中访问
 * (编译单元是预处理器在对单个源文件进行处理后产生的)
 * 
 * 当使用 static 关键字时, 可以指定内部链接(internal linkage);
 * 内部链接意味着变量不能被其他编译单元访问.
 * 也可以指定外部链接(external linkage)使变量可以被其他编译单元访问,
 * 对于外部链接, 使用 extern 关键字而不是 static 关键字.
 * 
 */

static int rat_things_power        = 200;
extern int rat_things_power_extern = 200; // external linkage

void power_thing(int nuclear_isotopes)
{
    rat_things_power      = rat_things_power + nuclear_isotopes;
    const auto waste_heat = rat_things_power * 20;
    if (waste_heat > 10000)
    {
        printf("Warning! Hot doggie!\n");
    }
}

/**
 * @brief 局部静态变量
 * 局部静态变量是一种特殊的静态变量, 它是局部有效的, 而不是全局有效的.
 * 局部静态变量是在函数作用域声明的, 就像自动变量一样, 
 * 但是它们的生命周期从包含它的函数的第一次调用开始, 直到程序退出时结束.
 * 
 * 由于变量的局部性, 不能从function的外部引用, 这是名为封装的编程模式的一个例子
 * 封装是指将数据与操作这些数据的函数捆绑在一起, 它有助于防止意外的修改.
 * 
 */
void power_up_rat_thing_static(int nuclear_isotopes)
{
    static int rat_things_power = 200;

    rat_things_power      = rat_things_power + nuclear_isotopes;
    const auto waste_heat = rat_things_power * 20;
    if (waste_heat > 10000)
    {
        printf("Warning! Hot doggie!\n");
    }
    printf("Rat-thing power: %d\n", rat_things_power);
}

/**
 * @brief 静态成员
 * 静态成员是指类的成员, 但是和类的任何实例都不关联,
 * 普通类成员的生命周期嵌套在类的生命周期中, 但静态成员具有静态存储期.
 * 这些成员本质上类似于在全局作用域中声明的静态变量和函数, 但是必须使用类的名称加上作用域解析运算符::来引用它们.
 * 事实上必须在全局作用域初始化静态成员, 不能在类定义中初始化静态成员.
 * 注意 静态成员初始化规则有一个例外:可以在类定义中声明和定义整数类型, 只要它们也被限定为 const.
 * 和其他静态变量一样, 静态成员只有一个实例,
 * 拥有静态成员的类的所有实例都共享同一个静态成员, 所以如果修改了静态成员, 所有的类实例都会观察到这个修改.
 * 
 */
struct RatThing
{
    static int rat_things_power;

    static void power_up_rat_thing(int nuclear_isotopes)
    {
        rat_things_power = rat_things_power + nuclear_isotopes;

        const auto waste_heat = rat_things_power * 20;
        if (waste_heat > 10000)
        {
            printf("Warning! Hot doggie!\n");
        }
        printf("Rat-thing power: %d\n", rat_things_power);
    }
};

int RatThing::rat_things_power = 200;

// -----------------------------------
int main(int argc, const char **argv)
{
    power_up_rat_thing(42);

    printf("Rat-thing power: %d\n", rat_things_power);
    power_thing(100);
    printf("Rat-thing power: %d\n", rat_things_power);
    power_up_rat_thing(500);
    printf("Rat-thing power: %d\n", rat_things_power);

    power_up_rat_thing_static(800);
    power_up_rat_thing_static(500);

    RatThing::power_up_rat_thing(100);
    RatThing::power_up_rat_thing(500);

    return 0;
}
