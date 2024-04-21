/**
 * @file 07_ClassTypes.cpp
 * @author Wei Li (weili_yzzcq@163.com)
 * @brief 
 * @version 0.1
 * @date 2024-04-21
 * 
 * @copyright Copyright (c) 2024
 * 
 */

#include <cstdio>

/**
 * @brief Fully Featured C++ Classes 全功能的C++类
 * POD类只包含数据成员, 有时这就是对类的全部要求, 然而只用POD类设计程序会很复杂,
 * 可以用封装来处理这种复杂性, 封装是一种设计模式, 将数据与操作它的函数结合起来, 
 * 将相关的函数和数据放在一起, 至少在两个方面有助于简化代码.
 * 1. 可以把相关的代码放在一个地方, 这有助于对程序进行推理,
 *    有助于对代码工作原理的理解, 因为它在一个地方同时描述了程序状态以及代码如何修改该状态.
 * 2. 其次, 可以通过一种叫作信息隐藏的做法将类的一些代码和数据相对程序的其他部分隐藏起来,
 *    在C++中, 向类定义添加方法和访问控制即可实现封装.
 * 
 * ==== 方法
 * 方法就是成员函数, 在类、其数据成员和一些代码之间建立了明确的联系,
 * 定义方法就像在类的定义中加入函数一样简单, 方法可以访问类的所有成员.
 * 
 * ==== 访问控制
 * 访问控制可以限制类成员的访问, 公有和私有是两个主要的访问控制,
 * 任何人都可以访问公有成员, 但只有类自身可以访问其私有成员.
 * NOTE: 所有的 struct 成员默认都是公有的(包括成员变量和成员函数).
 * 
 * ==== 关键字class
 * class 关键字代替 struct 关键字,
 * class 关键字默认成员声明为 private, 除了默认的访问控制外
 * 用 struct 和 class 关键字声明的类是一样的.
 * 
 * ==== 初始化成员
 * ClockOfTheLongNowControl 有一个问题: 当 clock 被声明0时, year 是未初始化的;
 * 想保证 year 在任何情况下都不会小于2019, 
 * 这样的要求被称为类不变量: 一个总是真的类特性(也就是说它从不改变).
 * 在这个程序中,最终会进入一个良好的状态, 但通过构造函数可以做得更好,
 * 构造函数会初始化对象, 并在对象的生命周期之初就强制执行类不变量. 
 * 
 * ==== 构造函数 constructor
 * 构造函数是具有特殊声明的特殊方法, 
 * 构造函数声明不包含返回类型, 其名称与类的名称一致.
 * 该构造函数不需要参数, 并将 year 设置为2019.
 * 如果想用其他年份来初始化 ClockOfTheLongNowControl, 该怎么办?
 * 构造函数也可以接受任何数量的参数, 实现任意多的构造函数, 只要它们的参数类型不同(overload)
 * 
 * ==== 析构函数 deconstructor
 * 对象的析构函数是其清理函数, 析构函数在销毁对象之前被调用,
 * 析构函数几乎不会被明确地调用: 编译器将确保每个对象的析构函数在适当的时机被调用,
 * 可以在类的名称前加上 "~" 来声明该类的析构函数.
 * 析构函数的定义是可选的, 如果决定实现一个析构函数, 它不能接受任何参数.
 * 在析构函数中执行的操作包括释放文件句柄、刷新网络套接字(socket)和释放动态对象,
 * 如果没有定义析构函数, 编译器则会自动生成默认的析构函数， 默认析构函数的行为是不执行任何操作.
 * 
 */

struct ClockOfTheLongNow
{
    void add_year()
    {
        ++m_year;
    }

    int m_year;
};

class ClockOfTheLongNowControl
{
public:
    ClockOfTheLongNowControl()
        : m_year(2019)
    {
    }

    ClockOfTheLongNowControl(int year)
        : m_year(year)
    {
    }

    ~ClockOfTheLongNowControl()
    {
        m_year = 0;
        printf("the deconstructor and set year == %d\n", m_year);
    }

    void add_year()
    {
        ++m_year;
    }

    bool set_year(int year)
    {
        if (year < 2019)
            return false;
        m_year = year;
        return true;
    }

    int get_year()
    {
        return m_year;
    }

private:
    int m_year;
};

// -----------------------------------
int main(int argc, const char **argv)
{
    ClockOfTheLongNow clock;
    clock.m_year = 2023;
    clock.add_year();
    printf("year: %d\n", clock.m_year);
    clock.add_year();
    printf("year: %d\n", clock.m_year);

    printf("================================\n");
    ClockOfTheLongNowControl clock_control;
    // clock_control.set_year(2000);
    clock_control.set_year(2023);
    clock_control.add_year();
    printf("year: %d\n", clock_control.get_year());
    clock_control.add_year();
    printf("year: %d\n", clock_control.get_year());

    printf("================================\n");
    ClockOfTheLongNowControl clock_control_1;
    printf("year: %d\n", clock_control_1.get_year());
    clock_control_1.add_year();
    printf("year: %d\n", clock_control_1.get_year());

    ClockOfTheLongNowControl clock_control_2{2024};
    printf("year: %d\n", clock_control_2.get_year());
    clock_control_2.add_year();
    printf("year: %d\n", clock_control_2.get_year());

    printf("==== Object initialization or simply initialization ====\n");
    /**
     * @brief 1.将基本类型初始化为零
     * 基本类型的对象初始化为零:
     * 1. 使用字面量 0 明确地设置对象的值;
     * 2. 使用大括号 {};
     * 3. 使用等号=加大括号{}的方法;
     * 4. 声明对象时没有额外的符号是不可靠的, 它只在某些情况下有效.
     *
     * 不出所料, 使用大括号(初始化变量的方法被称为大括号初始化，
     * C++初始化语法如此混乱的部分原因是， 该语言从C语言(对象的生命周期是原始的)
     * 发展到具有健壮和丰富特性的对象生命周期的语言, 语言设计者在现代C++中加入了大括号初始化,
     * 以帮助在初始化语法中的各种尖锐冲突中平滑过渡，
     * 简而言之, 无论对象的作用域或类型如何,大括号初始化总是适用的, 而其他方法则不总适用.
     * 
     */
    int a = 0;  // initialized to 0
    int b{};    // initialized to 0
    int c = {}; // initialized to 0
    int d;      // initialized to 0(maybe)
    printf("the init value is: %d\n", a);
    printf("the init value is: %d\n", b);
    printf("the init value is: %d\n", c);
    printf("the init value is: %d\n", d);

    int e = 42;   // initialized to Arbitrary Value
    int f{42};    // initialized to Arbitrary Value
    int g = {42}; // initialized to Arbitrary Value
    int h(42);    // initialized to Arbitrary Value
    printf("the init value is: %d\n", e);
    printf("the init value is: %d\n", f);
    printf("the init value is: %d\n", g);
    printf("the init value is: %d\n", h);

    /**
     * @brief 初始化POD
     * 初始化POD的语法大多遵循基本类型的初始化语法
     * ! 警告 不能对POD使用 "等于0" 的初始化方法, 不会被编译通过, 因为它在语言规则中被明确禁止了
     * 
     * 将POD初始化为任意值
     * 可以使用括号内的初始化列表将字段初始化为任意值,
     * 大括号初始化列表中的参数必须与POD成员的类型相匹配,
     * 从左到右的参数顺序与从上到下的成员顺序一致, 任何省略的成员都被设置为零.
     * ! 警告 不能使用小括号来初始化POD, 代码将不会被编译:
     * PodStruct initialized pod(42,"Hello",true);
     * 
     */
    struct PodStruct
    {
        int  val;
        char str[256];
        bool cond;
    };

    PodStruct initialized_pod1{};                  // All fields zeroed
    PodStruct initialized_pod2 = {};               // All fields zeroed
    PodStruct initialized_pod3{42, "Hello"};       // Fields val & str set; cond = 0
    PodStruct initialized_pod4{42, "Hello", true}; // All fields set

    auto printInfo = [&](const PodStruct &pod)
    {
        printf("the member init value: %d, %s, %d\n", pod.val, pod.str, pod.cond);
    };
    printInfo(initialized_pod1);
    printInfo(initialized_pod2);
    printInfo(initialized_pod3);
    printInfo(initialized_pod4);

    // PodStruct initialized pod(42, "Hello", true); // !ERROR

    /**
     * @brief 初始化数组
     * 可以像初始化POD一样初始化数组, 数组声明和POD声明的主要区别是,
     * 数组指定了长度这个长度参数在方括号[]中,
     * 当使用大括号初始化列表来初始化数组时, 长度参数变得可有可无
     * 因为编译器可以从初始化列表参数的数量推断出长度参数.
     * 
     * ! 警告 array_4 是否被初始化实际上取决于与初始化基本类型相同的规则,
     * ! 对象的存储期决定了这些规则.
     * 
     */
    int array_1[]{1, 2, 3};  // Array of length 3; 1, 2, 3
    int array_2[5]{};        // Array of length 5; 0, 0, 0, 0, 0
    int array_3[5]{1, 2, 3}; // Array of length 5; 1, 2, 3, 0, 0
    int array_4[5];          // Array of length 5; uninitialized values

    auto printArray = [&](const int *arr, const int size)
    {
        printf("the member init value: ");
        for (size_t idx = 0; idx < size; ++idx)
        {
            printf(" %d ", arr[idx]);
        }
        printf("\n");
    };
    printArray(array_1, 3);
    printArray(array_2, 5);
    printArray(array_3, 5);
    printArray(array_4, 5); // *maybe random value

    return 0;
}
