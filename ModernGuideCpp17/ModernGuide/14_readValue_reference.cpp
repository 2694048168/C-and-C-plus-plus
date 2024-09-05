/**
 * @file 14_readValue_reference.cpp
 * @author Wei Li (weili_yzzcq@163.com)
 * @brief 
 * @version 0.1
 * @date 2024-09-02
 * 
 * @copyright Copyright (c) 2024
 * 
 */

/** lvalue 是locator value的缩写; rvalue 是 read value的缩写;
 * *左值是指存储在内存中、有明确存储地址（可取地址）的数据;
 * *右值是指可以提供数据值的数据（不可取地址）;
 * ?C++11 增加了一个新的类型, 称为右值引用(R-value reference),标记为 &&;
 * 区分左值与右值的便捷方法是: 可以对表达式取地址（&）就是左值,否则为右值.
 * 
 * C++11 中右值可以分为两种: 一个是将亡值(xvalue, expiring value);另一个则是纯右值(prvalue, PureRvalue);
 * !纯右值：非引用返回的临时变量、运算表达式产生的临时变量、原始字面量和 lambda 表达式等;
 * !将亡值：与右值引用相关的表达式，比如，T&&类型函数的返回值、 std::move 的返回值等;
 * 
 * 右值引用就是对一个右值进行引用的类型, 因为右值是匿名的, 所以只能通过引用的方式找到它.
 * 无论声明左值引用还是右值引用都必须立即进行初始化, 因为引用类型本身并不拥有所绑定对象的内存,
 * 只是该对象的一个别名. 通过右值引用的声明, 右值又"重获新生",
 * 其生命周期与右值引用类型变量的生命周期一样, 只要该变量还活着, 右值临时量将会一直存活下去.
 * 
 * ====2. 性能优化
 * 在C++中在进行对象赋值操作的时候, 很多情况下会发生对象之间的深拷贝, 如果堆内存很大,
 * 这个拷贝的代价也就非常大, 在某些情况下, 如果想要避免对象的深拷贝, 就可以使用右值引用进行性能的优化.
 * *对于需要动态申请大量资源的类,应该设计移动构造函数,以提高程序效率;
 * *需要注意的是,一般在提供移动构造函数的同时,也会提供常量左值引用的拷贝构造函数,以保证移动不成还可以使用拷贝构造函数. 
 * 
 * ====3 && 的特性
 * 在C++中,并不是所有情况下 && 都代表是一个右值引用,具体的场景体现在模板和自动类型推导中,
 * 如果是模板参数需要指定为T&&, 如果是自动类型推导需要指定为auto &&,
 * 在这两种场景下 &&被称作未定的引用类型.
 * ?另外还有一点需要额外注意const T&&表示一个右值引用,不是未定引用类型.
 * 
 * !最后总结一下关于&&的使用：
 * 1. 左值和右值是独立于他们的类型的，右值引用类型可能是左值也可能是右值;
 * 2. 编译器会将已命名的右值引用视为左值，将未命名的右值引用视为右值;
 * 3. auto&&或者函数参数类型自动推导的T&&是一个未定的引用类型，它可能是左值引用也可能是右值引用类型，
 * 这取决于初始化的值类型;\
 * 4. 通过右值推导 T&& 或者 auto&& 得到的是一个右值引用类型，其余都是左值引用类型.
 * 
 */

#include <iostream>

// value 是对纯右值 '520' 的引用, 即value是右值引用
int &&value = 520;

class Test
{
public:
    Test()
    {
        std::cout << "construct: my name is jerry\n";
    }

    Test(const Test &a)
    {
        std::cout << "copy construct: my name is tom\n";
    }
};

Test getObj()
{
    // 返回的临时对象被称之为将亡值
    return Test();
}

// ============== 2. 性能优化 ==============
class TestPerformance
{
public:
    TestPerformance()
        : m_pNum(new int{100})
    {
        std::cout << "construct: my name is jerry\n";
    }

    TestPerformance(const TestPerformance &a)
        : m_pNum(new int(*a.m_pNum))
    {
        std::cout << "copy construct: my name is tom\n";
    }

    ~TestPerformance()
    {
        if (m_pNum)
        {
            delete m_pNum;
            m_pNum = nullptr;
        }
    }

    int *m_pNum = nullptr;
};

TestPerformance getObjPerformance()
{
    TestPerformance t;
    // VS2019等未做优化前,这里会发生调用拷贝构造函数对返回的临时对象进行了深拷贝得到了对象t
    // VS2022后做优化, 看不到拷贝构造函数的调用,
    return t;

    /*  通过输出的结果可以看到调用Test t = getObj(); 的时候调用
    拷贝构造函数对返回的临时对象进行了深拷贝得到了对象t，
    在getObj()函数中创建的对象虽然进行了内存的申请操作,但是没有使用就释放掉了;
    如果能够使用临时对象已经申请的资源,既能节省资源,还能节省资源申请和释放的时间(特别是堆内存的申请和释放),
    ===== malloc(new) free(delete) 涉及到内核态和用户态的切换,导致CPU资源的占用=====
    如果要执行这样的操作就需要使用右值引用了, 右值引用具有移动语义,
    移动语义可以将资源（堆、系统对象等）通过浅拷贝从一个对象转移到另一个对象
    这样就能减少不必要的临时对象的创建、拷贝以及销毁，可以大幅提高C++应用程序的性能.
     */
}

class TestPerformanceMove
{
public:
    TestPerformanceMove()
        : m_pNum(new int{100})
    {
        std::cout << "construct: my name is jerry\n";
    }

    TestPerformanceMove(const TestPerformance &a)
        : m_pNum(new int(*a.m_pNum))
    {
        std::cout << "copy construct: my name is tom\n";
    }

    // 添加移动构造函数
    TestPerformanceMove(TestPerformance &&a)
        : m_pNum(a.m_pNum)
    {
        a.m_pNum = nullptr;
        std::cout << "move construct: my name is sunny\n";
    }

    ~TestPerformanceMove()
    {
        if (m_pNum)
        {
            delete m_pNum;
            m_pNum = nullptr;
        }
    }

    int *m_pNum = nullptr;
};

TestPerformanceMove getObjPerformanceMove()
{
    TestPerformanceMove t;
    return t;

    /* 移动构造函数(参数为右值引用类型)，这样在进行Test t = getObj(); 
    操作的时候并没有调用拷贝构造函数进行深拷贝，而是调用了移动构造函数，
    在这个函数中只是进行了浅拷贝，没有对临时对象进行深拷贝，提高了性能.
    在测试程序中getObj()的返回值就是一个将亡值，也就是说是一个右值，
    在进行赋值操作的时候如果=右边是一个右值，那么移动构造函数就会被调用。
    移动构造中使用了右值引用，会将临时对象中的堆内存地址的所有权转移给对象t，
    这块内存被成功续命，因此在t对象中还可以继续使用这块内存.
  */
}

// ======== 3 && 特殊性
template<typename T>
void f(T &&param)
{
    std::cout << "void f(T &&param) the value is: " << param << std::endl;
}

template<typename T>
void f1(const T &&param)
{
    std::cout << "void f1(const T &&param) the value is: " << param << std::endl;
}

// ----------------------------
void printValue(int &i)
{
    std::cout << "l-value: " << i << std::endl;
}

void printValue(int &&i)
{
    std::cout << "r-value: " << i << std::endl;
}

void forwardValue(int &&k)
{
    printValue(k);
}

// -----------------------------------
int main(int argc, const char **argv)
{
    int a1 = 42;
    // 使用左值初始化一个右值引用类型是不合法的
    // int &&a2 = a1; // error

    // 右值不能给普通的左值引用赋值
    // Test       &t = getObj(); // error

    // t是这个将亡值的右值引用
    Test &&t = getObj();

    // 常量左值引用是一个万能引用类型，它可以接受左值、右值、常量左值和常量右值
    const Test &t_const = getObj();

    // ============== 2. 性能优化 ==============
    std::cout << "=======================================\n";
    TestPerformance t_performance = getObjPerformance();
    std::cout << "t.m_num: " << *t_performance.m_pNum << std::endl;
    /* 当时使用的vs版本为2019，vs2022已无法看到相同的输出，代码被优化了
    construct: my name is jerry
    copy construct: my name is tom
    t.m_num: 100
    */

    // ============== 2. 性能优化 ==============
    std::cout << "=======================================\n";
    TestPerformanceMove t_performance_move = getObjPerformanceMove();
    std::cout << "t.m_num: " << *t_performance_move.m_pNum << std::endl;
    /* 当时使用的vs版本为2019，vs2022已无法看到相同的输出，代码被优化了
    construct: my name is jerry
    move construct: my name is sunny
    destruct Test class ...
    t.m_num: 100
    destruct Test class ...
     */

    // ============== 3. && 特殊性 ==============
    // 函数模板进行了自动类型推导，需要通过传入的实参来确定参数param的实际类型
    f(10); // 传入的实参10是右值，因此T&&表示右值引用
    int x = 10;
    f(x); // 传入的实参是x是左值，因此T&&表示左值引用

    // f1(x)的参数是const T&&不是未定引用类型，不需要推导，本身就表示一个右值引用
    // f1(x);  // error, x是左值
    f1(10); // ok, 10是右值

    // ===============================
    int    x__ = 520, y = 1314;
    auto &&v1 = x__; // auto&&表示一个整形的左值引用
    auto &&v2 = 250; // auto&&表示一个整形的右值引用

    // decltype(x)&&等价于int&&是一个右值引用不是未定引用类型，y是一个左值
    // 不能使用左值初始化一个右值引用类型
    // decltype(x) &&v3 = y; // !error

    std::cout << "v1: " << v1 << ", v2: " << v2 << std::endl;
    /* 由于上述代码中存在T&&或者auto&&这种未定引用类型,当它作为参数时,
    有可能被一个右值引用初始化，也有可能被一个左值引用初始化，
    在进行类型推导时右值引用类型（&&）会发生变化,这种变化被称为引用折叠.
    在C++11中引用折叠的规则如下：
    1. 通过右值推导 T&& 或者 auto&& 得到的是一个右值引用类型;
    2. 通过非右值(右值引用、左值、左值引用、常量右值引用、常量左值引用)推导 T&& 或者 auto&& 得到的是一个左值引用类型;
     */

    // =====================
    int i = 520;
    printValue(i);
    printValue(1314);

    /* 编译器会根据传入的参数的类型(左值还是右值)调用对应的重置函数(printValue),
    函数forwardValue()接收的是一个右值, 但是在这个函数中调用函数printValue()时,
    参数k变成了一个命名对象,编译器会将其当做左值来处理. */
    forwardValue(250);

    return 0;
}
