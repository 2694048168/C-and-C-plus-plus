/**
 * @file 01_interface.cpp
 * @author Wei Li (weili_yzzcq@163.com)
 * @brief 
 * @version 0.1
 * @date 2024-04-30
 * 
 * @copyright Copyright (c) 2024
 * 
 */
#include <cstdio>

/**
 * @brief Interfaces 接口
 * 在软件工程中, 接口是不包含数据或代码的共享边界.
 * 它定义了接口的所有实现(implementation)都支持的函数签名, 
 * 而实现是支持接口的代码或数据.可以把接口看作实现接口的类和该类的用户(也称为消费者)之间的契约.
 * 消费者知道如何使用实现, 因为它们知道契约, 消费者从来都不需要知道实现的细节.
 * 接口有严格的要求, 接口的消费者只能使用接口中明确定义的方法.
 * 接口的使用可以促进高度可重用和松耦合的代码的产生.
 * 
 * ====== 实现接口
 * 要声明接口, 必须声明纯虚类, 要实现接口, 就要从它派生出来;
 * 因为接口是纯虚的, 所以它的实现必须实现接口的所有方法,
 * 最好的做法是用 override 关键字来标记这些方法, 这表示打算覆写虚函数, 让编译器避免简单错误.
 * ====== 使用接口
 * 作为消费者, 只能处理接口的引用或指针, 编译器无法提前知道要为底层类型分配多少内存,
 * 如果想让编译器知道底层类型, 那么最好使用模板.
 * 设置类成员的方法有两种:
 * 1. 构造函数注入: 对于构造函数注入, 通常需要使用接口引用,
 *    因为引用不能被重定位, 所以在对象的生命周期内不会改变;
 * 2. 属性注入: 对于属性注入, 可以使用方法来设置指针成员, 这允许改变该成员指向的对象;
 * 可以将这两种方法结合起来, 在构造函数中接受接口指针, 同时提供一个方法来将指针设置为其他东西.
 * 通常情况下, 当注入的字段在对象的整个生命周期内不会改变时, 可以使用构造函数注入,
 * 如果需要灵活地修改字段, 则应提供方法来执行属性注入.
 * 
 * Object Composition and Implementation Inheritance
 * ====== 对象组合和实现继承
 * 对象组合是一种设计模式, 在这种模式下, 类包含其他类类型的成员;
 * 另一种过时的设计模式叫作实现继承, 它实现了运行时多态, 
 * 实现继承允许建立类的层次结构, 每个子类都从其父类继承功能.
 * !Go和Rust(两种新的、越来越流行的系统编程语言)就不支持实现继承
 * 
 * ===== Defining Interfaces 定义接口
 * 不幸的是C++中没有 interface 关键字, 必须使用老式的继承机制来定义接口.
 * *需要理解 virtual, 虚析构函数, 纯虚方法, 基类继承, override.
 * 
 */

// interface
class Logger
{
public:
    // !must be virtual deconstructor
    virtual ~Logger() = default;

    // pure-virtual function
    virtual void log_transfer(long from, long to, double amount) = 0;
};

class ConsoleLogger : Logger
{
    void log_transfer(long from, long to, double amount) override
    {
        printf("[console] %ld -> %ld: %f\n", from, to, amount);
    }
};

/**
 * @brief Base Class Inheritance 基类继承
 * 定义继承关系, 可以在冒号(:)后面加上基类的名称 BaseClass
 * 派生类可以像其他类一样被声明, 可以把派生类引用当作基类引用类型来使用.
 * 从类派生子类的主要原因是想继承它的成员
 * 
 * ==== Member Inheritance 成员继承
 * 派生类从基类中继承非私有成员, 类可以像使用普通成员一样使用继承的成员;
 * 成员继承的好处是, 只需在基类中定义一次功能, 而不必在派生类中重复定义;
 * 不幸的是, 多年的经验使编程界的许多人认为要避免成员继承, 因为与基于组合的多态相比
 * 它很容易产生脆弱的, 难以理解的代码(这就是许多现代编程语言(Go and Rust)排除了它的原因).
 * 
 * ==== 虚方法 virtual Methods
 * 如果想让派生类覆盖基类的方法, 可以使用 virtual 关键字,
 * 通过在方法的定义中添加 virtual 声明如果存在派生类的实现, 就应该使用它,
 * 在实现中将 override 关键字添加到方法的声明中, 编译器检查是否和基类方法签名一致.
 * 
 * ==== 存虚函数  pure virtual methods
 * 如果想要求派生类来实现该方法, 则可以在方法定义中添加 =0 后缀,
 * 可以将同时使用virtual 关键字和 =0 后缀的方法称为纯虚方法, 
 * !包含任何纯虚方法的类都不能实例化
 *  
 */
class BaseClass
{
public:
    int the_answer() const
    {
        return 42;
    }

    void printInfo()
    {
        printf("the memory: %s and %s\n", member, holistic_detective);
    }

    virtual const char *final_message() const
    {
        return "We apologize for the incontinence.";
    }

public:
    const char *member = "gold";

private:
    const char *holistic_detective = "Dirk Gently";
};

struct DerivedClass : BaseClass
{
    const char *final_message() const override
    {
        return "We apologize for the inconvenience.";
    }
};

struct BaseClassPure
{
    virtual const char *final_message() const = 0;
};

struct DerivedPure : BaseClassPure
{
    const char *final_message() const override
    {
        return "We apologize for the inconvenience.";
    }
};

void are_belong_to_us(BaseClass &base) {}

/** Pure-Virtual Classes and Virtual Destructors
 * @brief 纯虚类和虚析构函数
 * 通过从只包含纯虚方法的基类派生来实现接口继承, 这种类被称为纯虚类;
 * 在C++中, 接口总是纯虚类, 通常需要在接口中添加虚析构函数;
 * 在某些罕见的情况下, 如果没有把析构函数标记为虚函数, 就有可能泄漏资源d
 * 
 * ===== 在声明接口时, 声明虚析构函数是可选的, 
 * 但是要注意, 如果忘记了已在接口中实现虚析构函而不小心做了, 可能会泄漏资源, 而编译器不会发出警告.
 * NOTE: 与其声明公有虚析构函数, 不如声明受保护的非虚析构函数, 因为当编写删除基类指针的代码时, 会引起编译错误.
 * 有些人不喜欢这种方法, 因为最终还是要声明一个有公有析构函数的类, 而如果从这个类派生其他类, 仍然会遇到同样的问题.
 * 
 */
class BaseClassInterface
{
public:
    virtual ~BaseClassInterface() = default;
};

struct DerivedClassInterface : BaseClassInterface
{
    DerivedClassInterface()
    {
        printf("DerivedClass() invoked.\n");
    }

    ~DerivedClassInterface()
    {
        printf("~DerivedClass() invoked.\n");
    }
};

// ------------------------------------
int main(int argc, const char **argv)
{
    printf("=============================\n");
    DerivedClass x;
    printf("The answer is %d\n", x.the_answer());
    printf("%s member\n", x.member);
    // !This line doesn't compile:
    // printf("%s's Holistic Detective Agency\n", x.holistic_detective);
    x.printInfo();

    BaseClass    base;
    DerivedClass derived;
    BaseClass   &ref = derived;

    printf("=============================\n");
    printf("BaseClass: %s\n", base.final_message());
    printf("DerivedClass: %s\n", derived.final_message());
    printf("BaseClass&: %s\n", ref.final_message());

    printf("=============================\n");
    // BaseClassPure  base_; // !ERROR
    DerivedPure    derived_;
    BaseClassPure &ref_ = derived_;
    printf("DerivedClass: %s\n", derived_.final_message());
    printf("BaseClass&: %s\n", ref_.final_message());

    /**
     * @brief 注意 虚函数可能会产生运行时开销,
     * 尽管开销通常很低(在普通函数调用的25%以内),编译器会生成包含函数指针的虚函数表(vtable);
     * 在运行时, 接口的消费者一般不知道它的底层类型, 但它知道如何调用接口的方法(多亏了vtable);
     * 在某些情况下, 链接器可以检测到接口的所有用法并将函数调用去虚化,
     * 这就从vtable中删除了函数调用, 从而消除了相关的运行时开销.
     * 
     */
    printf("Constructing DerivedClass x.\n");
    BaseClassInterface *p_interface{new DerivedClassInterface{}};
    printf("Deleting x as a BaseClass*.\n");
    delete p_interface;

    return 0;
}
