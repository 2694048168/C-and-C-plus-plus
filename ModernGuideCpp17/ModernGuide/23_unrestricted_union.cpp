/**
 * @file 23_unrestricted_union.cpp
 * @author Wei Li (weili_yzzcq@163.com)
 * @brief 
 * @version 0.1
 * @date 2024-09-04
 * 
 * @copyright Copyright (c) 2024
 * 
 */

/** 联合体又叫共用体,将其称之为union,它的使用方式和结构体类似,
 * 程序猿可以在联合体内部定义多种不同类型的数据成员,
 * 但是这些数据会共享同一块内存空间(也就是如果对多个数据成员同时赋值会发生数据的覆盖).
 * *在某些特定的场景下,通过这种特殊的数据结构可以实现内存的复用,从而达到节省内存空间的目的.
 *
 * 在C++11之前使用的联合体是有局限性的,主要有以下三点:
 * ?1. 不允许联合体拥有非POD类型的成员;
 * ?2. 不允许联合体拥有静态成员;
 * ?3. 不允许联合体拥有引用类型的成员;
 * 在新的C++11标准中,取消了关于联合体对于数据成员类型的限定,
 * 规定任何非引用类型都可以成为联合体的数据成员, 这样的联合体称之为非受限联合体(Unrestricted Union)
 * 
 * 2. 非受限联合体的使用
 * ----对于非受限联合体来说,静态成员有两种分别是静态成员变量和静态成员函数;
 * ----非POD类型成员;
 *    ----在 C++11标准中会默认删除一些非受限联合体的默认函数.
 *    ----非受限联合体有一个非 POD 的成员, 而该非 POD成员类型拥有 非平凡的构造函数
 *    ----那么非受限联合体的默认构造函数将被编译器删除.
 * 其他的特殊成员函数,例如默认拷贝构造函数、拷贝赋值操作符以及析构函数等，也将遵从此规则.
 * ?在定义构造函数的时候我们需要用到定位放置 new操作
 * 
 * *---placement new
 * 一般情况下,使用new申请空间时,是从系统的堆(heap)中分配空间,
 * 申请所得的空间的位置是根据当时的内存的实际使用情况决定的.
 * 但是在某些特殊情况下,可能需要在已分配的特定内存创建对象,这种操作就叫做placement new即定位放置 new.
 * 
 * -----自定义非受限联合体构造函数
 * -----匿名的非受限联合体
 * 
 */

#include <iostream>

union Test
{
    int  age;
    long id;

    // 语法错误，非受限联合体中不允许出现引用类型
    // int& tmp = age; // !error

    static char c;

    // 在静态函数print()只能访问非受限联合体Test中的静态变量，对于非静态成员变量（age、id）是无法访问的。
    static int print()
    {
        std::cout << "c value: " << c << std::endl;
        // std::cout << "id value: " << id << std::endl;
        // std::cout << "id value: " << age << std::endl;
        return 0;
    }
};

// 非受限联合体中的静态成员变量,需要在非受限联合体外部声明或者初始化之后才能使用
// char Test::c = 'a';
char Test::c;

class Base
{
public:
    Base() {}

    ~Base() {}

    void print()
    {
        std::cout << "number value: " << number << std::endl;
    }

private:
    int number;
};

// 自定义非受限联合体构造函数
class BaseCustom
{
public:
    void setText(std::string str)
    {
        notes = str;
    }

    void print()
    {
        std::cout << "BaseCustom notes: " << notes << std::endl;
    }

private:
    std::string notes;
};

union Student
{
    // 给非受限制联合体显示的指定了构造函数和析构函数
    Student()
    {
        // 通过定位放置 new的方式将构造出的对象地址定位到了联合体的成员string name的地址上了,
        // 这样联合体内部其他非静态成员也就可以访问这块地址了
        new (&name) std::string;
    }

    ~Student() {}

    int         id;
    BaseCustom  tmp;
    std::string name;
};

//-----匿名的非受限联合体
// 一般情况下使用的非受限联合体都是具名的(有名字),
// 但是也可以定义匿名的非受限联合体,一个比较实用的场景就是配合着类的定义使用.
// 外来人口信息
struct Foreigner
{
    Foreigner(std::string s, std::string ph)
        : addr(s)
        , phone(ph)
    {
    }

    std::string addr;
    std::string phone;
};

// 登记人口信息
class Person
{
public:
    enum class Category : char
    {
        Student,
        Local,
        Foreign
    };

    Person(int num)
        : number(num)
        , type(Category::Student)
    {
    }

    Person(std::string id)
        : idNum(id)
        , type(Category::Local)
    {
    }

    Person(std::string addr, std::string phone)
        : foreign(addr, phone)
        , type(Category::Foreign)
    {
    }

    ~Person() {}

    void print()
    {
        std::cout << "Person category: " << (int)type << std::endl;
        switch (type)
        {
        case Category::Student:
            std::cout << "Student school number: " << number << std::endl;
            break;
        case Category::Local:
            std::cout << "Local people ID number: " << idNum << std::endl;
            break;
        case Category::Foreign:
            std::cout << "Foreigner address: " << foreign.addr << ", phone: " << foreign.phone << std::endl;
            break;
        default:
            break;
        }
    }

private:
    Category type;

    // 匿名的非受限联合体用来存储人口信息,
    // 仔细分析之后就会发现这种处理方式的优势非常明显: 尽可能地节省了内存空间.
    union
    {
        int         number;
        std::string idNum;
        Foreigner   foreign;
    };
};

// ------------------------------------
int main(int argc, const char **argv)
{
    // t和t1对象共享这个静态成员变量（和类 class/struct 中的静态成员变量的使用是一样的）
    Test t;
    Test t1;

    t.c    = 'b';
    t1.c   = 'c';
    t1.age = 666;
    std::cout << "t.c: " << t.c << std::endl;

    // !在非受限联合体中静态成员变量和非静态成员变量使用的不是同一块内存
    std::cout << "t1.c: " << t1.c << std::endl;
    std::cout << "t1.age: " << t1.age << std::endl;
    std::cout << "t1.id: " << t1.id << std::endl;

    // 非受限联合体中的静态成员函数
    // 调用这个静态方法可以通过对象,也可以通过类名实现
    t.print();
    Test::print();

    //---placement new
    int n = 1024;

    //使用定位放置的方式为指针b申请了一块内存,
    // 也就是说此时指针 b指向的内存地址和变量 n对应的内存地址是同一块(栈内存)
    // 而在Base类中成员变量 number的起始地址和Base对象的起始地址是相同的
    Base *b = new (&n) Base;
    b->print();

    /* 关于placement new的一些细节:
    * 1. 使用定位放置new操作,既可以在栈(stack)上生成对象,也可以在堆(heap)上生成对象,这取决于定位时指定的内存地址是在堆还是在栈上;
    * 2. 从表面上看,定位放置new操作是申请空间,其本质是利用已经申请好的空间,真正的申请空间的工作是在此之前完成的;
    * 3. 使用定位放置new 创建对象时会自动调用对应类的构造函数,但是由于对象的空间不会自动释放,如果需要释放堆内存必须显示调用类的析构函数;
    * 4. 使用定位放置new操作,可以反复动态申请到同一块堆内存,这样可以避免内存的重复创建销毁,从而提高程序的执行效率(比如网络通信中数据的接收和发送);
     */
    // 自定义非受限联合体构造函数
    Student s;
    s.name = "蒙奇·D·路飞";
    s.tmp.setText("我是要成为海贼王的男人!");
    s.tmp.print();
    std::cout << "Student name: " << s.name << std::endl;

    //-----匿名的非受限联合体
    Person p1(9527);
    Person p2("1101122022X");
    Person p3("砂隐村村北", "1301810001");
    p1.print();
    p2.print();
    p3.print();

    return 0;
}
