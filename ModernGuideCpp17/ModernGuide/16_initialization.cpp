/**
 * @file 16_initialization.cpp
 * @author Wei Li (weili_yzzcq@163.com)
 * @brief 
 * @version 0.1
 * @date 2024-09-03
 * 
 * @copyright Copyright (c) 2024
 * 
 */

/** 关于C++中的变量,数组,对象等都有不同的初始化方法,在这些繁琐的初始化方法中没有任何一种方式适用于所有的情况.
 *  为了统一初始化方式,并且让初始化行为具有确定的效果,在C++11中提出了列表初始化的概念.
 * 
 * {} 是C++11中新添加的语法格式, 使用这种方式可以直接在变量名后边跟上初始化列表,来进行变量或者对象的初始化.
 * 既然使用列表初始化可以对普通类型以及对象进行直接初始化,
 * *那么在使用 new 操作符创建新对象的时候可以使用列表初始化进行对象的初始化吗
 * ?看出在C++11使用列表初始化是非常便利的,它统一了各种对象的初始化方式,而且还让代码的书写更加简单清晰.
 * 
 * 2. 列表初始化细节
 * * 聚合体
 * 因为如果使用列表初始化对对象初始化时，还需要判断这个对象对应的类型是不是一个聚合体，如果是初始化列表中的数据就会拷贝到对象中
 * ---普通数组本身可以看做是一个聚合类型;
 * ---满足以下条件的类（class、struct、union）可以被看做是一个聚合类型:
 *     ---无用户自定义的构造函数
 *     ---无私有或保护的非静态数据成员
 *     ---无基类
 *     ---无虚函数
 * * 非聚合体
 * 对于聚合类型的类可以直接使用列表初始化进行对象的初始化,
 * 如果不满足聚合条件还想使用列表初始化其实也是可以的,需要在类的内部自定义一个构造函数,
 * 在构造函数中使用初始化列表对类成员变量进行初始化.
 * ?聚合类型的定义并非递归的,也就是说当一个类的非静态成员是非聚合类型时,这个类也可能是聚合类型.
 * 对于一个聚合类型,使用列表初始化相当于对其中的每个元素分别赋值,而对于非聚合类型,
 * 则需要先自定义一个合适的构造函数,此时使用列表初始化将会调用它对应的构造函数.
 * 
 * 3. std::initializer_list
 * 在C++的STL容器中,可以进行任意长度的数据的初始化,
 * 使用初始化列表也只能进行固定参数的初始化,如果想要做到和STL一样有任意长度初始化的能力,
 * 可以使用std::initializer_list这个轻量级的类模板来实现.
 * 
 * *1. 它是一个轻量级的容器类型，内部定义了迭代器iterator等容器必须的概念，遍历时得到的迭代器是只读的;
 * *2. 对于std::initializer_list<T>而言，它可以接收任意长度的初始化列表，但是要求元素必须是同种类型T;
 * *3. 在std::initializer_list内部有三个成员接口：size(), begin(), end();
 * *4. std::initializer_list对象只能被整体初始化或者赋值;
 * 
 */

#include <iostream>
#include <string>
#include <vector>

class Person
{
public:
    Person(int id, std::string name)
    {
        std::cout << "id: " << id << ", name: " << name << std::endl;
    }
};

Person func()
{
    return {9527, "华安"};
}

// --------------------
struct T1
{
    int    x;
    double y;

    // 在构造函数中使用初始化列表初始化类成员
    T1(int a, double b, int c)
        : x(a)
        , y(b)
        , z(c)
    {
    }

    virtual void print()
    {
        std::cout << "x: " << x << ", y: " << y << ", z: " << z << std::endl;
    }

private:
    int z;
};

// 作为普通函数参数
// 如果想要自定义一个函数并且接收任意个数的参数（变参函数）
// 只需要将函数参数指定为std::initializer_list，使用初始化列表{ }作为实参进行数据传递即可
void traversal(std::initializer_list<int> a)
{
    for (auto it = a.begin(); it != a.end(); ++it)
    {
        std::cout << *it << " ";
    }
    std::cout << std::endl;
}

//  作为构造函数参数
// 自定义的类如果在构造对象的时候想要接收任意个数的实参,
// 可以给构造函数指定为std::initializer_list类型,在自定义类的内部还是使用容器来存储接收的多个实参.
class Test
{
public:
    Test(std::initializer_list<std::string> list)
    {
        for (auto it = list.begin(); it != list.end(); ++it)
        {
            std::cout << *it << " ";
            m_names.push_back(*it);
        }
        std::cout << std::endl;
    }

private:
    std::vector<std::string> m_names;
};

// -------------------------------------
int main(int argc, const char **argv)
{
    // 指针p指向了一个new操作符返回的内存，通过列表初始化将内存数据初始化为了520
    int *p = new int{520};
    std::cout << std::hex << p << " pointer into: " << *p << std::endl;

    // 变量b是对匿名对象使用列表初始之后，再进行拷贝初始化
    double b = double{52.134};
    std::cout << "The value is: " << b << std::endl;

    // 数组array在堆上动态分配了一块内存，通过列表初始化的方式直接完成了多个元素的初始化
    int *array = new int[3]{1, 2, 3};
    std::cout << "The value is: " << array[0] << std::endl;

    // 直接返回了一个匿名对象
    Person person = func();

    T1 t{520, 13.14, 1314}; // ok, 基于构造函数使用初始化列表初始化类成员
    t.print();

    // ====================================
    std::initializer_list<int> list;
    std::cout << "current list size: " << list.size() << std::endl;
    traversal(list);

    list = {1, 2, 3, 4, 5, 6, 7, 8, 9, 0};
    std::cout << "current list size: " << list.size() << std::endl;
    traversal(list);
    std::cout << std::endl;

    list = {1, 3, 5, 7, 9};
    std::cout << "current list size: " << list.size() << std::endl;
    traversal(list);
    std::cout << std::endl;

    // ==================== 直接通过初始化列表传递数据 ====================
    // std::initializer_list的效率是非常高的,
    // 它的内部并不负责保存初始化列表中元素的拷贝,仅仅存储了初始化列表中元素的引用.
    traversal({2, 4, 6, 8, 0});
    std::cout << std::endl;

    traversal({11, 12, 13, 14, 15, 16});
    std::cout << std::endl;

    // ==================== 直接通过初始化列表传递数据 ====================
    Test t_({"jack", "lucy", "tom"});
    Test t1_({"hello", "world", "ni_hao", "shi_jie"});

    return 0;
}
