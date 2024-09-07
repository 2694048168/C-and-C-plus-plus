/**
 * @file 24_shared_pointer.cpp
 * @author Wei Li (weili_yzzcq@163.com)
 * @brief 
 * @version 0.1
 * @date 2024-09-05
 * 
 * @copyright Copyright (c) 2024
 * 
 */

/** 在C++中没有垃圾回收机制,必须自己释放分配的内存,否则就会造成内存泄露.
 * 解决这个问题最有效的方法是使用智能指针(smart pointer).
 * 智能指针是存储指向动态分配(堆)对象指针的类,用于生存期的控制,能够确保在离开指针所在作用域时，
 * 自动地销毁动态分配的对象,防止内存泄露.
 * *智能指针的核心实现技术是引用计数,每使用它一次,内部引用计数加1;
 * *每析构一次内部的引用计数减1,减为0时,删除所指向的堆内存.
 * 
 * C++11中提供了三种智能指针,使用这些智能指针时需要引用头文件<memory>:
 * *1. std::shared_ptr: 共享的智能指针
 * *2. std::unique_ptr: 独占的智能指针
 * *3. std::weak_ptr: 弱引用的智能指针,它不共享指针,不能操作资源,是用来监视shared_ptr的
 *  
 * 1. shared_ptr的初始化
 * ?共享智能指针是指多个智能指针可以同时管理同一块有效的内存,
 * ?共享智能指针shared_ptr 是一个模板类,如果要进行初始化有三种方式:
 * 通过构造函数、std::make_shared辅助函数以及reset方法.
 * 共享智能指针对象初始化完毕之后就指向了要管理的那块堆内存,
 * 如果想要查看当前有多少个智能指针同时管理着这块内存可以使用共享智能指针提供的一个成员函数use_count.
 * 
 * 2. 指定删除器
 * 当智能指针管理的内存对应的引用计数变为0的时候,这块内存就会被智能指针析构掉了.
 * 另外在初始化智能指针的时候也可以自己指定删除动作, 这个删除操作对应的函数被称之为删除器,
 * ?这个删除器函数本质是一个回调函数, 只需要进行实现,其调用是由智能指针完成的.
 * 
 */

#include <cstring>
#include <iostream>
#include <memory>
#include <string>

class Test
{
public:
    Test()
    {
        std::cout << "construct Test..." << std::endl;
    }

    Test(int x)
    {
        std::cout << "construct Test, x = " << x << std::endl;
    }

    Test(std::string str)
    {
        std::cout << "construct Test, str = " << str << std::endl;
    }

    ~Test()
    {
        std::cout << "destruct Test ..." << std::endl;
    }
};

// 自定义删除器函数，释放int型内存
void deleteIntPtr(int *p)
{
    delete p;
    std::cout << "int 型内存被释放了...\n";
}

// 还可以自己封装一个make_shared_array方法来让shared_ptr支持数组
template<typename T>
std::shared_ptr<T> make_share_array(size_t size)
{
    // 返回匿名对象
    return std::shared_ptr<T>(new T[size], std::default_delete<T[]>());
}

// ---------------------------------------
int main(int argc, const char **argv)
{
    // *====================== 通过构造函数初始化
    // 使用智能指针管理一块 int 型的堆内存
    std::shared_ptr<int> ptr1(new int(520));
    std::cout << "ptr1管理的内存引用计数: " << ptr1.use_count() << std::endl;

    // 使用智能指针管理一块字符数组对应的堆内存
    std::shared_ptr<char> ptr2(new char[12]);
    std::cout << "ptr2管理的内存引用计数: " << ptr2.use_count() << std::endl;

    // 创建智能指针对象, 不管理任何内存
    std::shared_ptr<int> ptr3;
    std::cout << "ptr3管理的内存引用计数: " << ptr3.use_count() << std::endl;

    // 创建智能指针对象, 初始化为空
    std::shared_ptr<int> ptr4(nullptr);
    std::cout << "ptr4管理的内存引用计数: " << ptr4.use_count() << std::endl;

    // !不要使用一个原始指针初始化多个shared_ptr
    int                 *p = new int;
    std::shared_ptr<int> p1(p);
    std::shared_ptr<int> p2(p); // !error, 编译不会报错, 运行会出错
    std::cout << "不要使用一个原始指针初始化多个shared_ptr\n";

    // *====================== 通过拷贝和移动构造函数初始化
    // 当一个智能指针被初始化之后,就可以通过这个智能指针初始化其他新对象.
    // 在创建新对象的时候,对应的拷贝构造函数或者移动构造函数就被自动调用了.
    // ?如果使用拷贝的方式初始化共享智能指针对象,这两个对象会同时管理同一块堆内存,堆内存对应的引用计数也会增加;
    // ?如果使用移动的方式初始智能指针对象,只是转让了内存的所有权,管理内存的对象并不会增加,因此内存的引用计数不会变化;

    // 使用智能指针管理一块 int 型的堆内存, 内部引用计数为 1
    std::shared_ptr<int> ptr1_(new int(520));
    std::cout << "ptr1管理的内存引用计数: " << ptr1_.use_count() << std::endl;

    //调用拷贝构造函数
    std::shared_ptr<int> ptr2_(ptr1_);
    std::cout << "ptr2管理的内存引用计数: " << ptr2_.use_count() << std::endl;
    std::shared_ptr<int> ptr3_ = ptr1_;
    std::cout << "ptr3管理的内存引用计数: " << ptr3_.use_count() << std::endl;

    //调用移动构造函数
    std::shared_ptr<int> ptr4_(std::move(ptr1_));
    std::cout << "ptr4管理的内存引用计数: " << ptr4_.use_count() << std::endl;
    std::shared_ptr<int> ptr5_ = std::move(ptr2_);
    std::cout << "ptr5管理的内存引用计数: " << ptr5_.use_count() << std::endl;

    // *====================== 通过std::make_shared初始化
    // template< class T, class... Args >
    // shared_ptr<T> make_shared( Args&&... args );
    //? Args&&... args ：要初始化的数据，如果是通过make_shared创建对象，需按照构造函数的参数列表指定
    // 通过C++提供的std::make_shared() 就可以完成内存对象的创建并将其初始化给智能指针
    // 使用std::make_shared()模板函数可以完成内存地址的创建,
    // 并将最终得到的内存地址传递给共享智能指针对象管理.
    // 如果申请的内存是普通类型,通过函数的（）可完成地址的初始化;
    // 如果要创建一个类对象, 函数的（）内部需要指定构造对象需要的参数,也就是类构造函数的参数;

    // 使用智能指针管理一块 int 型的堆内存, 内部引用计数为 1
    std::shared_ptr<int> ptr1__ = std::make_shared<int>(520);
    std::cout << "ptr1管理的内存引用计数: " << ptr1__.use_count() << std::endl;

    std::shared_ptr<Test> ptr2__ = std::make_shared<Test>();
    std::cout << "ptr2管理的内存引用计数: " << ptr2__.use_count() << std::endl;

    std::shared_ptr<Test> ptr3__ = std::make_shared<Test>(520);
    std::cout << "ptr3管理的内存引用计数: " << ptr3__.use_count() << std::endl;

    std::shared_ptr<Test> ptr4__ = std::make_shared<Test>("我是要成为海贼王的男人!!!");
    std::cout << "ptr4管理的内存引用计数: " << ptr4__.use_count() << std::endl;

    // *====================== 通过 reset方法初始化
    // 对于一个未初始化的共享智能指针,可以通过reset方法来初始化;
    // 当智能指针中有值的时候,调用reset会使引用计数减1;

    // 使用智能指针管理一块 int 型的堆内存, 内部引用计数为 1
    std::shared_ptr<int> _ptr1 = std::make_shared<int>(520);
    std::shared_ptr<int> _ptr2 = ptr1;
    std::shared_ptr<int> _ptr3 = ptr1;
    std::shared_ptr<int> _ptr4 = ptr1;
    std::cout << "ptr1管理的内存引用计数: " << _ptr1.use_count() << std::endl;
    std::cout << "ptr2管理的内存引用计数: " << _ptr2.use_count() << std::endl;
    std::cout << "ptr3管理的内存引用计数: " << _ptr3.use_count() << std::endl;
    std::cout << "ptr4管理的内存引用计数: " << _ptr4.use_count() << std::endl;

    _ptr4.reset();
    std::cout << "ptr1管理的内存引用计数: " << _ptr1.use_count() << std::endl;
    std::cout << "ptr2管理的内存引用计数: " << _ptr2.use_count() << std::endl;
    std::cout << "ptr3管理的内存引用计数: " << _ptr3.use_count() << std::endl;
    std::cout << "ptr4管理的内存引用计数: " << _ptr4.use_count() << std::endl;

    std::shared_ptr<int> _ptr5;
    _ptr5.reset(new int(250));
    std::cout << "ptr5管理的内存引用计数: " << _ptr5.use_count() << std::endl;

    // *====================== 获取原始指针
    // 通过智能指针可以管理一个普通变量或者对象的地址,此时原始地址就不可见了;
    // 当想要修改变量或者对象中的值的时候,就需要从智能指针对象中先取出数据的原始内存的地址再操作,
    // 解决方案是调用共享智能指针类提供的get()方法
    int                   len = 128;
    std::shared_ptr<char> ptr(new char[len]);
    // 得到指针的原始地址
    char                 *add = ptr.get();
    std::memset(add, 0, len);
    std::strcpy(add, "我是要成为海贼王的男人!!!");
    std::cout << "string: " << add << std::endl;

    std::shared_ptr<int> _p(new int);
    *_p = 100;
    std::cout << *_p.get() << "  " << *_p << std::endl;

    // *====================== 指定删除器
    std::shared_ptr<int> __ptr(new int(250), deleteIntPtr);

    // 删除器函数也可以是lambda表达式
    std::shared_ptr<int> __ptr__(new int(250),
                                 [](int *p)
                                 {
                                     delete p;
                                     std::cout << "int 型内存被释放了... fom lambda expression\n";
                                 });

    // *在C++11中使用shared_ptr管理动态数组时, 需要指定删除器,
    // 因为std::shared_ptr的默认删除器不支持数组对象
    std::shared_ptr<int> _ptr_(new int[10],
                               [](int *p)
                               {
                                   delete[] p;

                                   std::cout << "int 型数组内存被释放了... fom lambda expression\n";
                               });

    // 在删除数组内存时, 除了自己编写删除器,
    // 也可以使用C++提供的std::default_delete<T>()函数作为删除器,
    // 这个函数内部的删除功能也是通过调用delete来实现的,要释放什么类型的内存就将模板类型T指定为什么类型即可
    std::shared_ptr<int> _ptr__(new int[10], std::default_delete<int[]>());

    // 还可以自己封装一个make_shared_array方法来让shared_ptr支持数组
    std::shared_ptr<int> __ptr1__ = make_share_array<int>(10);
    std::cout << __ptr1__.use_count() << std::endl;
    std::shared_ptr<char> __ptr2__ = make_share_array<char>(128);
    std::cout << __ptr2__.use_count() << std::endl;

    return 0;
}
