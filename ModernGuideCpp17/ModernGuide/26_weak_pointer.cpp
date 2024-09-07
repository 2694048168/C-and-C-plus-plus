/**
 * @file 26_weak_pointer.cpp
 * @author Wei Li (weili_yzzcq@163.com)
 * @brief 
 * @version 0.1
 * @date 2024-09-07
 * 
 * @copyright Copyright (c) 2024
 * 
 */

/** 1. 基本使用方法
 * 弱引用智能指针std::weak_ptr可以看做是shared_ptr的助手,
 * 它不管理shared_ptr内部的指针, std::weak_ptr没有重载操作符*和->, 因为它不共享指针,不能操作资源,
 * ?所以它的构造不会增加引用计数,析构也不会减少引用计数,
 * *它的主要作用就是作为一个旁观者监视shared_ptr中管理的资源是否存在.
 * 
 * -----初始化
 * 1. 通过调用std::weak_ptr类提供的use_count()方法可以获得当前所观测资源的引用计数;
 * 2. 通过调用std::weak_ptr类提供的expired()方法来判断观测的资源是否已经被释放;
 * 3. 当共享智能指针调用shared.reset();之后管理的资源被释放;
 * 4. 通过调用std::weak_ptr类提供的lock()方法来获取管理所监测资源的shared_ptr对象;
 * 5. 通过调用std::weak_ptr类提供的reset()方法来清空对象，使其不监测任何资源;
 * 
 * -----返回管理this的shared_ptr
 * !-----解决循环引用问题(智能指针如果循环引用会导致内存泄露)
 * 
 */

#include <iostream>
#include <memory>

// C++11中提供了一个模板类叫做std::enable_shared_from_this<T>,
// 这个类中有一个方法叫做shared_from_this(),
// 通过这个方法可以返回一个共享智能指针,在函数的内部就是使用weak_ptr来监测this对象,
// 并通过调用weak_ptr的lock()方法返回一个shared_ptr对象.
struct Test : public std::enable_shared_from_this<Test>
{
    std::shared_ptr<Test> getSharedPtr()
    {
        return shared_from_this();
    }

    ~Test()
    {
        std::cout << "class Test is dis_struct ...\n";
    }
};

// 解决循环引用问题(智能指针如果循环引用会导致内存泄露)
struct TA;
struct TB;

struct TA
{
    // 在共享智能指针离开作用域之后引用计数只能减为1,
    // 这种情况下不会去删除智能指针管理的内存，导致类TA、TB的实例对象不能被析构，最终造成内存泄露
    // 通过使用weak_ptr可以解决这个问题，只要将类TA或者TB的任意一个成员改为weak_ptr
    // std::shared_ptr<TB> b_ptr;

    std::weak_ptr<TB> b_ptr;

    ~TA()
    {
        std::cout << "class TA is dis_struct ...\n";
    }
};

struct TB
{
    std::shared_ptr<TA> a_ptr;

    ~TB()
    {
        std::cout << "class TB is dis_struct ...\n";
    }
};

void testPtr()
{
    std::shared_ptr<TA> ap(new TA);
    std::shared_ptr<TB> bp(new TB);
    std::cout << "TA object use_count: " << ap.use_count() << std::endl;
    std::cout << "TB object use_count: " << bp.use_count() << std::endl;

    ap->b_ptr = bp;
    bp->a_ptr = ap;
    std::cout << "TA object use_count: " << ap.use_count() << std::endl;
    std::cout << "TB object use_count: " << bp.use_count() << std::endl;
}

// -------------------------------------
int main(int argc, const char **argv)
{
    std::shared_ptr<int> sp(new int);

    std::weak_ptr<int> wp1;
    std::weak_ptr<int> wp2(wp1);
    std::weak_ptr<int> wp3(sp);
    std::weak_ptr<int> wp4;
    wp4 = sp;
    std::weak_ptr<int> wp5;
    wp5 = wp3;

    std::cout << "the shared_ptr use_cout: " << sp.use_count() << std::endl;

    // !通过调用std::weak_ptr类提供的use_count()方法可以获得当前所观测资源的引用计数
    std::cout << "use_count: " << std::endl;
    std::cout << "wp1: " << wp1.use_count() << std::endl;
    std::cout << "wp2: " << wp2.use_count() << std::endl;
    std::cout << "wp3: " << wp3.use_count() << std::endl;
    std::cout << "wp4: " << wp4.use_count() << std::endl;
    std::cout << "wp5: " << wp5.use_count() << std::endl;

    // !通过调用std::weak_ptr类提供的expired()方法来判断观测的资源是否已经被释放
    std::shared_ptr<int> shared(new int(10));
    std::weak_ptr<int>   weak(shared);
    std::cout << "1. weak " << (weak.expired() ? "is" : "is not") << " expired" << std::endl;

    // !当共享智能指针调用shared.reset();之后管理的资源被释放
    shared.reset();
    std::cout << "2. weak " << (weak.expired() ? "is" : "is not") << " expired" << std::endl;

    // !通过调用std::weak_ptr类提供的lock()方法来获取管理所监测资源的shared_ptr对象
    std::shared_ptr<int> sp1, sp2;
    std::weak_ptr<int>   wp;

    sp1 = std::make_shared<int>(520);
    wp  = sp1;
    sp2 = wp.lock();
    std::cout << "use_count: " << wp.use_count() << std::endl;

    sp1.reset();
    std::cout << "use_count: " << wp.use_count() << std::endl;

    sp1 = wp.lock();
    std::cout << "use_count: " << wp.use_count() << std::endl;

    std::cout << "*sp1: " << *sp1 << std::endl;
    std::cout << "*sp2: " << *sp2 << std::endl;

    // !通过调用std::weak_ptr类提供的reset()方法来清空对象，使其不监测任何资源
    std::shared_ptr<int> sp_(new int(10));
    std::weak_ptr<int>   wp_(sp_);
    std::cout << "1. wp_ " << (wp_.expired() ? "is" : "is not") << " expired" << std::endl;

    wp_.reset();
    std::cout << "2. wp_ " << (wp_.expired() ? "is" : "is not") << " expired" << std::endl;

    // ===================================
    // 在调用enable_shared_from_this类的shared_from_this()方法之前,
    // ?必须要先初始化函数内部weak_ptr对象,否则该函数无法返回一个有效的shared_ptr对象
    std::shared_ptr<Test> _sp_(new Test);
    std::cout << "use_count: " << _sp_.use_count() << std::endl;

    std::shared_ptr<Test> sp__ = _sp_->getSharedPtr();
    std::cout << "use_count: " << _sp_.use_count() << std::endl;

    return 0;
}
