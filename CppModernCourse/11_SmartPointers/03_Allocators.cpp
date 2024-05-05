/**
 * @file 03_Allocators.cpp
 * @author Wei Li (weili_yzzcq@163.com)
 * @brief 
 * @version 0.1
 * @date 2024-05-05
 * 
 * @copyright Copyright (c) 2024
 * 
 */

#include <cassert>
#include <cstddef>
#include <cstdio>
#include <memory>
// #include <new>

/**
 * @brief 分配器 Allocators
 * *分配器是为内存请求提供服务的底层对象, stdlib 和 Boost 库使们能够提供分配器, 这可以定制库分配动态内存的方式.
 * 使用默认分配器 std::allocate 完全足够了, 使用 operator new(size_t) 分配内存,
 *  该运算符从空闲存储空间(堆 heap)分配原始内存, 使用operator delete(void*) 释放内存,
 *  它也从空闲存储空间中释放原始内存
 *
 * 一些场景(例如游戏、高频交易、科学分析和嵌入式应用程序)中,
 * 与默认自由存储操作相关的内存和计算开销是不可接受的,
 * 在这种情况下, 实现自己的分配器相对容易.
 * NOTE: 注意除非已经进行了一些性能测试, 且这些测试表明默认分配器是性能的瓶颈, 否则不应该实现自定义分配器.
 * 自定义分配器的理念是, 与默认分配器模型的设计者相比, 你更了解自己的程序, 因此可以进行改进以提高分配性能.
 * 
 * 需要提供一个具有以下特征的模板类才能作为分配器工作:
 * 1. 合适的默认构造函数;
 * 2. 对应于模板参数的 value_type 成员;
 * 3. 模板构造函数, 它应可以在处理 value_type 的变化时复制分配器的内部状态;
 * 4. allocate 方法;
 * 5. deallocate 方法;
 * 6. operator== 和 operator!=
 * 
 * TODO: https://en.cppreference.com/w/cpp/memory#Allocators
 * 
 */
static size_t n_allocated;
static size_t n_deallocated;

template<typename T>
struct MyAllocator
{
    using value_type = T;

    MyAllocator() noexcept = default;

    template<typename U>
    MyAllocator(const MyAllocator<U> &) noexcept
    {
    }

    T *allocate(size_t n)
    {
        auto p = operator new(sizeof(T) * n);
        ++n_allocated;
        return static_cast<T *>(p);
    }

    void deallocate(T *p, size_t n)
    {
        operator delete(p);
        ++n_deallocated;
    }
};

template<typename T1, typename T2>
bool operator==(const MyAllocator<T1> &, const MyAllocator<T2> &)
{
    return true;
}

template<typename T1, typename T2>
bool operator!=(const MyAllocator<T1> &, const MyAllocator<T2> &)
{
    return false;
}

struct DeadMenOfDun
{
    DeadMenOfDun(const char *m = "")
        : message{m}
    {
        oaths_to_fulfill++;
    }

    ~DeadMenOfDun()
    {
        oaths_to_fulfill--;
    }

    const char *message;
    static int  oaths_to_fulfill;
};

int DeadMenOfDun::oaths_to_fulfill{};

// -----------------------------------
int main(int argc, const char **argv)
{
    auto message = "The way is shut.\n";

    MyAllocator<DeadMenOfDun> my_alloc;

    {
        auto aragon = std::allocate_shared<DeadMenOfDun>(my_alloc, message);

        assert(aragon->message == message);
        assert(n_allocated == 1);
        assert(n_deallocated == 0);
    }
    assert(n_allocated == 1);
    assert(n_deallocated == 1);

    return 0;
}
