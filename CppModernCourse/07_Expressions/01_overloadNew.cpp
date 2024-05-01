/**
 * @file 01_overloadNew.cpp
 * @author Wei Li (weili_yzzcq@163.com)
 * @brief 
 * @version 0.1
 * @date 2024-05-01
 * 
 * @copyright Copyright (c) 2024
 * 
 */
#include <cstddef>
#include <cstdio>
#include <new>

/**
 * @brief 重载运算符 new
 * 可通过运算符 new 分配带有动态存储期的对象, 默认情况下, 运算符 new 将在自由存储区中分配内存,
 * 从而为动态对象分配空间. 
 * 自由存储区(free store)也被称为堆(heap), 是实现定义的存储位置.
 * 在桌面操作系统中, 内核通常管理自由存储区(请参考 Windows 上的 HeapAlloc 
 * 以及 Linux 和 macOS 上的 malloc), 并且该区域一般都很大.
 * 
 * 1. 自由存储区的可用性
 * 在某些环境(如 Windows 内核或嵌入式系统)中, 默认情况下没有可用的自由存储区.
 * 在其他环境(如游戏开发或高频交易)中, 自由存储区分配会带来大量延迟, 因为其管理权被委托给了操作系统.
 * *实时操作系统 Real Time Operating System, 简称RTOS
 * 可以重载自由存储区操作并控制分配, 这可以通过重载运算符 new 来实现.
 * 
 * 2. 头文件 ＜new＞
 * 在支持自由存储区操作的环境中,＜new＞包含以下四个运算符:
 * - void* operator new(size_t);
 * - void operator delete(void*);
 * - void* operator new[](size_t);
 * - void operator delete[](void*);
 * !请注意, new 运算符的返回类型为 void*; 自由存储区操作处理未初始化的原始内存.
 * 可以提供以上四个运算符的自定义版本, 唯一需要做的是在程序中定义一次,
 * 此时编译器会使用自定义版本而非默认版本.
 * 主要问题之一是存在内存碎片化问题, 随着时间的推移, 
 * 大量的内存分配和释放操作可能会使可用内存块散布在整个自由存储区中,
 * 可能导致有足够的可用内存但它们分散在已分配的内存间隙,
 * 即使从技术角度出发有足够的可用内存可提供给请求者, 也会导致大量内存请求失败.
 * 
 * *image processing with camera stream data high-speed,
 * *not new and delete memory, pre-malloc memory and ring-buffer.
 * 
 * 3. 桶(Bucket)
 * 一种方法是将已分配的内存分成固定大小的区域(称为桶),
 * 当请求内存时, 即使没有请求整个桶的内存, 环境也会分配整个桶.
 * 例如 Windows 提供了两个用于分配动态内存的函数: VirtualAllocEx 和 HeapAlloc.
 * *VirtualAllocEx 函数是一个底层函数, 它提供多种选项, 如将内存分配给哪个进程、
 * 首选的内存地址、请求的大小和权限(如内存是否可读、可写和可执行). 此函数分配的字节数不会少于4096(页内存大小).
 * *HeapAlloc 是一个更高级别的函数, 它可以分配小于一页的内存, 否则就调用VirtualAllocEx.
 * 至少对于 Visual Studio 的编译器, new 默认调用 HeapAlloc.
 * 此种策略让内存分配四舍五入到桶大小, 用牺牲一部分内存空间的代价来防止内存碎片化.
 * !Windows 等现代操作系统有极其复杂的策略来分配不同大小的内存.
 * 
 * 4. 控制自由存储区
 */
struct Bucket
{
    static const size_t data_size{4096};
    std::byte           data[data_size];
};

struct Heap
{
    void *allocate(size_t bytes)
    {
        if (bytes > Bucket::data_size)
            throw std::bad_alloc{};

        for (size_t i{}; i < n_heap_buckets; i++)
        {
            if (!bucket_used[i])
            {
                bucket_used[i] = true;
                return buckets[i].data;
            }
        }
        throw std::bad_alloc{};
    }

    void free(void *p)
    {
        for (size_t i{}; i < n_heap_buckets; i++)
        {
            if (buckets[i].data == p)
            {
                bucket_used[i] = false;
                return;
            }
        }
    }

    static const size_t n_heap_buckets{10};

    Bucket buckets[n_heap_buckets]{};
    bool   bucket_used[n_heap_buckets]{};
};

/**
 * @brief 5. 使用自定义 Heap 类
 * 在命名空间范围内声明 Heap 类是分配 Heap 的一个方法, 这样它就具有静态存储期.
 * 因为其生命周期在程序启动时开始, 
 * 所以可以在 operator new 和 operator delete 重载中使用它 
 * 
 */
Heap heap;

void *operator new(size_t n_bytes)
{
    return heap.allocate(n_bytes);
}

void operator delete(void *p)
{
    return heap.free(p);
}

// -----------------------------------
int main(int argc, const char **argv)
{
    // 此时如果使用 new 和 delete, 那么动态内存管理将使用 heap,
    // 而不是使用环境提供的默认自由存储区

    printf("Buckets: %p\n", heap.buckets);
    auto breakfast = new unsigned int{0xC0FFEE};
    auto dinner    = new unsigned int{0xDEADBEEF};
    printf("Breakfast: %p 0x%x\n", breakfast, *breakfast);
    printf("Dinner: %p 0x%x\n", dinner, *dinner);

    delete breakfast;
    delete dinner;
    try
    {
        while (true)
        {
            new char;
            printf("Allocated a char.\n");
        }
    }
    catch (const std::bad_alloc &)
    {
        printf("std::bad_alloc caught.\n");
    }

    return 0;
}
