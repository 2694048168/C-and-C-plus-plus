/**
 * @file MemoryPoolNginx.cpp
 * @author Wei Li (weili_yzzcq@163.com)
 * @brief C++ 高性能内存池的设计与实现
 * @version 0.1
 * @date 2025-01-06
 * 
 * @copyright Copyright (c) 2025
 * 
 * g++ MemoryPoolNginx.cpp -std=c++20
 * clang++ MemoryPoolNginx.cpp -std=c++20
 * 
 */

/* 写内存池的原理之前,按照惯例先说内存池的应用场景
 * Why? 
 * 1. 因为malloc/new等分配内存的方式, 需要涉及到系统调用sbrk, 频繁地malloc和free会消耗系统资源;
 * 2. 如果频繁地malloc和free,由于malloc的地址是不确定的,因为每次malloc的时候,
 *    会先在freelist中找一个适合其大小的块, 如果找不到, 才会调用sbrk直接拓展堆的内存边界;
 *    (freelist是之前free掉的内存, 内核会将其组织成一个链表, 留待下次malloc的时候查找使用);
 *    因为不确定, 所以容易产生内存碎片(频繁操作,申请释放不同大小内存,导致有些地址无法连续被使用);
 * 
 * 两种内存分配方式, 大内存和小内存使用不同的数据结构来存储;
 * 
 */
// * 改编自nginx内程池，由于nginx源码纯C，我这里也用C接口进行内存管理。
// * 修改了很多nginx中晦涩的变量名，比较容易理解
#include <stdlib.h>
#include <string.h>

#include <iostream>
using namespace std;

class small_block
{
public:
    char        *cur_usable_buffer;
    char        *buffer_end;
    small_block *next_block;
    int          no_enough_times;
};

class big_block
{
public:
    char      *big_buffer;
    big_block *next_block;
};

class memory_pool
{
public:
    size_t              small_buffer_capacity;
    small_block        *cur_usable_small_block;
    big_block          *big_block_start;
    small_block         small_block_start[0];
    static memory_pool *createPool(size_t capacity);
    static void         destroyPool(memory_pool *pool);
    static char        *createNewSmallBlock(memory_pool *pool, size_t size);
    static char        *mallocBigBlock(memory_pool *pool, size_t size);
    static void        *poolMalloc(memory_pool *pool, size_t size);
    static void         freeBigBlock(memory_pool *pool, char *buffer_ptr);
};

//-创建内存池并初始化，api以静态成员(工厂)的方式模拟C风格函数实现
//-capacity是buffer的容量，在初始化的时候确定，后续所有小块的buffer都是这个大小
memory_pool *memory_pool::createPool(size_t capacity)
{
    //-我们先分配一大段连续内存,该内存可以想象成这段内存由pool+small_block+small_block_buffers三个部分组成.
    //-为什么要把三个部分(可以理解为三个对象)用连续内存来存,因为这样整个池看起来比较优雅.各部分地址不会天女散花地落在内存的各个角落.
    size_t total_size = sizeof(memory_pool) + sizeof(small_block) + capacity;
    void  *temp       = malloc(total_size);
    memset(temp, 0, total_size);

    memory_pool *pool = (memory_pool *)temp;
    fprintf(stdout, "pool address:%p\n", pool);
    //-此时temp是pool的指针，先来初始化pool对象
    pool->small_buffer_capacity  = capacity;
    pool->big_block_start        = nullptr;
    pool->cur_usable_small_block = (small_block *)(pool->small_block_start);

    //-pool+1的1是整个memory_pool的步长，别弄错了。此时sbp是small_block的指针
    small_block *sbp = (small_block *)(pool + 1);
    fprintf(stdout, "first small block address:%p\n", sbp);
    //-初始化small_block对象
    sbp->cur_usable_buffer = (char *)(sbp + 1);
    fprintf(stdout, "first small block buffer address:%p\n", sbp->cur_usable_buffer);
    sbp->buffer_end      = sbp->cur_usable_buffer + capacity; //-第一个可用的buffer就是开头，所以end=开头+capacity
    sbp->next_block      = nullptr;
    sbp->no_enough_times = 0;

    return pool;
};

//-销毁内存池
void memory_pool::destroyPool(memory_pool *pool)
{
    //-销毁大内存
    big_block *bbp = pool->big_block_start;
    while (bbp)
    {
        if (bbp->big_buffer)
        {
            free(bbp->big_buffer);
            bbp->big_buffer = nullptr;
        }
        bbp = bbp->next_block;
    }
    //-为什么不删除big_block节点？因为big_block在小内存池中，等会就和小内存池一起销毁了

    //-销毁小内存
    small_block *temp = pool->small_block_start->next_block;
    while (temp)
    {
        small_block *next = temp->next_block;
        free(temp);
        temp = next;
    }
    free(pool);
}

//-当所有small block都没有足够空间分配，则创建新的small block并分配size空间，返回分配空间的首指针
char *memory_pool::createNewSmallBlock(memory_pool *pool, size_t size)
{
    //-先创建新的small block，注意还有buffer
    size_t malloc_size = sizeof(small_block) + pool->small_buffer_capacity;
    void  *temp        = malloc(malloc_size);
    memset(temp, 0, malloc_size);

    //-初始化新的small block
    small_block *sbp = (small_block *)temp;
    fprintf(stdout, "new small block address:%p\n", sbp);
    sbp->cur_usable_buffer = (char *)(sbp + 1); //-跨越一个small_block的步长
    fprintf(stdout, "new small block buffer address:%p\n", sbp->cur_usable_buffer);
    sbp->buffer_end      = (char *)temp + malloc_size;
    sbp->next_block      = nullptr;
    sbp->no_enough_times = 0;
    //-预留size空间给新分配的内存
    char *res              = sbp->cur_usable_buffer; //-存个副本作为返回值
    sbp->cur_usable_buffer = res + size;

    //-因为目前的所有small_block都没有足够的空间了。
    //-意味着可能需要更新线程池的cur_usable_small_block，也就是寻找的起点
    small_block *p = pool->cur_usable_small_block;
    while (p->next_block)
    {
        if (p->no_enough_times > 4)
        {
            pool->cur_usable_small_block = p->next_block;
        }
        ++(p->no_enough_times);
        p = p->next_block;
    }

    //-此时p正好指向当前pool中最后一个small_block,将新节点接上去。
    p->next_block = sbp;

    //-因为最后一个block有可能no_enough_times>4导致cur_usable_small_block更新成nullptr
    //-所以还要判断一下
    if (pool->cur_usable_small_block == nullptr)
    {
        pool->cur_usable_small_block = sbp;
    }
    return res; //-返回新分配内存的首地址
}

//-分配大块的内存
char *memory_pool::mallocBigBlock(memory_pool *pool, size_t size)
{
    //-先分配size大小的空间
    void *temp = malloc(size);
    memset(temp, 0, size);

    //-从big_block_start开始寻找,注意big block是一个栈式链，插入新元素是插入到头结点的位置。
    big_block *bbp = pool->big_block_start;
    int        i   = 0;
    while (bbp)
    {
        if (bbp->big_buffer == nullptr)
        {
            bbp->big_buffer = (char *)temp;
            return bbp->big_buffer;
        }
        if (i > 3)
        {
            break; //-为了保证效率，如果找三轮还没找到有空buffer的big_block，就直接建立新的big_block
        }
        bbp = bbp->next_block;
        ++i;
    }

    //-创建新的big_block，这里比较难懂的点，就是Nginx觉得big_block的buffer虽然是一个随机地址的大内存
    //-但是big_block本身算一个小内存，那就不应该还是用随机地址，应该保存在内存池内部的空间。
    //-所以这里有个套娃的内存池malloc操作
    big_block *new_bbp = (big_block *)memory_pool::poolMalloc(pool, sizeof(big_block));
    //-初始化
    new_bbp->big_buffer   = (char *)temp;
    new_bbp->next_block   = pool->big_block_start;
    pool->big_block_start = new_bbp;

    //-返回分配内存的首地址
    return new_bbp->big_buffer;
}

//-分配内存
void *memory_pool::poolMalloc(memory_pool *pool, size_t size)
{
    //-先判断要malloc的是大内存还是小内存
    if (size < pool->small_buffer_capacity)
    { //-如果是小内存
        //-从cur small block开始寻找
        small_block *temp = pool->cur_usable_small_block;
        do
        {
            //-判断当前small block的buffer够不够分配
            //-如果够分配,直接返回
            if (temp->buffer_end - temp->cur_usable_buffer > size)
            {
                char *res               = temp->cur_usable_buffer;
                temp->cur_usable_buffer = temp->cur_usable_buffer + size;
                return res;
            }
            temp = temp->next_block;
        }
        while (temp);
        //-如果最后一个small block都不够分配，则创建新的small block;
        //-该small block在创建后,直接预先分配size大小的空间,所以返回即可.
        return createNewSmallBlock(pool, size);
    }
    //-分配大内存
    return mallocBigBlock(pool, size);
}

//-释放大内存的buffer，由于是一个链表，所以，确实，这是效率最低的一个api了
void memory_pool::freeBigBlock(memory_pool *pool, char *buffer_ptr)
{
    big_block *bbp = pool->big_block_start;
    while (bbp)
    {
        if (bbp->big_buffer == buffer_ptr)
        {
            free(bbp->big_buffer);
            bbp->big_buffer = nullptr;
            return;
        }
        bbp = bbp->next_block;
    }
}

// --------------------------------------------
int main(int /* argc */, char ** /* argv */)
{
    memory_pool *pool = memory_pool::createPool(1024);
    //-分配小内存
    char        *p1 = (char *)memory_pool::poolMalloc(pool, 2);
    fprintf(stdout, "little malloc1:%p\n", p1);
    char *p2 = (char *)memory_pool::poolMalloc(pool, 4);
    fprintf(stdout, "little malloc2:%p\n", p2);
    char *p3 = (char *)memory_pool::poolMalloc(pool, 8);
    fprintf(stdout, "little malloc3:%p\n", p3);
    char *p4 = (char *)memory_pool::poolMalloc(pool, 256);
    fprintf(stdout, "little malloc4:%p\n", p4);
    char *p5 = (char *)memory_pool::poolMalloc(pool, 512);
    fprintf(stdout, "little malloc5:%p\n", p5);

    //-测试分配不足开辟新的small block
    char *p6 = (char *)memory_pool::poolMalloc(pool, 512);
    fprintf(stdout, "little malloc6:%p\n", p6);

    //-测试分配大内存
    char *p7 = (char *)memory_pool::poolMalloc(pool, 2048);
    fprintf(stdout, "big malloc1:%p\n", p7);

    char *p8 = (char *)memory_pool::poolMalloc(pool, 4096);
    fprintf(stdout, "big malloc2:%p\n", p8);

    //-测试free大内存
    memory_pool::freeBigBlock(pool, p8);

    //-测试再次分配大内存（我这里测试结果和p8一样）
    char *p9 = (char *)memory_pool::poolMalloc(pool, 2048);
    fprintf(stdout, "big malloc3:%p\n", p9);

    //-销毁内存池
    memory_pool::destroyPool(pool);

    return 0;
}
