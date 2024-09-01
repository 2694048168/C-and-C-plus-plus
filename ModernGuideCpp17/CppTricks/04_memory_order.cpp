/**
 * @file 04_memory_order.cpp
 * @author Wei Li (weili_yzzcq@163.com)
 * @brief 搞懂C++多线程内存模型(Memory Order)
 * @version 0.1
 * @date 2024-09-01
 * 
 * @copyright Copyright (c) 2024
 * 
 */

// 在多线程编程中,有两个需要注意的问题:
// ?一个是数据竞争; 数据竞争(Data Racing)
// ?另一个是内存执行顺序;

#include <atomic>
#include <cassert>
#include <iostream>
#include <thread>

// 什么是数据竞争(Data Racing), 数据竞争会导致什么问题
#if 0
int counter = 0;

void increment()
{
    for (int i = 0; i < 100000; ++i)
    {
        // ++counter实际上3条指令
        // 1. int tmp = counter;
        // 2. tmp = tmp + 1;
        // 3. counter = tmp;
        ++counter;
    }
}
#endif

// c++11引入了std::atomic<T>, 将某个变量声明为std::atomic<T>后,
// 通过std::atomic<T>的相关接口即可实现原子性的读写操作, 以解决数据竞争的问题
#if 1
// 只能用bruce-initialization的方式来初始化std::atomic<T>.
// std::atomic<int> counter{0} is ok,
// std::atomic<int> counter(0) is NOT ok.
std::atomic<int> counter{0};

void increment()
{
    for (int i = 0; i < 100000; ++i)
    {
        ++counter;
    }
}
#endif

// std::atomic提供了以下几个常用接口来实现原子性的读写操作, memory_order用于指定内存顺序
// 1. 原子性的读取值
// std::atomic<T>::store(T val, memory_order sync = memory_order_seq_cst);

// 2. 原子性的写入值
// std::atomic<T>::load(memory_order sync = memory_order_seq_cst);

// 3. 原子性的增加
// counter.fetch_add(1)等价于++counter
// std::atomic<T>::fetch_add(T val, memory_order sync = memory_order_seq_cst);

// 4. 原子性的减少
// counter.fetch_sub(1)等价于--counter
// std::atomic<T>::fetch_sub(T val, memory_order sync = memory_order_seq_cst);

// 5. 原子性的按位与
// counter.fetch_and(1)等价于counter &= 1
// std::atomic<T>::fetch_and(T val, memory_order sync = memory_order_seq_cst);

// 6. 原子性的按位或
// counter.fetch_or(1)等价于counter |= 1
// std::atomic<T>::fetch_or(T val, memory_order sync = memory_order_seq_cst);

// 7. 原子性的按位异或
// counter.fetch_xor(1)等价于counter ^= 1
// std::atomic<T>::fetch_xor(T val, memory_order sync = memory_order_seq_cst);

// ---------------------------------------------------------------------------
// 编译器和处理器进行优化, 可能会与代码中的顺序不同, 同时执行这两条语句, 从而提高程序的执行效率
// 指令重排的好处：它可以充分利用处理器的执行流水线, 提高程序的执行效率.
// *但是在多线程的场景下, 指令重排可能会引起一些问题
#if 0
std::atomic<bool> ready{false};
std::atomic<int>  data{0};

void producer()
{
    data.store(42, std::memory_order_relaxed);    // 原子性的更新data的值, 但是不保证内存顺序
    ready.store(true, std::memory_order_relaxed); // 原子性的更新ready的值, 但是不保证内存顺序
}

void consumer()
{
    // 原子性的读取ready的值, 但是不保证内存顺序
    while (!ready.load(std::memory_order_relaxed))
    {
        std::this_thread::yield(); // 啥也不做, 只是让出CPU时间片
    }

    // 当ready为true时, 再原子性的读取data的值
    std::cout << data.load(std::memory_order_relaxed); // 4. 消费者线程使用数据
}
#endif

#if 1
std::atomic<bool> ready{false};
std::atomic<int>  data{0};

void producer()
{
    /* producer线程里的ready.store(true, std::memory_order_relaxed);
    改为ready.store(true, std::memory_order_released);
    *一方面限制ready之前的所有操作不得重排到ready之后, 
    以保证先完成data的写操作, 再完成ready的写操作. 
    *另一方面保证先完成data的内存同步, 再完成ready的内存同步, 
    以保证consumer线程看到ready新值的时候, 一定也能看到data的新值.
     */
    data.store(42, std::memory_order_relaxed);    // 原子性的更新data的值, 但是不保证内存顺序
    ready.store(true, std::memory_order_release); // 保证data的更新操作先于ready的更新操作
}

void consumer()
{
    /* consumer线程里的while (!ready.load(memory_order_relaxed))改为
    while (!ready.load(memory_order_acquire)), 
    限制ready之后的所有操作不得重排到ready之前, 以保证先完成读ready操作, 再完成data的读操作;
     */
    // 保证先读取ready的值, 再读取data的值
    while (!ready.load(std::memory_order_acquire))
    {
        std::this_thread::yield(); // 啥也不做, 只是让出CPU时间片
    }

    // 当ready为true时, 再原子性的读取data的值
    std::cout << data.load(std::memory_order_relaxed); // 4. 消费者线程使用数据
}
#endif

// =================================
// 看看memory_order的所有取值与作用
// 1. memory_order_relaxed: 只确保操作是原子性的, 不对内存顺序做任何保证, 会带来上述producer-consumer例子中的问题.
// 2. memory_order_release: 用于写操作, 在写操作之前插入一个StoreStore屏障, 确保屏障之前的所有操作不会重排到屏障之后.
// 3. memory_order_acquire: 用于读操作, 在读操作之后插入一个LoadLoad屏障, 确保屏障之后的所有操作不会重排到屏障之前.
// 4. memory_order_acq_rel: 等效于memory_order_acquire和memory_order_release的组合,
//    同时插入一个StoreStore屏障与LoadLoad屏障. 用于读写操作.
// 5. memory_order_seq_cst: 最严格的内存顺序, 在memory_order_acq_rel的基础上, 保证所有线程看到的内存操作顺序是一致的.
//   这个可能不太好理解, 什么情况下需要用到memory_order_seq_cst?
//   *memory_order_seq_cst常用于multi producer - multi consumer的场景.
//   memory_order_seq_cst会带来最大的性能开销了, 相比其他的memory_order来说, 因为相当于禁用了CPU Cache.
// =================================
#if 1
std::atomic<bool> x = {false};
std::atomic<bool> y = {false};
std::atomic<int>  z = {0};

void write_x()
{
    x.store(true, std::memory_order_seq_cst);
}

void write_y()
{
    y.store(true, std::memory_order_seq_cst);
}

void read_x_then_y()
{
    while (!x.load(std::memory_order_seq_cst));
    if (y.load(std::memory_order_seq_cst))
    {
        ++z;
    }
}

void read_y_then_x()
{
    while (!y.load(std::memory_order_seq_cst));
    if (x.load(std::memory_order_seq_cst))
    {
        ++z;
    }
}
#endif

// ------------------------------------
int main(int argc, const char **argv)
{
#if 0
    std::thread t1(increment);
    std::thread t2(increment);

    t1.join();
    t2.join();

    std::cout << "Counter = " << counter << "\n";
/* 有一个全局变量counter(多线程共享数据), 两个线程同时对它进行增加操作. 
    理论上, counter的最终值应该是200000. 然而由于数据竞争的存在, counter的实际值可能会小于200000.
    这是因为, ++counter并不是一个原子操作, CPU会将++counter分成3条指令来执行:
    --> 先读取counter的值 --> 增加它 --> 然后再将新值写回counter.
    =========================================================
    Thread 1                  Thread 2
    ---------                 ---------
    int tmp = counter;        int tmp = counter;
    tmp = tmp + 1;            tmp = tmp + 1;
    counter = tmp;            counter = tmp;
    =========================================================
    两个线程可能会读取到相同的counter值, 然后都将它增加1, 然后将新值写回counter.
    这样两个线程实际上只完成了一次+操作,这就是数据竞争. */
#endif

#if 1
    // launch一个生产者线程
    std::thread t1(producer);
    // launch一个消费者线程
    std::thread t2(consumer);
    t1.join();
    t2.join();
    /* 有两个线程：一个生产者（producer）和一个消费者（consumer）.
    生产者先将data的值修改为一个有效值, 比如42, 然后再将ready的值设置为true, 通知消费者可以读取data的值了.
    预期的效果应该是当消费者看到ready为true时, 此时再去读取data的值, 应该是个有效值了, 对于本例来说即为42.
    *但实际的情况是, 消费者看到ready为true后, 读取到的data值可能仍然是0. 为什么呢?
    *[该情况在x64的机器上,因为内存模型是严格的, 很那复现该问题, 在ARM等内存模型宽松下很容易复现]
    [Weak VS strong Memory Models](http://dreamrunner.org/blog/2014/06/28/qian-tan-memory-reordering/)
    =========================================================
    | 非常弱	                 | 数据依赖性的弱 |	强制	   | 顺序一致
    | DEC Alpha	                | ARM	       | X86/64	   | dual 386
    | C/C++11 low-level atomics | PowerPC      | SPARC TSO | Java volatile/C/C++11 atomics
    =======================================================================================
    ?一方面可能是指令重排引起的. 在producer线程里, data和store是两个不相干的变量, 
    所以编译器或者处理器可能会将data.store(42, std::memory_order_relaxed);
    重排到ready.store(true, std::memory_order_relaxed);之后执行, 
    这样consumer线程就会先读取到ready为true, 但是data仍然是0.
    ?另一方面可能是内存顺序不一致引起的. 即使producer线程中的指令没有被重排, 
    但CPU的多级缓存会导致consumer线程看到的data值仍然是0. 充分理解现代CPU多级缓存关系.
    每个CPU核心都有自己的L1 Cache与L2 Cache. 
    producer线程修改了data和ready的值, 但修改的是L1 Cache中的值, 
    producer线程和consumer线程的L1 Cache并不是共享的, 
    所以consumer线程不一定能及时的看到producer线程修改的值. 
    !CPU Cache的同步是件很复杂的事情, 生产者更新了data和ready后, 
    !还需要根据MESI协议将值写回内存,并且同步更新其他CPU核心Cache里data和ready的值, 
    !这样才能确保每个CPU核心看到的data和ready的值是一致的. 
    而data和ready同步到其他CPU Cache的顺序也是不固定的, 可能先同步ready, 再同步data, 
    这样的话consumer线程就会先看到ready为true, 但data还没来得及同步, 所以看到的仍然是0.
    这就是我们所说的内存顺序不一致问题.
    =========================================================
    为了避免这个问题, 需要在producer线程中, 
    在data和ready的更新操作之间插入一个内存屏障(Memory Barrier), 
    保证data和ready的更新操作不会被重排, 并且保证data的更新操作先于ready的更新操作.
    =============== std::memory_order_release ===============
    *std::memory_order_release: 用于写操作, 比如std::atomic::store(T, memory_order_release), 
    *会在写操作之前插入一个StoreStore屏障, 确保屏障之前的所有操作不会重排到屏障之后.
    =========================================================
        +
        |
        |
        | No Moving Down
        |
    +-----v---------------------+
    |     StoreStore Barrier    |
    +---------------------------+
    =========================================================

    =============== std::memory_order_acquire ===============
    *std::memory_order_acquire: 用于读操作, 比如std::atomic::load(memory_order_acquire), 
    *会在读操作之后插入一个LoadLoad屏障, 确保屏障之后的所有操作不会重排到屏障之前.
    =========================================================
    +---------------------------+
    |     LoadLoad Barrier      |
    +-----^---------------------+
        |
        |
        | No Moving Up
        |
        |
        +
    ========================================================= */
#endif

    std::thread a(write_x);
    std::thread b(write_y);
    std::thread c(read_x_then_y);
    std::thread d(read_y_then_x);
    a.join();
    b.join();
    c.join();
    d.join();
    assert(z.load() != 0); // ?will never happen
    /* 以上这个例子中, a, c, c, d四个线程运行完毕后, 期望z的值一定是不等于0的, 
    也就是read_x_then_y和read_y_then_x两个线程至少有一个会执行++z. 
    要保证这个期望成立, 我们必须对x和y的读写操作都使用memory_order_seq_cst, 
    以保证所有线程看到的内存操作顺序是一致的.

    ?什么意思呢, 如果不使用memory_order_seq_cst, 
    read_x_then_y可能会先看到y为true才看到x为true, 同时read_y_then_x也可能会先看到x才看到y, 
    因为write_x和write_y是在不同的CPU核心上执行, read_x_then_y有可能先同步y再同步x, 
    read_y_then_x也有可能先同步x再同步y, 
    这样就会导致read_x_then_y与read_y_then_x都无限阻塞在while循环里.
    *使用memory_order_seq_cst标记的读写操作将会以一致的内存顺序执行, 
    比如read_x_then_y的CPU核心如果先同步y再同步x, 
    那么read_y_then_x的CPU核心也一定会先同步y再同步x, 
    这样就能保证read_x_then_y与read_y_then_x总有一个能执行到++z.

    =========================================================
    ?这样说可能还是会比较抽象, 可以直接理解下memory_order_seq_cst是怎么实现的. 
    *被memory_order_seq_cst标记的写操作, 会立马将新值写回内存, 而不仅仅只是写到Cache里就结束了; 
    *被memory_order_seq_cst标记的读操作, 会立马从内存中读取新值, 而不是直接从Cache里读取. 
    这样相当于write_x, write_y, read_x_then_y, read_y_then_x 
    四个线程都是在同一个内存中读写x和y, 也就不存在Cache同步的顺序不一致问题了.
    ?所以也就能理解为什么memory_order_seq_cst会带来最大的性能开销了, 
    ?相比其他的memory_order来说, 因为相当于禁用了CPU Cache.
    =========================================================
    
    memory_order_seq_cst常用于multi producer - multi consumer的场景, 
    比如上述例子里的write_x和write_y都是producer, 会同时修改x和y, 
    read_x_then_y和read_y_then_x两个线程, 都是consumer, 会同时读取x和y, 
    并且这两个consumer需要按照相同的内存顺序来同步x和y. */

    return 0;
}
