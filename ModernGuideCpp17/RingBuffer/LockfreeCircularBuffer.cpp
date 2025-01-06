/**
 * @file LockfreeCircularBuffer.cpp
 * @author Wei Li (weili_yzzcq@163.com)
 * @brief Circular buffer 无锁读写环形缓存区
 * @version 0.1
 * @date 2025-01-06
 * 
 * @copyright Copyright (c) 2025
 * 
 * g++ LockfreeCircularBuffer.cpp -std=c++20
 * clang++ LockfreeCircularBuffer.cpp -std=c++20
 * 
 */

/* 无锁读写环形缓存区 Lock-free Circular Buffer
一种并发编程中常用的数据结构,尤其适用于需要在多线程或多进程环境中快速高效地读写数据的场景;
它的特点是避免了锁的使用, 从而减少了线程之间的竞争, 提高了性能;
特别是在嵌入式底层驱动中, 无锁的方式更加高效, 能够有效提升驱动数据传输的性能; 

应用特点:
1. 环形结构: 数据结构是一个固定大小的缓冲区,当写指针达到缓冲区的末尾时,
            会自动绕回到缓冲区的开始位置,形成一个环形结构.
2. 无锁: 读写操作不使用锁(例如互斥锁), 
        通过原子操作(如CAS：Compare-And-Swap)来保证并发读写时的数据一致性.
3. 高效的并发读写: 读写操作不会阻塞, 不同线程可以同时访问读写指针,且不需要锁,减少了线程的切换和同步开销.
4. 线程安全: 无锁的实现通常依赖于现代硬件提供的原子操作(如CPU的 cmpxchg 指令)来确保线程安全.

实现原理:
1. 环形结构: 缓存区的头部和尾部通过索引表示, 使用数组来存储数据;
        头指针(read_ptr)和尾指针(write_ptr)指示当前读写位置;
        假设缓存区大小为 N, 当索引超过 N-1 时, 指针会回绕.
2. 原子操作: 为了保证多线程读写的安全性, 可以使用原子操作来更新 read_ptr 和 write_ptr;
        例如使用 CAS(Compare-and-Swap)来保证指针在更新时的正确性.
3. 缓冲区管理: 缓冲区的管理通常包括检查缓冲区是否已满(当写指针紧随读指针时)
        和是否为空(当读指针等于写指针时); 为了避免重叠和数据丢失, 需要保证读写指针正确同步.

关键细节:
<1>. 使用指针代替索引:
    使用指针来代替索引的方式能显著提高效率, 尤其是在读取和写入数据时;
    通过直接操作指针, 可以避免计算索引和取模运算, 从而减少一些不必要的开销.
指针增量: 通过将读写指针作为指针直接操作, 而不是使用索引, 可以避免频繁的取模运算和整数运算;
内存对齐: 如果数据是按某种对齐方式存储(例如每次读取/写入一个 int32_t 数据), 
        那么内存对齐也可以保证读取和写入时的效率;

<2>. 使用引用计数分离读写:
    这种方法可以进一步提高并发性, 通过为读写操作分别设置引用计数,
    确保在不同线程中可以并行执行读写操作, 而不会发生冲突;
读引用计数: 用于跟踪当前有多少个线程在读取缓冲区的内容;
写引用计数: 用于跟踪当前有多少个线程在写入缓冲区的内容;
这种分离式设计使得读操作和写操作在计数上互不干扰, 从而实现无锁分离的并发控制.

<3>. 原子操作:
    在嵌入式系统中, 原子操作对于无锁设计至关重要;
    通过硬件支持的原子操作, 线程间可以安全地进行内存共享而不需要显式的锁.
原子操作的硬件支持: 现代的嵌入式系统通常支持对特定数据类型的原子操作(例如32位或64位的整数),
     这有助于确保在多个线程或中断之间访问数据时不会发生竞态条件.
内存对齐: 原子操作通常要求操作的数据是按字节对齐的, 这对于性能尤为重要;
    确保缓冲区和指针的内存对齐可以最大限度地利用硬件的原子操作支持;
例如对于32位的嵌入式系统, 在操作对齐的整数时, 
CPU能直接通过单条指令进行读写, 因此可以避免使用更复杂的同步机制.
* */

/**
 * @brief 初始化环形缓存区
 * @param  self             缓存区句柄
 * @param  buff             缓存区地址
 * @param  size             缓存区长度
 * @return int 初始化结果
 */
// int CbbRingBufferInit(TRingBuffer *self, uint8_t *buff, int size);

/**
 * @brief 反初始化
 * @param  self             缓存区句柄
 */
// void CbbRingBufferDeInit(TRingBuffer *self);

/**
 * @brief 存入单个数据
 * @param  self             缓存区句柄
 * @param  data             要存入的数据
 * @return int 存入的结果 >=0 : 实际存入的数据个数
 */
// int CbbRingBufferWrite(TRingBuffer *self, uint8_t data);

/**
 * @brief 数据入队
 * @param  self             缓存区句柄
 * @param  data             要入队的数据
 * @param  len              要入队的数据的个数
 * @return int >= 0 : 实际存入的数据个数,空间不足则不会存储
 */
// int CbbRingBufferPushBytes(TRingBuffer *self, const uint8_t *data, int len);

/**
 * @brief 数据强制入队
 * @param  self             缓存区句柄
 * @param  data             要入队的数据
 * @param  len              要入队的数据的个数
 * @note  读写双线程不安全
 * @return int >= 0 : 实际存入的数据个数,会循环覆盖旧数据
 */
// int CbbRingBufferForcePush(TRingBuffer *self, const uint8_t *data, int len);

/**
 * @brief 从缓存区中取出数据
 * @param  self             缓存区句柄
 * @param  buffer           取出数据的存放区
 * @param  pop_size         要取出的数据个数
 * @return int >=0 : 实际取出的数据个数
 */
// int CbbRingBufferRead(TRingBuffer *self, uint8_t *buffer, int pop_size);

/**
 * @brief 从头部(写指针处)丢弃数据
 * @param  self             缓存区句柄
 * @param  discard_cnt      丢弃的数据个数
 * @return int 执行的结果 >= 0 : 实际丢弃的个数
 */
// int  CbbRingBufferDiscardFromHead(TRingBuffer *self, int discard_cnt);
/**
 * @brief 判断缓存区是否空
 * @param  self             缓存区句柄
 * @return Bool true: 缓存区已空 false: 缓存区未空
 */
// bool CbbRingBufferIsEmpty(TRingBuffer *self);

/**
 * @brief 判断缓存区是否满
 * @param  self             缓存区句柄
 * @return Bool true: 缓存区已满 false: 缓存区未满
 */
// bool CbbRingBufferIsFull(TRingBuffer *self);

/**
 * @brief 清除缓存区
 * @param  self             缓存区句柄
 */
// void CbbRingBufferClear(TRingBuffer *self);

/**
 * @brief 获取缓存区中数据的个数
 * @param  self             缓存区句柄
 * @return int >= 0:缓存区中的数据个数
 */
// int CbbRingBufferGetCnt(TRingBuffer *self);

/**
 * @brief 获取缓存区容量
 * @param  self             缓存区句柄
 * @return In32 int >= 0:缓存区的容量
 */
// int CbbRingBufferGetCapacity(TRingBuffer *self);

/**
 * @brief  获取缓存区剩余容量
 * 
 * @param  self             缓存区句柄
 * @return int int >= 0:缓存区的剩余容量
 */
// int CbbRingBufferGetResidue(TRingBuffer *self);

// --------------------------------------------
int main(int /* argc */, char ** /* argv */)
{
    return 0;
}
