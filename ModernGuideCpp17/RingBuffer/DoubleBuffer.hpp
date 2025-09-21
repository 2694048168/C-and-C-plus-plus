/**
 * @file DoubleBuffer.hpp
 * @author Wei Li (Ithaca) (weili_yzzca@163.com)
 * @brief C++实现DoubleBuffer类：兼顾线程安全与易用性
 * @version 0.1
 * @date 2025-09-21
 * 
 * @copyright Copyright (c) 2025
 * 
 */

#include <memory>
#include <mutex>
#include <utility>

/**
 * @brief 双缓冲模板类，实现读写分离。
 * @tparam CACHE 实际存储数据的类型，例如 std::vector<int>, MyGameState 等。
 */
template<typename CACHE>
class DoubleBuffer
{
public:
    /**
     * @brief 构造函数。
     * 初始时，两个缓冲区都为空（nullptr）。
     * 客户端必须先调用 set() 提供初始数据，get() 才能返回有效对象。
     */
    DoubleBuffer()
        : _oddeven(0)
    {
        // std::shared_ptr 默认构造为 nullptr，无需显式初始化
    }

    /**
     * @brief 获取当前的读缓冲。
     * @return 指向当前读缓冲的 const shared_ptr。
     *         返回 const shared_ptr 保证了读取线程无法修改数据。
     */
    std::shared_ptr<const CACHE> get()
    {
        std::lock_guard<std::mutex> lock(_lock); // 保护 _oddeven 和 _caches 数组的访问
        return _caches[_oddeven % 2];
    }

    /**
     * @brief 设置新的写缓冲，并触发缓冲翻转。
     * @param c 指向新数据的 shared_ptr。
     *          调用者需要负责创建并填充这个新的 CACHE 对象。
     */
    void set(std::shared_ptr<CACHE> c)
    {
        std::lock_guard<std::mutex> lock(_lock); // 保护 _oddeven 和 _caches 数组的访问
        _caches[(_oddeven + 1) % 2] = c;         // 将新数据赋给“另一个”缓冲区
        _oddeven++;                              // 翻转索引，使新数据成为读缓冲
    }

    /**
     * @brief 辅助函数：方便地创建 CACHE 类型的 shared_ptr。
     *        使用完美转发，支持 CACHE 类型的各种构造函数。
     * @tparam ARGS 构造函数参数类型。
     * @param args 构造函数参数。
     * @return 新创建的 CACHE 对象的 shared_ptr。
     */
    template<typename... ARGS>
    static std::shared_ptr<CACHE> make(ARGS &amp; &amp;... args)
    {
        return std::make_shared<CACHE>(std::forward<ARGS>(args)...);
    }

private:
    std::mutex                   _lock;      // 保护缓冲区指针和索引的访问
    unsigned int                 _oddeven;   // 用于切换缓冲区的索引 (0 或 1)
    std::shared_ptr<const CACHE> _caches[2]; // 存储两个缓冲区的 shared_ptr
};
