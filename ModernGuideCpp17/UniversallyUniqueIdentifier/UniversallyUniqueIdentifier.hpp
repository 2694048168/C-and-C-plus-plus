/**
 * @file UniversallyUniqueIdentifier.hpp
 * @author Wei Li (weili_yzzcq@163.com)
 * @brief 
 * @version 0.1
 * @date 2024-08-15
 * 
 * @copyright Copyright (c) 2024
 * 
 */

#ifndef __UniversallyUniqueIdentifier_HPP__
#define __UniversallyUniqueIdentifier_HPP__

#include <atomic>
#include <string>

/**
 * @brief 在软件开发中，生成唯一值是一项常见且重要的任务，
 * 特别是在需要唯一标识符（如数据库主键、临时文件名、用户会话ID等）的场景中。
 * C++作为一种广泛使用的编程语言，提供了多种生成唯一值的方法。
 * 1. 使用标准库中的随机数生成器和计数器组合生成唯一值;
 * 2. 利用时间戳和计数器组合生成唯一值;
 * 3. 通过UUID（Universally Unique Identifier）生成唯一值;
 * 
 */

// 仔细想一想, C语言标准和C++语言标准, 针对变量/函数的命名的修饰
// C++ 新增了命名空间和函数重载功能的缘故, 需要对函数签名进行一定规则的修饰
#ifdef __cplusplus
extern "C"
{
#endif

// 方法一：使用标准库中的随机数生成器
// C++11及更高版本的标准库中提供了随机数生成的功能，可以通过这些功能生成唯一值。
// std::random_device类用于生成一个非确定性随机数，可以作为随机数引擎的种子。
// 而std::mt19937是一个基于梅森旋转算法的伪随机数生成器，具有良好的随机性。
// *这种方法适用于需要生成随机唯一值的场景，但需要注意的是，在高并发环境下可能会生成重复的值。
// int _stdcall getUUID_random();
int getUUID_random();

// 方法二：利用时间戳和计数器组合生成唯一值
// 另一种生成唯一值的方法是利用当前时间戳和一个计数器组合。
// 时间戳保证了值的唯一性（在毫秒级别），而计数器则用于在同一时间戳内生成多个唯一值
// *在高并发环境下，如果多个线程或进程同时访问计数器，可能会导致生成的唯一值重复。
// !在实际应用中，可以通过加锁等同步机制来避免这一问题。
class UniqueIDGenerator
{
private:
    // static int counter;
    static std::atomic<int> counter;

public:
    static std::string generate();
};

// 方法三：通过UUID生成唯一值
// UUID是一种广泛使用的生成唯一值的方法，它可以生成128位的唯一标识符。
// 在C++中，可以通过Boost库中的UUID功能来生成UUID。

#ifdef __cplusplus
}
#endif

#endif /* __UniversallyUniqueIdentifier_HPP__ */
