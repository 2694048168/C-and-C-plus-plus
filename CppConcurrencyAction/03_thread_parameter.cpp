/**
 * @file 03_thread_parameter.cpp
 * @author Wei Li (weili_yzzcq@163.com)
 * @brief 
 * @version 0.1
 * @date 2023-06-06
 * 
 * @copyright Copyright (c) 2023
 * 
 */

#include <functional>
#include <iostream>
#include <iterator>
#include <memory>
#include <numeric>
#include <thread>
#include <utility>
#include <vector>

void foo(int i, const std::string &str)
{
    std::cout << "integer: " << i << " is: " << str << "\n";
}

void func(int &i, const std::string &str)
{
    std::cout << "integer: " << i << " is: " << str << "\n";
    ++i;
}

class X
{
public:
    void do_lengthy_work(const std::string &msg)
    {
        std::cout << msg << std::endl;
    }
};

class BigObject
{
public:
    void prepare_data(const int &big_number)
    {
        this->m_number = big_number;
        std::cout << "prepare data: " << big_number << std::endl;
    }

    int get_data() const
    {
        return m_number;
    }

private:
    int m_number{0};
};

void process_big_object(std::unique_ptr<BigObject>) {}

void some_function()
{
    std::cout << "do some function\n";
}

void some_other_function()
{
    std::cout << "do some other function\n";
}

// 量产线程，等待它们结束
void do_work(unsigned int id)
{
    std::cout << "the id : " << id << "\n";
}

void f()
{
    // 将 std::thread 放入 std::vector 是向线程自动化管理迈出的第一步：
    // 并非为这些线程创建独立的变量，而是把它们当做一个组。
    // 创建一组线程(数量在运行时确定)，而非代码创建固定数量(12)的线程。
    std::vector<std::thread> threads;
    for (size_t i = 0; i < 12; ++i)
    {
        threads.emplace_back(do_work, i);
    }

    for (auto &entry : threads)
    {
        entry.join();
    }
}

// 确定线程数量
template<typename Iterator, typename T>
class AccumulateBlock
{
public:
    void operator()(Iterator first, Iterator last, T &result)
    {
        result = std::accumulate(first, last, result);
    }
};

template<typename Iterator, typename T>
T parallel_accumulate(Iterator first, Iterator last, T init)
{
    const unsigned long length = std::distance(first, last);

    if (!length) /* first == last */
    {
        return init;
    }

    // 确定启动线程的数量
    const unsigned long min_per_thread   = 25;
    const unsigned long max_threads      = (length + min_per_thread - 1) / min_per_thread;
    const unsigned long hardware_threads = std::thread::hardware_concurrency();
    const unsigned long num_threads      = std::min(hardware_threads != 0 ? hardware_threads : 2, max_threads);

    // 确定每一个线程的计算量
    const unsigned long block_size = length / num_threads;

    std::vector<T>           results(num_threads);
    std::vector<std::thread> threads(num_threads - 1);

    Iterator block_start = first;
    for (size_t i = 0; i < (num_threads - 1); ++i)
    {
        Iterator block_end = block_start;
        std::advance(block_end, block_size);
        threads[i]  = std::thread{AccumulateBlock<Iterator, T>(), block_start, block_end, std::ref(results[i])};
        block_start = block_end;
    }

    AccumulateBlock<Iterator, T>()(block_start, last, results[num_threads - 1]);

    for (auto &entry : threads)
    {
        entry.join();
    }

    return std::accumulate(results.begin(), results.end(), init);
}

// 线程通用标识符 ID
std::thread::id master_thread;

void some_core_part_of_algorithm()
{
    std::cout << "this thread id: " << std::this_thread::get_id() << "\n";
    if (std::this_thread::get_id() == master_thread)
    {
        std::cout << "do_master_thread_work()\n";
    }

    std::cout << "do_common_work()\n";
}

int main(int argc, const char **argv)
{
    // 向可调用对象或函数传递参数，只需要将参数作为 std::thread 构造函数的附加参数即可。
    // 注意，这些参数会拷贝至新线程的内存空间中(同临时变量一样),
    // 即使函数中的参数是引用的形式，拷贝操作(value copy)也会执行

    // 这里使用的是字符串的字面值，也就是 char const * 类型，
    // 线程的上下文完成字面值向 std::string 的转化
    int magic_number = 42;

    std::thread t{foo, magic_number, "magic number"};
    t.join();
    std::cout << "magic number: " << magic_number << std::endl;

    const char *str = "magic number pass by reference";
    // 无法保证隐式转换的操作和 std::thread 构造函数的拷贝操作的顺序
    // std::thread t1{func, magic_number, std::string(str)}; /* error? */
    std::thread t2{func, std::ref(magic_number), std::string(str)};
    t2.join();
    std::cout << "magic number: " << magic_number << std::endl;

    // std::thread 构造函数和 std::bind 的操作在标准库中以相同的机制进行定义。
    // 比如，也可以传递一个成员函数指针作为线程函数，并提供一个合适的对象指针作为第一个参数：
    // std::thread 构造函数的第三个传递参数才是可调用函数的第一个参数
    X my_x;

    std::thread t3{&X::do_lengthy_work, &my_x, std::string(str)};
    t3.join();

    // std::move “移动”是指原始对象中的数据所有权转移给另一对象，
    // 从而这些数据就不再在原始对象中保存(译者：比较像在文本编辑的剪切操作)。
    std::unique_ptr<BigObject> ptr{new BigObject};
    ptr->prepare_data(42);
    std::cout << "the data: " << ptr->get_data() << "\n";

    std::thread t5{process_big_object, std::move(ptr)};

    // 通过在 std::thread 构造函数中执行 std::move(ptr),
    // BigObject 对象的所有权首先被转移到新创建线程的的内部存储中,
    // 之后再传递给 process_big_object 函数.
    // std::cout << "the data: " << ptr->get_data() << "\n"; /* runtime error */
    t5.join();

    // 转移所有权, C++标准库中有很多资源占有(resource-owning)类型，
    // 比如 std::ifstream | std::unique_ptr | std::thread 都是可移动，但不可复制
    std::thread move_t1{some_function};
    std::thread move_t2 = std::move(move_t1);
    move_t1             = std::thread{some_other_function};
    std::thread move_t3;
    move_t3 = std::move(move_t2);
    // move_t1 = std::move(move_t3); /* runtime error */

    move_t1.join();
    if (move_t2.joinable()) /* false */
    {
        // 当某个对象转移了线程的所有权，就不能对线程进行汇入或分离
        std::cout << "move_t2 can be join or detach\n";
        move_t2.join();
    }
    move_t3.join();

    f();

    // std::thread::hardware_concurrency() 在新版C++中非常有用，其会返回并发线程的数量。
    // 例如，多核系统中，返回值可以是CPU核芯的数量。
    // 返回值也仅仅是一个标识，当无法获取时，函数返回0。
    unsigned int num_core = std::thread::hardware_concurrency();
    if (!num_core)
    {
        std::cout << "could not acquire the number of core for host computer\n";
    }
    std::cout << "the number of core for the host computer: " << num_core << "\n";

    // 确定线程数量和并行计算
    std::vector<int>   vec_integer{1, 3, 5, 9, 8, 7, 4, 2, 6};
    std::vector<float> vec_float{1.1f, 3.3f, 5.5f, 9.9f, 8.8f, 7.7f, 4.4f, 2.2f, 6.6f};

    std::cout << "the result of integer: "
              << parallel_accumulate<std::vector<int>::iterator, int>(vec_integer.begin(), vec_integer.end(), 0)
              << "\n";
    std::cout << "the result of float: "
              << parallel_accumulate<std::vector<float>::iterator, float>(vec_float.begin(), vec_float.end(), 0.f)
              << "\n";

    // 线程标识为 std::thread::id 类型，可以通过两种方式进行检索
    // 第一种，可以通过调用 std::thread 对象的成员函数 get_id() 来直接获取。
    // 如果 std::thread 对象没有与任何执行线程相关联， get_id() 将返回 std::thread::type
    // 默认构造值，这个值表示“无线程”。第二种，当前线程中调用 std::this_thread::get_id()
    std::thread threadID_1{[]()
                           {
                               std::cout << "this thread ID: " << std::this_thread::get_id() << std::endl;
                           }};
    std::thread threadID_2{some_function};
    // 具体的输出结果是严格依赖于具体实现的，C++标准的要求就是保证ID相同的线程必须有相同的输出
    std::cout << "the thread ID: " << threadID_2.get_id() << std::endl;
    threadID_1.join();
    threadID_2.join();
    // 有可能输出的结果并不如我们的预期, 线程的执行顺序没有控制

    some_core_part_of_algorithm();
    std::thread task_thread{some_core_part_of_algorithm};
    task_thread.join();
    // ? 只是为了说明问题和流程的代码
    std::cout << "the master thread ID: " << master_thread << "\n";

    std::cout << "------ main thread ending ------\n";

    return 0;
}