/**
 * @file TaskQueue.cpp
 * @author Wei Li (weili_yzzcq@163.com)
 * @brief 使用单例模式设计一个任务队列
 * @version 0.1
 * @date 2024-01-14
 * 
 * @copyright Copyright (c) 2024
 * 
 */

#include <atomic>
#include <iostream>
#include <mutex>
#include <queue>
#include <thread>

#if 0
// 定义一个单例模式的任务队列
class TaskQueue
{
public:
    // TaskQueue()                   = delete;
    TaskQueue(const TaskQueue &t)            = delete;
    TaskQueue &operator=(const TaskQueue &t) = delete;

    static TaskQueue *getInstance()
    {
        return m_taskQueue;
    }

    void printInfo()
    {
        std::cout << "This is a Singleten object\n";
    }

private:
    TaskQueue() = default;
    // TaskQueue(const TaskQueue &t) = default;
    // TaskQueue& operator=(const TaskQueue &t) = default;

    static TaskQueue *m_taskQueue;
};
// 饿汉模式, 定义类的时候就创建(new)该单例对象
// 没有多线程安全问题
TaskQueue *TaskQueue::m_taskQueue = new TaskQueue;
#endif

#if 0
// 定义一个单例模式的任务队列
class TaskQueue
{
public:
    // TaskQueue()                   = delete;
    TaskQueue(const TaskQueue &t)            = delete;
    TaskQueue &operator=(const TaskQueue &t) = delete;

    static TaskQueue *getInstance()
    {
        // 双重检查锁定, 多线程执行时候存在内存一致性问题
        // 机器指令角度多条指定的顺序存在重新排序, 不一定按照我们预期的执行, 原子操作
        // 但是实际上 m_taskQ = new TaskQueue; 在执行过程中对应的机器指令可能会被重新排序
        if (m_taskQueue == nullptr)
        {
            m_mutex.lock();
            if (m_taskQueue == nullptr)
                m_taskQueue = new TaskQueue(); // 不加锁存在线程不安全问题
            m_mutex.unlock();                  // 加锁后导致多线程访问该单例对象的效率低
        }

        return m_taskQueue;
    }

    void printInfo()
    {
        std::cout << "This is a Singleten object\n";
    }

private:
    TaskQueue() = default;
    // TaskQueue(const TaskQueue &t) = default;
    // TaskQueue& operator=(const TaskQueue &t) = default;

    static TaskQueue *m_taskQueue;
    static std::mutex m_mutex;
};

// 懒汉模式, 什么时候使用该单例对象再创建(new)
TaskQueue *TaskQueue::m_taskQueue = nullptr;
std::mutex TaskQueue::m_mutex;
#endif

#if 0
// 定义一个单例模式的任务队列
class TaskQueue
{
public:
    // TaskQueue()                   = delete;
    TaskQueue(const TaskQueue &t)            = delete;
    TaskQueue &operator=(const TaskQueue &t) = delete;

    static TaskQueue *getInstance()
    {
        TaskQueue *taskQueue = m_taskQueue.load();
        if (taskQueue == nullptr)
        {
            std::lock_guard<std::mutex> locker(m_mutex);
            taskQueue = m_taskQueue.load();
            if (taskQueue == nullptr)
            {
                taskQueue = new TaskQueue;
                m_taskQueue.store(taskQueue);
            }
        }

        return taskQueue;
    }

    void printInfo()
    {
        std::cout << "This is a Singleten object\nHello world\n";
    }

private:
    TaskQueue() = default;
    // TaskQueue(const TaskQueue &t) = default;
    // TaskQueue& operator=(const TaskQueue &t) = default;

    /** 使用原子变量atomic的store() 方法来存储单例对象，使用load() 方法来加载单例对象。
    * 在原子变量中这两个函数在处理指令的时候默认的原子顺序是:
    *  memory_order_seq_cst（顺序原子操作 - sequentially consistent），
    * 使用顺序约束原子操作库，整个函数执行都将保证顺序执行，并且不会出现数据竞态（data races）
    * 不足之处就是使用这种方法实现的懒汉模式的单例执行效率更低一些。
    */
    static std::atomic<TaskQueue *> m_taskQueue; // 原子变量
    static std::mutex               m_mutex;
};

// 懒汉模式, 什么时候使用该单例对象再创建(new)
std::atomic<TaskQueue *> TaskQueue::m_taskQueue;
std::mutex               TaskQueue::m_mutex;
#endif

#if 0
// 定义一个单例模式的任务队列
class TaskQueue
{
public:
    // TaskQueue()                   = delete;
    TaskQueue(const TaskQueue &t)            = delete;
    TaskQueue &operator=(const TaskQueue &t) = delete;

    static TaskQueue *getInstance()
    {
        /**
         * @brief 静态局部队列对象，并且将这个对象作为了唯一的单例实例。
         * 使用这种方式之所以是线程安全的，是因为在C++11标准中有如下规定，
         * 并且这个操作是在编译时由编译器保证的：
         * 如果指令逻辑进入一个未被初始化的声明变量，所有并发执行应当等待该变量完成初始化。
         */
        static TaskQueue taskQueue;

        return &taskQueue;
    }

    void printInfo()
    {
        std::cout << "This is a Singleten object\nHello world\n";
    }

private:
    TaskQueue() = default;
    // TaskQueue(const TaskQueue &t) = default;
    // TaskQueue& operator=(const TaskQueue &t) = default;
};
#endif

/**
 * @brief 设计一个单例模式的任务队列，那么就需要赋予这个类一些属性和方法：
 * 属性：
 * 1. 存储任务的容器，这个容器可以选择使用STL中的队列（queue)
 * 2. 互斥锁，多线程访问的时候用于保护任务队列中的数据
 *
 * 方法：主要是对任务队列中的任务进行操作:
 * 1. 任务队列中任务是否为空
 * 2. 往任务队列中添加一个任务
 * 3. 从任务队列中取出一个任务
 * 4. 从任务队列中删除一个任务
 * 
 * 注：正常情况下，任务队列中的任务应该是一个函数指针(这个指针指向的函数中有需要执行的任务动作)
 * 此处进行了简化，用一个整形数代替了任务队列中的任务。
 * 任务队列中的互斥锁保护的是单例对象的中的数据也就是任务队列中的数据，
 * 所说的线程安全指的是在创建单例对象的时候要保证这个对象只被创建一次，
 * 和此处完全是两码事儿，需要区别看待。
 * 
 */
// template<typename DataType>
class TaskQueue
{
public:
    TaskQueue(const TaskQueue &obj)            = delete;
    TaskQueue &operator=(const TaskQueue &obj) = delete;

    static TaskQueue *getInstance()
    {
        return &m_obj;
    }

    // 任务队列是否为空
    bool isEmpty()
    {
        std::lock_guard<std::mutex> locker(m_mutex);
        return m_taskQ.empty();
    }

    // 添加任务
    void addTask(int data)
    {
        std::lock_guard<std::mutex> locker(m_mutex);
        m_taskQ.push(data);
    }

    // 取出一个任务
    int takeTask()
    {
        std::lock_guard<std::mutex> locker(m_mutex);
        if (!m_taskQ.empty())
        {
            return m_taskQ.front();
        }
        return -1;
    }

    // 删除一个任务
    bool popTask()
    {
        std::lock_guard<std::mutex> locker(m_mutex);
        if (!m_taskQ.empty())
        {
            m_taskQ.pop();
            return true;
        }
        return false;
    }

private:
    TaskQueue() = default;
    static TaskQueue m_obj;
    std::queue<int>  m_taskQ;
    std::mutex       m_mutex;
};

TaskQueue TaskQueue::m_obj;

// =============================
int main(int argc, const char **argv)
{
    // TaskQueue *p_taskQueue = TaskQueue::getInstance();
    // p_taskQueue->printInfo();

    std::thread thread_producer(
        []()
        {
            TaskQueue *taskQ = TaskQueue::getInstance();
            for (int i = 0; i < 10; ++i)
            {
                taskQ->addTask(i + 100);
                std::cout << "+++push task: " << i + 100 << ", threadID: " << std::this_thread::get_id() << std::endl;
                std::this_thread::sleep_for(std::chrono::milliseconds(500));
            }
        });
    std::thread thread_consumer(
        []()
        {
            TaskQueue *taskQ = TaskQueue::getInstance();
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
            while (!taskQ->isEmpty())
            {
                int data = taskQ->takeTask();
                std::cout << "---take task: " << data << ", threadID: " << std::this_thread::get_id() << std::endl;
                taskQ->popTask();
                std::this_thread::sleep_for(std::chrono::seconds(1));
            }
        });

    thread_producer.join();
    thread_consumer.join();

    return 0;
}
