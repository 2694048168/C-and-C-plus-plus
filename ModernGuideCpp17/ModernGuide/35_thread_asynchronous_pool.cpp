/**
 * @file 35_thread_asynchronous_pool.cpp
 * @author Wei Li (weili_yzzcq@163.com)
 * @brief 异步线程池(基于C++11)
 * @version 0.1
 * @date 2024-09-08
 * 
 * @copyright Copyright (c) 2024
 * 
 * https://subingwen.cn/cpp/threadpool/#1-%E7%BA%BF%E7%A8%8B%E6%B1%A0%E7%9A%84%E8%AE%BE%E8%AE%A1
 */

/** 
 * ?线程池是一种用于管理和重用线程的技术,广泛用于需要大量短生命周期线程的应用场景,
 * 如并发任务处理、网络服务和高性能计算等. 使用线程池可以有效减少线程创建和销毁的开销,提升系统性能.
 * *线程池的基本思想是预先创建一定数量的线程,并将它们放入一个池中.
 * *线程池负责管理线程的生命周期,并将任务分配给空闲线程执行.这样可以避免每次任务执行时都创建和销毁线程的开销.
 * 
 * =====线程异步
 * 线程异步(Asynchronous Threading)是一种编程范式,
 * *用于执行任务或操作而不阻塞主线程或其他线程的执行.
 * 这种方法特别适用于需要同时处理多个操作或在后台执行长时间运行的任务的场景,
 * 线程异步的核心思想是将耗时的操作与主执行流程分离,使得系统能够继续处理其他任务,而无需等待耗时操作完成.
 *
 * ?异步执行:与同步操作不同,异步操作不要求调用者在任务完成之前等待结果.
 * 异步操作通常会在后台线程中执行,主线程或其他线程可以继续执行其他任务.
 * 线程: 在多线程编程中, 异步操作通常通过创建新的线程来实现.
 * 新线程会执行异步任务, 而主线程则继续进行其他操作.
 * 
 */

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <functional>
#include <future>
#include <iostream>
#include <map>
#include <mutex>
#include <queue>
#include <thread>
#include <vector>

// 线程池类
class ThreadPool
{
public:
    ThreadPool(int min, int max = std::thread::hardware_concurrency())
        : m_minThreads(min)
        , m_maxThreads(max)
        , m_stop(false)
        , m_exitNumber(0)
    {
        //m_idleThreads = m_curThreads = max / 2;
        m_idleThreads = m_curThreads = min;
        std::cout << "线程数量: " << m_curThreads << std::endl;
        m_pManager = new std::thread(&ThreadPool::manager, this);
        for (int i = 0; i < m_curThreads; ++i)
        {
            std::thread t(&ThreadPool::worker, this);
            m_workers.insert(std::make_pair(t.get_id(), std::move(t)));
        }
    }

    ~ThreadPool()
    {
        m_stop = true;
        m_condition.notify_all();
        for (auto &it : m_workers)
        {
            std::thread &t = it.second;
            if (t.joinable())
            {
                std::cout << "******** 线程 " << t.get_id() << " 将要退出了...\n";
                t.join();
            }
        }
        if (m_pManager->joinable())
        {
            m_pManager->join();
        }
        if (m_pManager)
        {
            delete m_pManager;
            m_pManager = nullptr;
        }
    }

    template<typename F, typename... Args>
    auto addTask(F &&f, Args &&...args) -> std::future<typename std::result_of<F(Args...)>::type>
    {
        using returnType = typename std::result_of<F(Args...)>::type;
        auto task        = std::make_shared<std::packaged_task<returnType()>>(
            std::bind(std::forward<F>(f), std::forward<Args>(args)...));
        std::future<returnType> res = task->get_future();

        {
            std::unique_lock<std::mutex> lock(m_queueMutex);
            m_tasks.emplace([task]() { (*task)(); });
        }

        m_condition.notify_one();
        return res;
    }

private:
    void manager()
    {
        while (!m_stop.load())
        {
            std::this_thread::sleep_for(std::chrono::seconds(2));
            int idle    = m_idleThreads.load();
            int current = m_curThreads.load();
            if (idle > current / 2 && current > m_minThreads)
            {
                m_exitNumber.store(2);
                m_condition.notify_all();
                std::unique_lock<std::mutex> lck(m_idsMutex);
                for (const auto &id : m_ids)
                {
                    auto it = m_workers.find(id);
                    if (it != m_workers.end())
                    {
                        std::cout << "############## 线程 " << (*it).first << "被销毁了....\n";
                        (*it).second.join();
                        m_workers.erase(it);
                    }
                }
                m_ids.clear();
            }
            else if (idle == 0 && current < m_maxThreads)
            {
                std::thread t(&ThreadPool::worker, this);
                std::cout << "+++++++++++++++ 添加了一个线程, id: " << t.get_id() << std::endl;
                m_workers.insert(std::make_pair(t.get_id(), std::move(t)));
                m_curThreads++;
                m_idleThreads++;
            }
        }
    }

    void worker()
    {
        while (!m_stop.load())
        {
            std::function<void()> task = nullptr;
            {
                std::unique_lock<std::mutex> locker(m_queueMutex);
                while (!m_stop && m_tasks.empty())
                {
                    m_condition.wait(locker);
                    if (m_exitNumber.load() > 0)
                    {
                        std::cout << "----------------- 线程任务结束, ID: " << std::this_thread::get_id() << std::endl;
                        m_exitNumber--;
                        m_curThreads--;
                        m_idleThreads--;
                        std::unique_lock<std::mutex> lck(m_idsMutex);
                        m_ids.emplace_back(std::this_thread::get_id());
                        return;
                    }
                }

                if (!m_tasks.empty())
                {
                    std::cout << "取出一个任务...\n";
                    task = std::move(m_tasks.front());
                    m_tasks.pop();
                }
            }

            if (task)
            {
                m_idleThreads--;
                task();
                m_idleThreads++;
            }
        }
    }

private:
    std::thread                           *m_pManager;
    std::map<std::thread::id, std::thread> m_workers;
    std::vector<std::thread::id>           m_ids;
    int                                    m_minThreads;
    int                                    m_maxThreads;
    std::atomic<bool>                      m_stop;
    std::atomic<int>                       m_curThreads;
    std::atomic<int>                       m_idleThreads;
    std::atomic<int>                       m_exitNumber;
    std::queue<std::function<void()>>      m_tasks;
    std::mutex                             m_idsMutex;
    std::mutex                             m_queueMutex;
    std::condition_variable                m_condition;
};

int calc(int x, int y)
{
    int res = x + y;
    //cout << "res = " << res << endl;
    std::this_thread::sleep_for(std::chrono::seconds(2));
    return res;
}

// ------------------------------------
int main(int argc, const char **argv)
{
    ThreadPool                    pool(4);
    std::vector<std::future<int>> results;

    for (int i = 0; i < 10; ++i)
    {
        results.emplace_back(pool.addTask(calc, i, i * 2));
    }

    // 等待并打印结果
    for (auto &&res : results)
    {
        std::cout << "线程函数返回值: " << res.get() << std::endl;
    }

    return 0;
}
