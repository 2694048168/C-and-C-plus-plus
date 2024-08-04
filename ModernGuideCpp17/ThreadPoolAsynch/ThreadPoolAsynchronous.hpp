/**
 * @file ThreadPoolAsynchronous.hpp
 * @author Wei Li (weili_yzzcq@163.com)
 * @brief https://subingwen.cn/cpp/threadpool/
 * @version 0.1
 * @date 2024-08-04
 * 
 * @copyright Copyright (c) 2024
 * 
 * @note https://www.bilibili.com/video/BV1fw4m1r7cT
 * 
 */

#pragma once

#include <atomic>
#include <condition_variable>
#include <functional>
#include <future>
#include <map>
#include <mutex>
#include <queue>
#include <thread>
#include <type_traits>
#include <vector>


class ThreadPoolAsynchronous
{
public:
    ThreadPoolAsynchronous(int min = 2, int max = std::thread::hardware_concurrency());
    ~ThreadPoolAsynchronous();

    void addTask(std::function<void()> f);

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
    void manager();
    void worker();

private:
    std::thread                           *m_pManager;
    std::map<std::thread::id, std::thread> m_workers;
    std::vector<std::thread::id>           m_ids;

    int m_minThreads;
    int m_maxThreads;

    std::atomic<bool> m_stop;
    std::atomic<int>  m_curThreads;
    std::atomic<int>  m_idleThreads;
    std::atomic<int>  m_exitNumber;

    std::queue<std::function<void()>> m_tasks;
    std::mutex                        m_idsMutex;
    std::mutex                        m_queueMutex;
    std::condition_variable           m_condition;
};
