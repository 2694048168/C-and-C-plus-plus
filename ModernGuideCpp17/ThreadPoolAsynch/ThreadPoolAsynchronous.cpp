#include "ThreadPoolAsynchronous.hpp"

#include <chrono>
#include <iostream>

ThreadPoolAsynchronous::ThreadPoolAsynchronous(int min, int max)
    : m_minThreads(min)
    , m_maxThreads(max)
    , m_stop(false)
    , m_exitNumber(0)
{
    //m_idleThreads = m_curThreads = max / 2;
    m_idleThreads = m_curThreads = min;
    std::cout << "线程数量: " << m_curThreads << std::endl;
    m_pManager = new std::thread(&ThreadPoolAsynchronous::manager, this);
    for (int i = 0; i < m_curThreads; ++i)
    {
        std::thread t(&ThreadPoolAsynchronous::worker, this);
        m_workers.insert(std::make_pair(t.get_id(), std::move(t)));
    }
}

ThreadPoolAsynchronous::~ThreadPoolAsynchronous()
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

void ThreadPoolAsynchronous::addTask(std::function<void()> f)
{
    {
        std::lock_guard<std::mutex> locker(m_queueMutex);
        m_tasks.emplace(f);
    }
    m_condition.notify_one();
}

void ThreadPoolAsynchronous::manager()
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
            std::thread t(&ThreadPoolAsynchronous::worker, this);
            std::cout << "+++++++++++++++ 添加了一个线程, id: " << t.get_id() << std::endl;
            m_workers.insert(std::make_pair(t.get_id(), std::move(t)));
            m_curThreads++;
            m_idleThreads++;
        }
    }
}

void ThreadPoolAsynchronous::worker()
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
