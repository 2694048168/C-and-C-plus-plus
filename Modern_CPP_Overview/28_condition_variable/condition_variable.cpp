/**
 * @file condition_variable.cpp
 * @author Wei Li (2694048168@qq.com)
 * @brief Condition Variable;
 * @version 0.1
 * @date 2022-01-14
 * 
 * @copyright Copyright (c) 2022
 * 
 */

#include <queue>
#include <chrono>
#include <mutex>
#include <thread>
#include <iostream>
#include <condition_variable>

int main(int argc, char const *argv[])
{
    std::queue<int> produced_nums;
    std::mutex mtx;
    std::condition_variable cv;
    bool notified = false; /* notification sign */

    auto producer = [&]()
    {
        for (int i = 0;; i++)
        {
            std::this_thread::sleep_for(std::chrono::milliseconds(500));
            std::unique_lock<std::mutex> lock(mtx);
            std::cout << "producing " << i << std::endl;
            produced_nums.push(i);
            notified = true;
            cv.notify_all();
        }
    };
    auto consumer = [&]()
    {
        while (true)
        {
            std::unique_lock<std::mutex> lock(mtx);
            while (!notified)
            {
                /* avoid spurious wakeup */
                cv.wait(lock);
            }

            /* temporal unlock to allow producer produces more rather than
            let consumer hold the lock until its consumed. */
            lock.unlock();
            std::this_thread::sleep_for(std::chrono::milliseconds(1000)); /* consumer is slower */
            lock.lock();
            if (!produced_nums.empty())
            {
                std::cout << "consuming " << produced_nums.front() << std::endl;
                produced_nums.pop();
            }
            notified = false;
        }
    };

    std::thread p(producer);
    std::thread cs[2];
    for (int i = 0; i < 2; ++i)
    {
        cs[i] = std::thread(consumer);
    }
    p.join();
    for (int i = 0; i < 2; ++i)
    {
        cs[i].join();
    }

    return 0;
}
