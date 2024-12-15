/**
 * @file 14_condition_variable.cpp
 * @author Wei Li (weili_yzzcq@163.com)
 * @brief 
 * @version 0.1
 * @date 2024-12-15
 * 
 * @copyright Copyright (c) 2024
 * 
 */

#include <condition_variable>
#include <iostream>
#include <mutex>

// 使用条件变量实现的简单的生产者和消费者
void cv_example()
{
    std::mutex              m;
    std::condition_variable cv;

    bool terminate   = false;
    int  shared_data = 0;
    int  counter     = 0;

    auto consumer = [&]()
    {
        std::unique_lock<std::mutex> lk(m);
        do
        {
            while (!(terminate || shared_data != 0)) cv.wait(lk);
            if (terminate)
                break;
            std::cout << "consuming " << shared_data << std::endl;
            shared_data = 0;
            cv.notify_one();
        }
        while (true);
    };

    auto producer = [&]()
    {
        std::unique_lock<std::mutex> lk(m);
        for (counter = 1; true; ++counter)
        {
            cv.wait(lk, [&]() { return terminate || shared_data == 0; });
            if (terminate)
                break;
            shared_data = counter;
            std::cout << "producing " << shared_data << std::endl;
            cv.notify_one();
        }
    };

    auto p = std::thread(producer);
    auto c = std::thread(consumer);
    std::this_thread::sleep_for(std::chrono::milliseconds(1000));
    {
        std::lock_guard<std::mutex> l(m);
        terminate = true;
    }
    std::cout << "total items consumed " << counter << std::endl;
    cv.notify_all();

    p.join();
    c.join();

    exit(0);
}

int main(int argc, const char *argv[])
{
    cv_example();

    return 0;
}
