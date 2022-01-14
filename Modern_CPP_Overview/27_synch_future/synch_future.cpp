/**
 * @file synch_future.cpp
 * @author Wei Li (2694048168@qq.com)
 * @brief Future in C++11 and To get results of asynchronous tasks; thread synchronization
 * @version 0.1
 * @date 2022-01-14
 * 
 * @copyright Copyright (c) 2022
 * 
 */

#include <iostream>
#include <thread>
#include <future>

int main(int argc, char **argv)
{
    /* pack a lambda expression that returns 7 into a std::packaged_task */
    std::packaged_task<int()> task([]()
                                   { return 7; });

    /* get the future of task */
    std::future<int> result = task.get_future(); /* run task in a thread */
    std::thread(std::move(task)).detach();
    std::cout << "Waiting...";
    result.wait(); /* block until future has arrived */
    std::cout << "Done!" << std::endl
              << "future result is " << result.get() << std::endl;

    return 0;
}
