/**
 * @file testThread.cpp
 * @author Wei Li (weili_yzzcq@163.com)
 * @brief 
 * @version 0.1
 * @date 2023-12-17
 * 
 * @copyright Copyright (c) 2023
 * 
 */

#include "ThreadBase.hpp"

#include <iostream>

class ThreadTask : public ThreadBase
{
public:
    ThreadTask() {}

    ~ThreadTask() {}

    virtual void Run() override
    {
        int i = 5;
        while (i)
        {
            Sleep(1000);

            std::cout << "hello world\n";
            i--;
        }
    }
};

// ======================================
int main(int argc, const char **argv)
{
    ThreadTask *task = new ThreadTask;

    task->Start();
    task->SetThreadEnd();

    task->Join();
    std::cout << "子线程结束\n";

    return 0;
}
