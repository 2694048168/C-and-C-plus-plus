/**
 * @file main.cpp
 * @author Wei Li (weili_yzzcq@163.com)
 * @brief 异步学习之回调函数
 * @version 0.1
 * @date 2024-08-20
 * 
 * @copyright Copyright (c) 2024
 * 
 */

/**
  * @brief 回调函数是一种编程模式,允许程序在完成某些操作后,通过一个预先定义的函数来执行额外的代码
  * *回调函数注册：回调函数的注册通常涉及到将回调函数存储在某个数据结构中,
  * *以便在特定事件发生时调用它们. 这在实现事件监听器、观察者模式或中间件时非常有用.
  * 
  */

#include <chrono>
#include <functional>
#include <iostream>
#include <thread>

void AsyncDatabaseQuery(std::function<void(const std::string &)> callback)
{
    std::cout << "开始异步查询数据库..." << std::endl;
    std::this_thread::sleep_for(std::chrono::seconds(2)); // 模拟耗时操作

    // 查询完成，调用回调函数
    std::string result = "查询结果";
    callback(result);
}

class EventManager
{
private:
    std::vector<std::function<void()>> callbacks;

public:
    // 注册回调函数
    void RegisterCallback(std::function<void()> callback)
    {
        callbacks.push_back(callback);
    }

    // 触发事件，调用所有注册的回调函数
    void TriggerEvent()
    {
        for (auto &callback : callbacks)
        {
            callback();
        }
    }
};

// -----------------------------
int main(int argc, char **argv)
{
    // 使用Lambda表达式作为回调函数
    AsyncDatabaseQuery([](const std::string &result)
                       { std::cout << "查询完成，回调函数被调用，结果: " << result << std::endl; });

    std::cout << "主线程继续执行其他任务...\n\n";
    /* 是调用完 AsyncDatabaseQuery 函数之后再调用后面的callback函数。这就是回调，
    当然主程序中最好重新创建个 线程 执行 AsyncDatabaseQuery 函数。 */

    // ================================
    EventManager eventManager;
    // 注册回调函数
    eventManager.RegisterCallback([]() { std::cout << "回调函数1被调用" << std::endl; });
    eventManager.RegisterCallback([]() { std::cout << "回调函数2被调用" << std::endl; });
    // 触发事件
    std::cout << "触发事件，所有注册的回调函数将被调用" << std::endl;
    eventManager.TriggerEvent();
    /* 这种方式允许你在不同的时间点注册多个回调函数，然后在某个事件发生时统一触发它们 */

    return 0;
}
