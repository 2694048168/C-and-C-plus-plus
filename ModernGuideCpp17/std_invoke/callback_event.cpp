/**
 * @file callback_event.cpp
 * @author Wei Li (Ithaca) (weili_yzzca@163.com)
 * @brief 
 * @version 0.1
 * @date 2025-10-17
 * 
 * @copyright Copyright (c) 2025
 * 
 */

#include <any>
#include <functional>
#include <iostream>
#include <map>
#include <string>
#include <vector>

/**
 * @brief 使用 std::invoke简化实现
 * 泛型回调与事件系统
 * 
 * g++ callback_event.cpp -std=c++20
 * clang++ callback_event.cpp -std=c++20
 */

// 简单的日志记录器，用于演示成员函数回调
struct Logger
{
    void log(const std::string &message)
    {
        std::cout << "[Logger]: " << message << std::endl;
    }
};

// 优化后的事件处理器
class EventHandlerAfter
{
private:
    // 使用 std::function 存储任何可调用对象
    std::map<std::string, std::function<void(const std::string &)>> connections;

public:
    /**
     * @brief 注册一个回调。std::invoke 的思想让这种封装变得简单。
     * @tparam Callable 回调的类型 (函数, lambda, 成员函数指针等)
     * @tparam Args 绑定到回调的参数 (例如 'this' 指针)
     * @param event_name 事件名称
     * @param callable 可调用实体
     * @param args 调用时需要的第一个参数 (对于成员函数，通常是对象实例)
     */
    template<typename Callable, typename... Args>
    void connect(const std::string &event_name, Callable &&callable, Args &&...args)
    {
        // 使用 lambda 和 std::invoke 捕获回调和参数
        connections[event_name] = [c              = std::forward<Callable>(callable),
                                   ... bound_args = std::forward<Args>(args)](const std::string &payload) mutable
        {
            // 统一的调用语法！
            std::invoke(c, bound_args..., payload);
        };
    }

    /**
     * @brief 触发一个事件
     * @param event_name 事件名称
     * @param payload 事件负载
     */
    void emit(const std::string &event_name, const std::string &payload)
    {
        if (connections.count(event_name))
        {
            connections[event_name](payload);
        }
    }
};

void global_event_handler(const std::string &payload)
{
    std::cout << "[Global]: Received '" << payload << "'\n";
}

// -------------------------------------
int main(int argc, const char *argv[])
{
    EventHandlerAfter handler;
    Logger            my_logger;

    // 1. 注册一个全局函数
    handler.connect("data_received", &global_event_handler);

    // 2. 注册一个 Lambda
    handler.connect("error_occurred",
                    [](const std::string &err) { std::cerr << "[Lambda]: Error - " << err << std::endl; });

    // 3. 注册一个成员函数
    handler.connect("data_received", &Logger::log, &my_logger);

    // 触发事件，所有回调都能被正确调用
    handler.emit("data_received", "Packet #123");
    handler.emit("error_occurred", "Connection timed out");

    return 0;
}
