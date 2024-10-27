/**
 * @file 13_log_pattern.cpp
 * @author Wei Li (weili_yzzcq@163.com)
 * @brief 
 * @version 0.1
 * @date 2024-10-26
 * 
 * @copyright Copyright (c) 2024
 * 
 */

#include "spdlog/pattern_formatter.h"
#include "spdlog/sinks/stdout_color_sinks.h"
#include "spdlog/spdlog.h"

#include <memory>
#include <string>

// 定义自定义的格式化标志类
class my_formatter_flag : public spdlog::custom_flag_formatter
{
public:
    // 格式化方法，向日志中添加自定义文本
    void format(const spdlog::details::log_msg &, const std::tm &, spdlog::memory_buf_t &dest) override
    {
        std::string some_txt = "custom_flag";
        dest.append(some_txt.data(), some_txt.data() + some_txt.size());
    }

    // 克隆方法，用于复制自定义格式化器
    std::unique_ptr<custom_flag_formatter> clone() const override
    {
        return spdlog::details::make_unique<my_formatter_flag>();
    }
};

// -----------------------------------
int main(int argc, const char **argv)
{
    /* 14. User-defined flags in the log pattern 用户定义日志模式
    * 在 spdlog 中, 可以定义自定义的日志模式标志,
    * 通过继承 spdlog::custom_flag_formatter 类来实现.
    * 这允许将特定的信息或格式直接嵌入到日志输出的模式中, 
    * 从而增强日志的可读性或增加特定的上下文信息.
    * 
    */
    // 初始化 spdlog
    spdlog::set_level(spdlog::level::info);                         // 设置日志级别
    spdlog::set_default_logger(spdlog::stdout_color_mt("console")); // 使用彩色控制台输出

    // 创建一个自定义的日志格式化器
    auto formatter = std::make_unique<spdlog::pattern_formatter>();

    // 添加自定义标志 '%'，并设置日志输出模式
    formatter->add_flag<my_formatter_flag>('*').set_pattern("[%n] [%*] [%^%l%$] %v");

    // 设置全局日志格式化器
    spdlog::set_formatter(std::move(formatter));

    // 记录一些日志消息
    spdlog::info("This is an info message with a custom flag.");
    spdlog::warn("This is a warning message with a custom flag.");

    /* 代码解释:
    * 1. 定义自定义格式化标志类:
    * ----my_formatter_flag 类继承自 spdlog::custom_flag_formatter, 并实现了两个方法:
    * ----format: 该方法用于在日志输出中插入自定义的文本. 此示例中插入了 "custom_flag";
    * ----clone: 该方法返回一个指向新创建的 my_formatter_flag 实例的唯一指针, 用于复制格式化器;
    * 2. 创建自定义日志格式化器:
    * ----spdlog::pattern_formatter 是 spdlog 中用于定义日志模式的类;
    * ----formatter->add_flag<my_formatter_flag>('*') 将自定义标志 %* 
    *     绑定到 my_formatter_flag 实例;
    * ----formatter->set_pattern("[%n] [%*] [%^%l%$] %v"); 
    *    设置日志输出模式，包含自定义标志 %*, 
    * 3. 设置全局日志格式化器:
    * ----spdlog::set_formatter(std::move(formatter)); 
    *     将自定义的格式化器设置为全局默认的日志格式化器.
    * 4. 记录日志消息:
    * ----使用 spdlog::info 和 spdlog::warn 记录日志消息, 输出的日志将包含自定义标志生成的文本.
    * 
    * 总结: 通过扩展 spdlog::custom_flag_formatter, 
    * 可以为 spdlog 添加自定义的日志模式标志. 这使得能够根据具体需求增强日志输出的格式和内容,
    * 从而为日志增加更多的上下文信息或自定义标识.
    * 这种方法特别适用于复杂应用程序的日志管理需求, 可以根据不同的场景灵活调整日志的输出内容.
    * 
    */

    return 0;
}
