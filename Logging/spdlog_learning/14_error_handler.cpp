/**
 * @file 14_error_handler.cpp
 * @author Wei Li (weili_yzzcq@163.com)
 * @brief 
 * @version 0.1
 * @date 2024-10-26
 * 
 * @copyright Copyright (c) 2024
 * 
 */

#include "spdlog/sinks/stdout_color_sinks.h"
#include "spdlog/spdlog.h"

#include <string>

// -----------------------------------
int main(int argc, const char **argv)
{
    /* 15. Custom error handler 自定义错误处理
    * 在 spdlog 中, 可以设置自定义错误处理程序, 
    * 当日志记录过程中发生错误时, 该错误处理程序将被调用.
    * 错误处理程序可以是全局的, 也可以是针对特定日志记录器的.
    * 这样可以在遇到日志记录问题时执行自定义操作, 比如记录错误信息、发送警报等.
    *  
    */

    // 设置一个默认的控制台日志记录器
    auto console = spdlog::stdout_color_mt("console");

    // 设置全局错误处理程序
    spdlog::set_error_handler(
        [](const std::string &msg)
        {
            // 当发生日志错误时，记录一条错误信息
            spdlog::get("console")->error("*** LOGGER ERROR ***: {}", msg);
        });

    // 试图记录一条格式错误的日志消息以触发错误
    spdlog::get("console")->info("some invalid message to trigger an error {}{}{}{}", 3);

    /* 代码解释:
    * 1. 创建日志记录器:
    * ----使用 spdlog::stdout_color_mt("console") 创建了一个彩色控制台日志记录器,并将其命名为 "console".
    * 2. 设置全局错误处理程序:
    * ----使用 spdlog::set_error_handler 设置全局错误处理程序.
    * ----错误处理程序是一个 lambda 函数, 它接受一个 std::string 类型的参数 msg,该参数包含了错误信息.
    * ----当日志记录过程中发生错误时, 该处理程序会被调用, 并将错误信息输出到控制台.
    * 3.触发日志记录错误:
    * ----试图记录一条格式错误的日志消息 
    * spdlog::get("console")->info("some invalid message to trigger an error {}{}{}{}", 3);
    * 由于占位符 {} 的数量不匹配, 这会触发一个日志记录错误.
    * 4. 错误处理: 当上述错误发生时, 之前设置的错误处理程序会被调用, 并将错误信息记录下来.
    * 
    * 其他注意事项:
    * 1. 局部错误处理程序: 如果只想为某个特定的日志记录器设置错误处理程序,
    *    而不是全局设置, 可以使用 logger->set_error_handler(...) 来为特定的日志记录器设置错误处理程序.
    * 2. 错误处理的应用场景: 自定义错误处理程序对于处理和记录日志错误信息非常有用,
    *    尤其是在需要监控日志系统健康状况或在日志系统出现问题时采取纠正措施的场景中.
    * 
    */

    return 0;
}
