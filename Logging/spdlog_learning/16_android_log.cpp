/**
 * @file 16_android_log.cpp
 * @author Wei Li (weili_yzzcq@163.com)
 * @brief 
 * @version 0.1
 * @date 2024-10-26
 * 
 * @copyright Copyright (c) 2024
 * 
 */

#include "spdlog/sinks/android_sink.h"
#include "spdlog/spdlog.h"

#include <string>

int main(int argc, const char **argv)
{
    /* 17. Android example
    * 在 Android 平台上, spdlog 提供了一个 android_sink,
    * 用于将日志消息输出到 Android 的日志系统（logcat）.
    * 这使得开发者可以将 spdlog 与 Android 的日志系统集成,
    * 在开发和调试应用程序时更加方便地查看日志输出.
    */
    // 定义日志的标签
    std::string tag = "spdlog-android";

    // 创建一个 Android 日志记录器
    auto android_logger = spdlog::android_logger_mt("android", tag);

    // 记录一条严重错误级别的日志消息
    android_logger->critical("Use \"adb shell logcat\" to view this message.");

    /* 代码解释:
    * 1. 包含必要的头文件:
    * ----spdlog/spdlog.h 是 spdlog 的主头文件, 包含了日志记录功能的核心内容;
    * ----spdlog/sinks/android_sink.h 包含了 android_sink,用于将日志消息发送到 Android 的日志系统;
    * 2. 定义日志的标签:
    * ----std::string tag = "spdlog-android"; 定义了日志消息的标签(tag).
    * ----在 Android 的 logcat 中, 可以通过这个标签来过滤日志消息;
    * 3. 创建 Android 日志记录器:
    * ----spdlog::android_logger_mt("android", tag); 创建了一个 Android 日志记录器;
    * ----"android" 是日志记录器的名称;
    * ----tag 是日志消息的标签;
    * 4. 记录日志到 Android 日志系统;
    *   查看日志, 在运行上述代码后, 使用以下命令查看日志消息:
    * adb shell logcat -s spdlog-android
    * ----这将显示带有 spdlog-android 标签的所有日志消息,输出类似以下内容:
    * E/spdlog-android(12345): Use "adb shell logcat" to view this message.
    * ---E 表示日志级别为 Error（与 critical 对应）;
    * ---spdlog-android 是我们设置的日志标签;
    * ---12345 是 Android 应用进程的 ID;
    * ---Use "adb shell logcat" to view this message. 是日志消息的内容.
    */

    return 0;
}
