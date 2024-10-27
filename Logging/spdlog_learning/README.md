## spdlog learning and tutorial

> [spdlog](https://github.com/gabime/spdlog) Very fast, header-only/compiled, C++ logging library.

### quick start

```shell
mkdir spd_learning && cd spd_learning
git init
git submodule add https://github.com/gabime/spdlog

# cmake .. -G"Visual Studio 15 2017 Win64" -D CMAKE_CONFIGURATION_TYPES="Release;Debug"
# 该命令将为Release和Debug配置生成一个构建树。然后可以使用 --config 标志来决定构建这两个中的哪一个

cmake -S . -B build -G "Ninja"
cmake --build build --config Release

```

### Attention

- **Q: spdlog-日志无法及时输出到文件?**
在使用spdlog的multi-sink方法时, 如果其中一个sink是输出到文件的, 那么可能会无法及时输出日志到文件, 只有当程序关闭或有大量日志时才会写入(buffer机制); 即便设置了, 也依旧无法及时刷新.

```
spdlog::flush_on(log_level)
spdlog::flush_every(duration)
```

- **A: 原因,spdlog有全局管理的机制**
如果自行声明的logger不是通过create方法创建的(如自行实例化), 那么不会纳管到spdlog的管理集合中; 进而flush_on和flush_every不会及时生效; 解决方案: 使用spdlog::register_logger方法, 将自行实例化的logger纳入spdlog的管理范围; 或使用create方法创建; 此时flush_on和flush_every机制生效.

### spdlog learning

1. 标准输出(stdout)和标准错误(stderr)的日志
    - **00_basic_console.cpp**
2. 基本的文件日志器(basic file logger)
    - **01_basic_file.cpp**
3. 每日日志文件
    - **02_daily_file.cpp**
4. Rotating files
    - **03_rotating_file.cpp**
5. Backtrace support 缓存异常日志
    - **04_backtrace_file.cpp**
6. Periodic flush 定期刷新日志缓冲区
    - **05_flush_file.cpp**
7. StopWatch 计时工具
    - **06_stop_watch.cpp**
8. Log binary data in hex 记录二进制
    - **07_binary_data.cpp**
9. Logger with multi sinks - each with a different format and log level 设置日志级别
    - **08_multi_sinks.cpp**
10. User-defined callbacks about log events 用户定义回调
    - **09_callback_log.cpp**
11. Asynchronous logging 异步日志记录
    - **10_synchrony_log.cpp**
12. Asynchronous logger with multi sinks  带sink异步日志记录器
    - **11_synchrony_multi_sinks.cpp**
13. User-defined types用户定义类型
    - **12_user_type.cpp**
14. User-defined flags in the log pattern 用户定义日志模式
    - **13_log_pattern.cpp**
15. Custom error handler 自定义错误处理
    - **14_error_handler.cpp**
16 syslog 系统日志
    - **15_system_log.cpp**
17. Android example
    - **16_android_log.cpp**
18.Load log levels from the env variable or argv
    - **17_command_line.cpp**
19. Log file open/close event handlers 打开和关闭事件注册回调函数
    - **18_event_handler.cpp**
20. Replace the Default Logger 替换默认的日志记录器
    - **19_default_logger.cpp**
21. Log to Qt with nice colors
    - **20_qt_color.cpp**
22. Mapped Diagnostic Context 附加特定的上下文信息（如用户 ID、会话 ID 等）
    - **21_mapped_context.cpp**
