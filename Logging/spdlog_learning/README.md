## spdlog learning and tutorial

> [spdlog](https://github.com/gabime/spdlog) Very fast, header-only/compiled, C++ logging library.

### quick start

```shell
mkdir spd_learning && cd spd_learning
git init
git submodule add https://github.com/gabime/spdlog

# cmake .. -G"Visual Studio 15 2017 Win64" -D CMAKE_CONFIGURATION_TYPES="Release;Debug"
# 该命令将为Release和Debug配置生成一个构建树。然后可以使用 --config 标志来决定构建这两个中的哪一个

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
