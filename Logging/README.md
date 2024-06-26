## Logging Library

> 日志在软件开发和维护中扮演着至关重要的角色。不仅帮助开发者追踪程序运行状态，还能在出现问题时提供宝贵的调试信息。通过日志能够了解软件在特定时间点的行为，分析性能瓶颈，甚至预测潜在的系统故障。日志的重要性不言而喻，它就像是软件世界中的“黑匣子”，记录着程序的每一个细微动作。


### 选择合适的日志库

> 选择合适的日志库对于确保日志的有效性和效率至关重要。有许多优秀的C++日志库，如glog、log4cplus和spdlog，各有千秋，提供了丰富的功能和灵活的配置选项。深入探讨这三个日志库的底层原理和性能特点，根据需求做出明智的选择。

> 从多个角度对glog、log4cplus和spdlog 三个C++日志库进行详细的比较和分析。探讨它们的底层实现，性能特点，以及如何控制日志输出到控制台和文件。能够更加深入地理解这些日志库的工作原理，从而在实际开发中做出更加明智的选择。逐一深入探讨这三个日志库，揭示它们的内部机制和性能表现，提供全面而深刻的见解。

### Underlying Principles of glog

- [glog on GitHub](https://github.com/google/glog)
- glog的设计充分考虑了性能和灵活性
    - 高效的日志级别管理 (Efficient Log Level Management)
        - glog 通过预处理指令在编译时确定日志级别, 这样在运行时就能快速判断是否需要记录某条日志信息. 例如设置了日志级别为ERROR, 那么所有级别低于ERROR的日志（如INFO和WARNING）都不会被记录, 从而提高了程序的运行效率.
    - 异步日志记录 (Asynchronous Logging)
        - glog 支持异步日志记录, 这意味着日志信息会被发送到一个单独的线程进行处理, 而不会阻塞当前线程的执行, 这对于性能要求较高的应用程序来说尤为重要.
    - 灵活的日志输出控制 (Flexible Log Output Control)
        - glog 允许开发者灵活地控制日志的输出方式和位置, 可以选择将日志输出到控制台,文件或者其他自定义的输出目的地.
- Performance Characteristics of glog
    - 低延迟: glog 的日志记录操作对程序的性能影响极小, 确保了即使在高负载情况下也能保持低延迟.
    - 高吞吐量: 得益于其异步日志记录的设计, glog能够处理大量的日志信息, 保证了高吞吐量.
    - 资源优化: glog 在资源使用上进行了优化, 确保了即使在资源受限的环境下也能正常工作.
- Output Control in glog
    - 日志级别控制: 开发者可以根据需要设置日志级别, 确保只有重要的信息被记录.
    - 日志分割: glog 支持自动分割日志文件, 帮助管理大量的日志信息.
    - 日志归档: glog 可以配置为自动归档旧的日志文件, 方便日后分析.
- Examples of Using glog
    - src/test_glog.cpp

### Underlying Principles of log4cplus

- [log4cplus on GitHub](https://github.com/log4cplus/log4cplus)
- [log4cplus reference on GitHub](https://log4cplus.github.io/log4cplus/)
- 丰富的日志级别, 日志格式和输出目标的配置选项, 使得开发者能够根据应用程序的需要灵活地记录信息.
    - 灵活性: log4cplus 提供多种日志级别和输出选项, 支持异步和同步日志记录.
    - 易用性: API 简单直观, 易于集成到现有项目中.
    - 可配置性: 可以通过配置文件或编程方式配置日志行为.
- 核心组件: 记录器（Logger）, 布局（Layout）和附加器（Appender）
    - 记录器负责生成日志消息, 布局负责格式化日志消息, 附加器负责将格式化后的消息输出到不同的目标.
    - 记录器是日志系统的入口点, 开发者通过记录器记录消息, 每个记录器都有一个名字和日志级别, 只有当消息的级别高于或等于记录器的级别时, 消息才会被记录.
    - 布局负责将日志消息转换为字符串, 并按照特定的格式输出, log4cplus 提供多种内置的布局, 如简单布局、模式布局等.
    - 附加器定义日志消息的输出目标, 常见的附加器有控制台附加器、文件附加器等, 可以根据需要配置一个或多个附加器.
- Performance Characteristics of log4cplus
    - log4cplus 的性能受到多种因素的影响, 包括日志级别、输出目标和日志消息的复杂性.
    - 日志级别越高, 记录的消息越少, 性能越好, 在生产环境中建议将日志级别设置较高以提高性能.
    - 将日志消息输出到控制台通常比输出到文件要慢, 对性能有较高要求, 建议将日志消息输出到内存或网络.
    - log4cplus 支持异步日志记录, 这可以显著提高性能, 特别是在高负载环境下.
- Output Control in glog
    - log4cplus 提供丰富的配置选项, 允许开发者灵活地控制日志输出.
    - 配置文件: 通过配置文件来设置日志级别、布局和附加器, 实现灵活的输出控制.
    - 程方式配置: log4cplus 允许开发者通过编程方式来配置日志系统.
- Examples of Using glog
    - src/test_log4cplus.cpp


### Underlying Principles of spdlog

- [spdlog on GitHub](https://github.com/gabime/spdlog)
- spdlog 是一款高效的 C++ 日志库, 它以其极高的性能和零成本的抽象而著称.
    - 零成本抽象: spdlog 通过模板和内联函数来实现零成本抽象, 确保只有在真正需要时才进行日志记录.
    - 异步日志记录: spdlog 支持异步日志记录, 将日志消息发送到一个单独的线程进行处理，从而减少对主线程性能的影响.
    - 高效的格式化: spdlog 使用 fmt 库进行高效的字符串格式化, 减少了格式化日志消息所需的时间.
- spdlog 支持异步和同步日志记录, 提供多种日志级别, 并允许用户将日志输出到控制台, 文件或自定义的接收器.
- Performance Characteristics of spdlog
    - 极高的日志记录速度: spdlog 能够在每秒记录数百万条日志消息,这对于需要处理大量日志数据的应用来说是非常重要的.
    - 低内存占用: spdlog 的设计确保了即使在高负载下, 也能保持低内存占用.
    - 灵活的配置: 用户可以根据需要配置 spdlog, 选择异步或同步日志记录, 以及选择不同的日志级别和输出目标.
- Output Control in spdlog    
    - 多种日志级别: spdlog 支持多种日志级别, trace、debug、info、warn、error 和 critical, 可以根据需要选择合适的日志级别.
    - 多种输出目标: 用户可以将日志输出到控制台、文件或通过网络发送到远程服务器.
    - 格式化输出: spdlog 支持格式化输出, 允许用户以结构化的方式输出日志消息.
- Examples of Using spdlog     
    - src/test_spdlog.cpp


### Performance Comparison Analysis
- glog 在简单字符串输出和多线程输出方面表现优秀, 但在复杂字符串拼接方面表现较差
- log4cplus 在文件输出方面表现优秀, 但在控制台输出和多线程输出方面表现较差
- spdlog 在设计上更加注重性能, 尤其是在高并发环境下


### 日志库的二次封装

- log4cplus secondary packaging
- logger.hpp and logger.cpp 



### 注意点
- glog 使用静态库和动态库, 需要配置预处理器
- static library for glog: -D GLOG_NO_EXPORT
- shared library for glog: -D GLOG_USE_GLOG_EXPORT
- log4cplus 编译生成库的时候需要注意字符集的选择, 默认是多字节，VS默认是Unicode
- log4cplus 使用的时候需要添加预处理器 -D UNICODE


### Configure the build

```shell
# Configure a debug build
cmake -S . -B build -D CMAKE_BUILD_TYPE=Debug
cmake --build build

# Configure a release build
cmake -S . -B build -D CMAKE_BUILD_TYPE=Release
cmake --build build
```

### Submodule of Git

- [Git Tools - Submodules](https://git-scm.com/book/en/v2/Git-Tools-Submodules)

```shell
# 首先需要是一个git项目
git init

# 远程git项目克隆到本地 external 文件夹
git submodule add https://github.com/google/glog.git external/glog/

```
