/**
 * @file 25_command_wrapper.hpp
 * @author Wei Li (Ithaca) (weili_yzzcq@163.com)
 * @brief 
 * @version 0.1
 * @date 2025-02-09
 * 
 * @copyright Copyright (c) 2025
 * 
 */

#include <iostream>
#include <map>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

class raw_environ_holder
{
public:
    raw_environ_holder()                           = delete;
    raw_environ_holder(const raw_environ_holder &) = delete;
    raw_environ_holder(raw_environ_holder &&);

    raw_environ_holder &operator=(const raw_environ_holder &) = delete;
    raw_environ_holder &operator=(raw_environ_holder &&);

    ~raw_environ_holder();

    operator char **()
    {
        return ppenv;
    };

private:
    friend class environ_map;

    explicit raw_environ_holder(char **ppenv)
        : ppenv(ppenv) {};

    void destroy();

    char **ppenv;
};

/**
 * @brief 环境变量通过 environ_map 类型传递,
 * 这是一个自定义的环境变量映射类，支持从当前进程环境变量初始化
 * 支持从当前进程环境变量初始化，并提供安全的内存管理机制
 * 
 */
class environ_map : public std::map<std::string, std::string>
{
public:
    environ_map() = default;
    environ_map(const std::map<std::string, std::string> &map)
        : std::map<std::string, std::string>(map) {};
    environ_map(const environ_map &) = default;

    raw_environ_holder raw() const;

    static environ_map get_for_current_process();
};

environ_map environ_map::get_for_current_process()
{
    environ_map result;

    int i = 0;
    while (environ[i])
    {
        std::string str(environ[i++]);
        size_t      indx = str.find('=');
        if (indx == std::string::npos)
            throw std::runtime_error("Failed to parse env");

        result[str.substr(0, indx)] = str.substr(indx + 1);
    }

    return result;
}

/**
 * @brief 为了将参数列表和环境变量转换为 execve 所需的格式,设计了以下辅助函数:参数转换
 */
std::shared_ptr<char *> to_argv(const std::string &cmd, const std::vector<std::string> &vec)
{
    char **argv = new char *[vec.size() + 2];
    argv[0]     = ::strdup(cmd.c_str());
    for (size_t i = 0; i < vec.size(); ++i) argv[i + 1] = ::strdup(vec[i].c_str());

    argv[vec.size() + 1] = nullptr;

    return std::shared_ptr<char *>(argv, argv_deleter);
}

/**
 * @brief 命令执行器类的设计
 * 命令执行器的核心是一个 command 类,它封装了命令名称、参数列表和环境变量.
 * 
 */
class command
{
public:
    // 构造函数构造函数接受命令名称、参数列表和环境变量
    command(const std::string cmd, const std::vector<std::string> &arguments, const environ_map &envs = environ_map());
    command(const command &) = default;
    command(command &&)      = default;

    command &operator=(const command &) = default;
    command &operator=(command &&)      = default;

    ~command() = default;

    // 执行逻辑exec() 方法是命令执行的核心;
    // 它使用 execve 系统调用执行命令，同时处理参数和环境变量的转换
    void exec();

private:
    std::string              m_cmd;
    std::vector<std::string> m_arguments;
    environ_map              m_envs;
};

// --------------------------------------------------
int main(int /* argc */, const char ** /* argv */)
{
    // 获取当前进程的环境变量
    environ_map m = environ_map::get_for_current_process();
    for (const auto &p : m)
    {
        std::cout << p.first << "=" << p.second << std::endl;
    }

    // 创建并执行命令
    command cmd("/bin/ls", std::vector<std::string>());
    cmd.exec();
}