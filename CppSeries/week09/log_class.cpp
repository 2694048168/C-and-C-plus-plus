/**
 * @file log_class.cpp
 * @author Wei Li (weili_yzzcq@163.com)
 * @date 2023-05-13
 * @version 0.1.1
 *
 * @copyright Copyright (c) 2023
 *
 * @brief a simple example for logger class
 * @attention
 *
 */

#include <iostream>

class MyLog
{
public:
    enum Level
    {
        LevelFatal = 0,
        LevelError,
        LevelWarning,
        LevelInfo,
        LevelDebug,
        NUM_LEVEL
    };

private:
    Level m_LogLevel = LevelInfo;

public:
    void set_level(Level level)
    {
        m_LogLevel = level;
    }
    Level get_level() const
    {
        std::cout << "the total level of log: " << NUM_LEVEL << std::endl;
        return m_LogLevel;
    }

    void Fatal(const char* message)
    {
        if (m_LogLevel >= LevelFatal)
        {
            std::cout << "[Fatal]: " << message << std::endl;
        }
    }
    void Error(const char* message)
    {
        if (m_LogLevel >= LevelError)
        {
            std::cout << "[Error]: " << message << std::endl;
        }
    }
    void Warn(const char* message)
    {
        if (m_LogLevel >= LevelWarning)
        {
            std::cout << "[Warning]: " << message << std::endl;
        }
    }
    void Info(const char* message)
    {
        if (m_LogLevel >= LevelInfo)
        {
            std::cout << "[Info]: " << message << std::endl;
        }
    }
    void Debug(const char* message)
    {
        if (m_LogLevel >= LevelDebug)
        {
            std::cout << "[Debug]: " << message << std::endl;
        }
    }
};


/**
 * @brief main function and the entry point of program.
 */
int main(int argc, char const *argv[])
{
    MyLog log;
    // log.set_level(MyLog::Level::LevelDebug);
    log.set_level(MyLog::Level::LevelInfo);
    // log.set_level(MyLog::Level::LevelWarning);
    // log.set_level(MyLog::Level::LevelError);
    log.get_level();

    log.Fatal("message for error!");
    log.Error("message for error!");
    log.Warn("message for warning!");
    log.Info("message for info!");
    log.Debug("message for info!");

    return 0;
}

/** Build(compile and link) commands via command-line.
 *
 * $ clang++ log_class.cpp
 * $ clang++ log_class.cpp -std=c++17
 * $ ./a.exe # on Windows
 * $ ./a.out # on Linux or Mac
 *
 */