/**
 * @file main.cpp
 * @author Wei Li (weili_yzzcq@163.com)
 * @brief Thread-safe logging to the console
 * @version 0.1
 * @date 2024-01-08
 * 
 * @copyright Copyright (c) 2024
 * 
 * You can using file-stream ---> std::cout stream; 
 * and log message into file(.txt)
 * 
 */

#include <iostream>
#include <mutex>
#include <random>
#include <string>

/**
 * @brief Thread-safe logging to the console
 * 
 * Write a class that enables components running in different threads to 
 * safely print log messages to the console by synchronizing access to 
 * the standard output stream to guarantee the integrity of the output. 
 * This logging component should have a method called log() with a string argument 
 * representing the message to be printed to the console.
 * 
 * Although C++ does not have the concept of a console and uses streams to perform
 *  input and output operations on sequential media such as files, 
 *  the std::cout and std::wcout global objects control the output to a stream 
 *  buffer associated with the C output stream stdout. 
 * These global stream objects cannot be safely accessed from different threads.
 * Should you need that, you must synchronize the access to them. 
 * That is exactly the purpose of the requested component for this problem.
 *
 * The logger class, shown as follows, uses an std::mutex to synchronize access to
 *  the std::cout object in the log() method. The class is implemented as 
 * a thread-safe singleton. The static method instance() returns a reference to
 *  a local static object (that has storage duration). 
 * In C++11, initialization of a static object happens only once, 
 * even if several threads attempt to initialize the same static object at the same time.
 *  In such a case, concurrent threads are blocked until the initialization executed on 
 * the first calling thread completes. 
 * Therefore, there is no need for additional user-defined synchronization mechanisms:
 * 
 */

/**
 * @brief Solution:
------------------------------------------------------ */
class Logger
{
protected:
    Logger() {}

public:
    static Logger &instance()
    {
        static Logger lg;
        return lg;
    }

    Logger(const Logger &)            = delete;
    Logger &operator=(const Logger &) = delete;

    void log(std::string_view message)
    {
        std::lock_guard<std::mutex> lock(mutex_);
        std::cout << "LOG: " << message << std::endl;
    }

private:
    std::mutex mutex_;
};

// ------------------------------
int main(int argc, char **argv)
{
    std::vector<std::thread> modules;

    for (int id = 1; id <= 5; ++id)
    {
        modules.emplace_back(
            [id]()
            {
                std::random_device              rd;
                std::mt19937                    mt(rd());
                std::uniform_int_distribution<> ud(100, 1000);

                Logger::instance().log("module " + std::to_string(id) + " started");

                std::this_thread::sleep_for(std::chrono::milliseconds(ud(mt)));

                Logger::instance().log("module " + std::to_string(id) + " finished");
            });
    }

    for (auto &m : modules)
    {
        m.join();
    }

    return 0;
}
