/**
 * @file main.cpp
 * @author Wei Li (weili_yzzcq@163.com)
 * @brief 
 * @version 0.1
 * @date 2023-11-26
 * 
 * @copyright Copyright (c) 2023
 * 
 */

#include <filesystem>
#include <iostream>

/**
 * @brief Deleting files older than a given date
 * 
 * Write a function that, given the path to a directory and a duration,
 * deletes all the entries (files or subdirectories) older than the specified duration,
 * in a recursive manner. 
 *
 * The duration can represent anything, such as days, hours, minutes, seconds, 
 * and so on, or a combination of that, such as one hour and twenty minutes. 
 *
 * If the specified directory is itself older than the given duration, 
 * it should be deleted entirely.
 * ---------------------------------------------------------*/

/**
 * @brief Solution:
To perform filesystem operations, you should be using the filesystem library.
For working with time and duration, you should be using the chrono library.
 
A function that implements the requested functionality has to do the following:
1. Check whether the entry indicated by the target path exists and is older than
 the given duration, and if so, delete it
2. If it is not older and it's a directory, iterate through all its entries
 and call the function recursively
------------------------------------------------------ */
template<typename Duration>
bool is_older_than(const std::filesystem::path &path, const Duration duration)
{
    auto last_write     = std::filesystem::last_write_time(path);
    auto ftime_duration = last_write.time_since_epoch();

    auto now_duration = (std::chrono::system_clock::now() - duration).time_since_epoch();

    // return std::chrono::duration_cast<Duration>(now_duration - ftime_duration).count() > 0;
    return std::chrono::duration_cast<Duration>(now_duration - ftime_duration).count() < 0;
}

template<typename Duration>
void remove_files_older_than(const std::filesystem::path &path, const Duration duration)
{
    try
    {
        if (std::filesystem::exists(path))
        {
            // 递归处理
            if (std::filesystem::is_directory(path))
            {
                for (const auto &entry : std::filesystem::directory_iterator(path))
                {
                    remove_files_older_than(entry.path(), duration);
                }
            }
            else if (is_older_than(path, duration))
            {
                if (std::filesystem::remove(path))
                    std::cout << "remove the older file by given duration\n";
            }
        }
    }
    catch (const std::exception &ex)
    {
        std::cerr << ex.what() << std::endl;
    }
}

// ------------------------------
int main(int argc, char **argv)
{
    std::cout << __cplusplus << '\n';

    using namespace std::chrono_literals;

#ifdef _WIN32
    auto path = R"(..\Test\)";
#else
    auto path = R"(../Test/)";
#endif

    /**
     * @brief 字面量
    定义于内联命名空间 std::literals::chrono_literals
    1. operator""h (C++14) 表示小时的 std::chrono::duration 字面量
    2. operator""min (C++14) 表示分钟的 std::chrono::duration 字面量
    3. operator""s (C++14) 表示秒的 std::chrono::duration 字面量
    4. operator""ms (C++14) 表示毫秒的 std::chrono::duration 字面量
    5. operator""us (C++14) 表示微秒的 std::chrono::duration 字面量
    6. operator""ns (C++14) 表示纳秒的 std::chrono::duration 字面量
    -------------------------------------------------------------*/
    remove_files_older_than(path, 1h + 20min);
    // remove_files_older_than(path, 1min);
    // remove_files_older_than(path, 1s);

    return 0;
}