/**
 * @file thread_callback.cpp
 * @author Wei Li (weili_yzzcq@163.com)
 * @brief C++ 多线程和回调函数
 * @version 0.1
 * @date 2024-08-23
 * 
 * @copyright Copyright (c) 2024
 * 
 */

#include <chrono>
#include <functional>
#include <iostream>
#include <string>
#include <thread>

class Download
{
public:
    Download()  = default;
    ~Download() = default;

    // param-3: 是一个回调函数
    void download(const std::string &host, const std::string &path,
                  const std::function<void(const std::string &, const std::string &)> callback)
    {
        std::cout << "线程ID: " << std::this_thread::get_id() << std::endl;

        // download from http server......
        // do-something
        std::string response = host;
        callback(path, response);
    }

    void start_download(const std::string &host, const std::string &path,
                        const std::function<void(const std::string &, const std::string &)> callback)
    {
        // 将类的成员函数 download 通过 std::bind 进行绑定包装
        auto download_func
            = std::bind(&Download::download, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
        std::thread thread_task(download_func, host, path, callback);

        // 该线程会导致阻塞，直到任务运行结束, 所以直接 detach 线程
        thread_task.detach();
    }
};

// -----------------------------------
int main(int argc, const char **argv)
{
    auto downloading   = Download();
    auto callback_func = [](const std::string &path, const std::string &result) -> void
    {
        std::cout << "下载完成" << path << ": " << result.length() << " ---> " << result << std::endl;
    };

    downloading.start_download("download host", "download path1", std::move(callback_func));
    downloading.start_download("download host", "download path2", std::move(callback_func));
    downloading.start_download("download host", "download path3", std::move(callback_func));
    downloading.start_download("download host", "download path4", std::move(callback_func));
    downloading.start_download("download host", "download path5", std::move(callback_func));
    downloading.start_download("download host", "download path6", std::move(callback_func));

    // 休眠10s, 等待所有线程任务结束，否则主线程结束，该进程直接被销毁
    std::this_thread::sleep_for(std::chrono::microseconds(1000 * 10));

    return 0;
}
