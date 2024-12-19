/**
 * @file 16_threadTask.cpp
 * @author Wei Li (weili_yzzcq@163.com)
 * @brief 
 * @version 0.1
 * @date 2024-12-19
 * 
 * @copyright Copyright (c) 2024
 * 
 */

#include <condition_variable>
#include <functional>
#include <future>
#include <iostream>
#include <mutex>
#include <queue>
#include <thread>
#include <vector>

#ifdef _WIN32
#    include <windows.h>
#endif /* _WIN32 */

void thread_task()
{
    std::cout << "Create Thread via std::thread, do something\n";
}

int async_compute()
{
    return 42;
}

DWORD WINAPI ThreadFunc(LPVOID lpParam)
{
    std::cout << "Create Thread via Windows API thread\n";

    return 0;
}

class ThreadPool
{
public:
    ThreadPool(size_t threads)
    {
        for (size_t i = 0; i < threads; ++i)
        {
            workers.emplace_back(
                [this]
                {
                    while (true)
                    {
                        std::function<void()> task;
                        {
                            std::unique_lock<std::mutex> lock(this->queue_mutex);
                            this->condition.wait(lock, [this] { return this->stop || !this->tasks.empty(); });
                            if (this->stop && this->tasks.empty())
                                return;
                            task = std::move(this->tasks.front());
                            this->tasks.pop();
                        }
                        task();
                    }
                });
        }
    }

    template<class F, class... Args>
    auto enqueue(F &&f, Args &&...args) -> std::future<decltype(f(args...))>
    {
        auto task = std::make_shared<std::packaged_task<decltype(f(args...))()>>(
            std::bind(std::forward<F>(f), std::forward<Args>(args)...));

        std::future<decltype(f(args...))> res = task->get_future();
        {
            std::unique_lock<std::mutex> lock(queue_mutex);
            if (stop)
                throw std::runtime_error("enqueue on stopped ThreadPool");
            tasks.emplace([task]() { (*task)(); });
        }
        condition.notify_one();
        return res;
    }

    ~ThreadPool()
    {
        {
            std::unique_lock<std::mutex> lock(queue_mutex);
            stop = true;
        }
        condition.notify_all();
        for (std::thread &worker : workers) worker.join();
    }

private:
    std::vector<std::thread>          workers;
    std::queue<std::function<void()>> tasks;

    std::mutex              queue_mutex;
    std::condition_variable condition;
    bool                    stop = false;
};

// --------------------------------------------
int main(int /* argc */, char ** /* argv */)
{
    // std::thread
    std::thread task(thread_task);
    task.join();

    // std::async
    std::future<int> result = std::async(std::launch::async, async_compute);
    std::cout << "Result: " << result.get() << std::endl;

    // Windows API
    HANDLE hThread = CreateThread(NULL, 0, ThreadFunc, NULL, 0, NULL);
    WaitForSingleObject(hThread, INFINITE);
    CloseHandle(hThread);

    // thread-pool
    ThreadPool pool(4);

    auto result_ = pool.enqueue([](int answer) { return answer; }, 42);
    std::cout << "Result: " << result_.get() << std::endl;

    return 0;
}
