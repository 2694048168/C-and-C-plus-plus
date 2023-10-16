/**
 * @file 08_thread_pool.cpp
 * @author Wei Li (weili_yzzcq@163.com)
 * @brief 
 * @version 0.1
 * @date 2023-10-16
 * 
 * @copyright Copyright (c) 2023
 * 
 */

#include <cstddef>
#include <functional>
#include <future>
#include <iostream>
#include <mutex>
#include <queue>
#include <thread>
#include <utility>
#include <vector>

// #define __cpp_lib_move_only_function
// clang++ 08_thread_pool.cpp -std=c++17
// clang++ 08_thread_pool.cpp -std=c++23

// thread pool via modern C++
class ThreadPool
{
public:
    ThreadPool(const size_t size)
        : busy_threads(size)
        , threads(std::vector<std::thread>(size))
        , shutdown_requested(false)
    {
        for (size_t idx = 0; idx < size; ++idx)
        {
            threads[idx] = std::thread(ThreadWorker(this));
        }
    }

    ThreadPool(const ThreadPool &) = delete;
    ThreadPool(ThreadPool &&)      = delete;

    ThreadPool &operator=(const ThreadPool &) = delete;
    ThreadPool &operator=(ThreadPool &&)      = delete;

    ~ThreadPool()
    {
        Shutdown();
    }

    // Waits until threads finish their current task and shutdowns the pool
    void Shutdown()
    {
        {
            std::lock_guard<std::mutex> lock(mutex);
            shutdown_requested = true;
            condition_variable.notify_all();
        }

        for (size_t idx = 0; idx < threads.size(); ++idx)
        {
            if (threads[idx].joinable())
            {
                threads[idx].join();
            }
        }
    }

    template<typename F, typename... Args>
    auto AddTask(F &&f, Args &&...args) -> std::future<decltype(f(args...))>
    {
#if __cpp_lib_move_only_function
        std::packaged_task<decltype(f(args...))()> task(std::bind(std::forward<F>(f), std::forward<Args>(args)...));

        auto future = task.get_future();

        auto wrapper_func = [task = std::move(task)]() mutable
        {
            std::move(task)();
        };
        {
            std::lock_guard<std::mutex> lock(mutex);
            queue.push(std::move(wrapper_func));
            // Wake up one thread if its waiting
            condition_variable.notify_one();
        }

        // Return future from promise
        return future;
#else
        // create function with bounded params ready to execute
        auto func     = std::bind(std::forward<F>(f), std::forward<Args>(args)...);
        auto task_ptr = std::make_shared<std::packaged_task<decltype(f(args...))()>>(func);

        // wrap the task pointer into a void lambda
        auto wrapper_func = [task_ptr]()
        {
            (*task_ptr)();
        };

        {
            std::lock_guard<std::mutex> lock(mutex);
            queue.push(func);
            // invoke one thread if its waiting
            condition_variable.notify_one();
        }

        // return future from promise
        return task_ptr->get_future();
#endif
    }

    size_t QueueSize()
    {
        std::unique_lock<std::mutex> lock(mutex);
        return queue.size();
    }

private:
    class ThreadWorker
    {
    public:
        ThreadWorker(ThreadPool *pool)
            : thread_pool(pool)
        {
        }

        void operator()()
        {
            std::unique_lock<std::mutex> lock(thread_pool->mutex);
            while (!thread_pool->shutdown_requested || (thread_pool->shutdown_requested && !thread_pool->queue.empty()))
            {
                thread_pool->busy_threads--;
                thread_pool->condition_variable.wait(
                    lock,
                    [this]() { return this->thread_pool->shutdown_requested || !this->thread_pool->queue.empty(); });
                thread_pool->busy_threads++;

                if (!this->thread_pool->queue.empty())
                {
#if __cpp_lib_move_only_function
                    auto func = std::move(thread_pool->queue.front());
#else
                    auto func = thread_pool->queue.front();
#endif
                    thread_pool->queue.pop();

                    lock.unlock();
                    func();
                    lock.lock();
                }
            }
        }

    private:
        ThreadPool *thread_pool;
    };

public:
    size_t busy_threads;

private:
    mutable std::mutex      mutex;
    std::condition_variable condition_variable;

    std::vector<std::thread> threads;

    bool shutdown_requested;

    // std::queue<std::function<void()>> queue;

// How C++23 made my Thread Pool twice as fast
#if __cpp_lib_move_only_function
    std::queue<std::move_only_function<void()>> queue;
#else
    std::queue<std::function<void()>> queue;
#endif
};

// Task
void EmptyTask() {}

void Checksum(const std::uint32_t num, std::atomic_uint64_t *checksum)
{
    *checksum += num;
}

// -----------------------------
int main(int argc, char **argv)
{
    std::cout << "======== the thread pool via Modern C++ ========\n";
    // TEST_CASE("Construction and deconstruction", "[thread_pool]")
    ThreadPool thread_pool{4};

    // TEST_CASE("N tasks", "[thread_pool]")
    std::queue<std::future<void>> results;

    // for (int n = 0; n < 1000; ++n)
    // {
    //     results.emplace(thread_pool.AddTask(EmptyTask));
    // }
    // while (results.size())
    // {
    //     results.front().get();
    //     results.pop();
    // }

    // TEST_CASE("N tasks checksum", "[thread_pool]")
    std::atomic_uint64_t checksum{0};
    std::uint64_t        localChecksum{0};

    for (std::uint32_t n = 0; n < 1000; ++n)
    {
        results.emplace(thread_pool.AddTask(Checksum, n, &checksum));
        localChecksum += n;
    }
    while (results.size())
    {
        results.front().get();
        results.pop();
    }

    return 0;
}
