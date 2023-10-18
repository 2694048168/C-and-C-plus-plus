/**
 * @file MCTPL.hpp
 * @author Wei Li (weili_yzzcq@163.com)
 * @brief 
 * @version 0.1
 * @date 2023-10-18
 * 
 * @copyright Copyright (c) 2023
 * 
 */

#ifndef __MODERN_CPP_THREAD_POOL_HPP__
#define __MODERN_CPP_THREAD_POOL_HPP__

#include <atomic>
#include <cstddef>
#include <functional>
#include <future>
#include <memory>
#include <mutex>
#include <queue>
#include <thread>
#include <vector>

/* thread pool to run user's functors with signature
    ret func(int id, other_params)
where id is the index of the thread that runs the functor
ret is some return type 
----------------------------------------------------------*/

namespace MCTPL {

// the task queue or task channel
template<typename T>
class ChannelQueue
{
public:
    bool push(const T &value)
    {
        std::unique_lock<std::mutex> lock(this->mutex);
        this->channel_queue.push(value);
        return true;
    }

    bool pop(T &value)
    {
        std::unique_lock<std::mutex> lock(this->mutex);
        if (this->channel_queue.empty())
        {
            return false;
        }

        value = this->channel_queue.front();
        this->channel_queue.pop();

        return true;
    }

private:
    std::queue<T> channel_queue;
    std::mutex    mutex;
};

class ThreadPool
{
public:
    ThreadPool()
    {
        this->init();
    }

    ThreadPool(size_t num_thread)
    {
        this->init();
        this->resize(num_thread);
    }

    // the destructor waits for all the functions in the queue to be finished
    ~ThreadPool()
    {
        this->stop(true);
    }

    // get the number of running threads in the pool
    size_t size()
    {
        // two concept: size and capacity
        // return static_cast<int>(this->threads.size());
        return this->threads.size();
    }

    // number of idle(not busy, free wait...) threads
    int num_idle()
    {
        return this->nWaiting;
    }

    std::thread &get_thread(int i)
    {
        return *this->threads[i];
    }

    // change the number of threads in the thread-pool
    // should be called from one thread,
    // otherwise be careful to not interleave, also with this->stop()
    // num_thread must be >= 0
    void resize(size_t num_thread)
    {
        if (!this->isStop && !this->isDone)
        {
            auto old_num_thread = this->size();

            // if the number of threads is increased
            if (old_num_thread <= num_thread)
            {
                this->threads.resize(num_thread);
                this->flags.resize(num_thread);

                for (size_t i = old_num_thread; i < num_thread; ++i)
                {
                    this->flags[i] = std::make_shared<std::atomic<bool>>(false);
                    this->set_thread(i);
                }
            }
            // if the number of threads is decreased
            else
            {
                for (size_t i = old_num_thread - 1; i >= num_thread; --i)
                {
                    // this thread will finish
                    *this->flags[i] = true;
                    this->threads[i]->detach();
                }

                {
                    // stop the detached threads that were waiting
                    std::unique_lock<std::mutex> lock(this->mutex);
                    this->condition_variable.notify_all();
                }

                // safe to delete because the threads are detached
                this->threads.resize(num_thread);
                // safe to delete because the threads have copies of shared_ptr of the flags, not originals
                this->flags.resize(num_thread);
            }
        }
    }

    // empty the queue or channel
    void clear_queue()
    {
        std::function<void(int id)> *_func;
        while (this->channel.pop(_func))
        {
            delete _func; /* empty the queue */
        }
    }

    // pops a functional wrapper to the original function
    std::function<void(int)> pop()
    {
        std::function<void(int id)> *_f = nullptr;
        this->channel.pop(_f);

        // at return, delete the function even if an exception occurred
        std::unique_ptr<std::function<void(int id)>> func(_f);

        std::function<void(int)> f;
        if (_f)
            f = *_f;

        return f;
    }

    // wait for all computing threads to finish and stop all threads
    // may be called asynchronously to not pause the calling thread while waiting
    // if isWait == true, all the functions in the queue are run,
    // otherwise the queue is cleared without running the functions
    void stop(bool isWait = false)
    {
        if (!isWait)
        {
            if (this->isStop)
                return;

            this->isStop = true;
            for (size_t i = 0, n = this->size(); i < n; ++i)
            {
                *this->flags[i] = true; /* command the threads to stop */
            }
            this->clear_queue(); /* empty the queue */
        }
        else
        {
            if (this->isDone || this->isStop)
                return;

            // give the waiting threads a command to finish
            this->isDone = true;
        }

        {
            std::unique_lock<std::mutex> lock(this->mutex);
            this->condition_variable.notify_all(); /* stop all waiting threads */
        }

        // wait for the computing threads to finish
        for (size_t i = 0; i < this->threads.size(); ++i)
        {
            if (this->threads[i]->joinable())
                this->threads[i]->join();
        }

        // if there were no threads in the pool but some functors in the queue,
        // the functors are not deleted by the threads
        // therefore delete them here
        this->clear_queue();
        this->threads.clear();
        this->flags.clear();
    }

    template<typename F, typename... Rest>
    auto push(F &&f, Rest &&...rest) -> std::future<decltype(f(0, rest...))>
    {
        auto pck = std::make_shared<std::packaged_task<decltype(f(0, rest...))(int)>>(
            std::bind(std::forward<F>(f), std::placeholders::_1, std::forward<Rest>(rest)...));

        auto _func = new std::function<void(int id)>([pck](int id) { (*pck)(id); });

        this->channel.push(_func);

        std::unique_lock<std::mutex> lock(this->mutex);
        this->condition_variable.notify_one();
        return pck->get_future();
    }

    // run the user's function that excepts argument int - id of the running thread.
    //  returned value is templatized
    // operator returns std::future,
    // where the user can get the result and rethrow the catched exceptins
    template<typename F>
    auto push(F &&f) -> std::future<decltype(f(0))>
    {
        auto pck = std::make_shared<std::packaged_task<decltype(f(0))(int)>>(std::forward<F>(f));

        auto _func = new std::function<void(int id)>([pck](int id) { (*pck)(id); });

        this->channel.push(_func);

        std::unique_lock<std::mutex> lock(this->mutex);
        this->condition_variable.notify_one();
        return pck->get_future();
    }

private:
    // deleted the default and behavior
    ThreadPool(const ThreadPool &);            // = delete;
    ThreadPool(ThreadPool &&);                 // = delete;
    ThreadPool &operator=(const ThreadPool &); // = delete;
    ThreadPool &operator=(ThreadPool &&);      // = delete;

    void init()
    {
        this->nWaiting = 0;
        this->isStop   = false;
        this->isDone   = false;
    }

    void set_thread(int i)
    {
        // a copy of the shared ptr to the flag
        std::shared_ptr<std::atomic_bool> flag(this->flags[i]);

        /* a copy of the shared ptr to the flag */
        auto func = [this, i, flag]()
        {
            std::atomic<bool> &_flag = *flag;

            std::function<void(int id)> *_func;

            bool isPop = this->channel.pop(_func);
            while (true)
            {
                while (isPop)
                { // if there is anything in the queue
                    // at return, delete the function even if an exception occurred
                    std::unique_ptr<std::function<void(int id)>> func(_func);

                    (*_func)(i);
                    if (_flag)
                        return; /* the thread is wanted to stop, return even if the queue is not empty yet */
                    else
                        isPop = this->channel.pop(_func);
                }

                // the queue is empty here, wait for the next command
                std::unique_lock<std::mutex> lock(this->mutex);
                ++this->nWaiting;
                this->condition_variable.wait(lock,
                                              [this, &_func, &isPop, &_flag]()
                                              {
                                                  isPop = this->channel.pop(_func);
                                                  return isPop || this->isDone || _flag;
                                              });
                --this->nWaiting;
                if (!isPop)
                    return; /* if the queue is empty and this->isDone == true or *flag then return */
            }
        };

        // compiler may not support std::make_unique()
        this->threads[i].reset(new std::thread(func));
    }

private:
    // std::vector<std::thread> threads;
    std::vector<std::unique_ptr<std::thread>>      threads;
    // std::vector<std::shared_ptr<std::atomic<bool>>> flags;
    std::vector<std::shared_ptr<std::atomic_bool>> flags;

    // MCTPL::ChannelQueue<std::function<void(int id)> *> channel;
    ChannelQueue<std::function<void(int id)> *> channel;

    // std::atomic<bool> isDone;
    std::atomic_bool isDone;
    // std::atomic<bool> isStop;
    std::atomic_bool isStop;
    // std::atomic<int>  nWaiting;
    std::atomic_int  nWaiting; /* how many threads are waiting */

    std::mutex              mutex;
    std::condition_variable condition_variable;
};

} // namespace MCTPL

#endif /* __MODERN_CPP_THREAD_POOL_HPP__ */