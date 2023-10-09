/**
 * @file 06_csp_channel.cpp
 * @author Wei Li (weili_yzzcq@163.com)
 * @brief 
 * @version 0.1
 * @date 2023-10-09
 * 
 * @copyright Copyright (c) 2023
 * 
 */

#include <chrono>
#include <cstddef>
#include <iostream>
#include <mutex>
#include <queue>
#include <thread>

// the channel of "CSP" Concurrency Model
// 通过条件变量实现通信的阻塞和同步的
template<typename T>
class Channel
{
private:
    std::queue<T> queue_;
    std::mutex    mutex_;

    // * the channel is NULL, or capacity is full
    // * the consumer or producer Thread is suspended
    std::condition_variable cv_producer_;
    std::condition_variable cv_consumer_;

    // ? capacity is the inin capacity, not the real size of queue
    // ! capacity is the inin capacity, not the real size of queue
    // * capacity is the inin capacity, not the real size of queue
    size_t capacity_;
    bool   closed_ = false;

public:
    // * constructor with init capacity
    Channel(size_t capacity = 0)
        : capacity_(capacity)
    {
    }

    // ! the producer send data into the channel(thread-safe queue)
    // bool send(T &value);
    bool send(T value)
    {
        std::unique_lock<std::mutex> lock(mutex_);
        cv_producer_.wait(lock,
                          [this]()
                          {
                              // 对于无缓冲的 Channel(capacity_=0)，应该等待直到有消费者准备好
                              return (capacity_ == 0 && queue_.empty()) || queue_.size() < capacity_ || closed_;
                          });

        if (closed_)
        {
            return false;
        }

        queue_.push(value);

        // invoke the first block-thread in the waiting queue...
        cv_consumer_.notify_one();
        return true;
    }

    // ! the consumer receive data from the channel(thread-safe queue)
    bool receive(T &value)
    {
        std::unique_lock<std::mutex> lock(mutex_);
        cv_consumer_.wait(lock, [this]() { return !queue_.empty() || closed_; });

        if (closed_ && queue_.empty())
        {
            return false;
        }

        value = queue_.front();
        queue_.pop();

        // invoke the first block-thread in the waiting queue...
        cv_producer_.notify_one();
        return true;
    }

    void close()
    {
        std::unique_lock<std::mutex> lock(mutex_);
        closed_ = true;

        // Unblocks all threads currently waiting for *this.
        cv_producer_.notify_all();
        cv_consumer_.notify_all();
    }
};

/**
 * @brief test example
 * 
 * @param argc 
 * @param argv 
 * @return int 
 */
int main(int argc, char **argv)
{
    // const unsigned long long SIZE_DATA        = 10000;
    // const unsigned long long CAPACITY_CHANNEL = 200;
    const unsigned long long SIZE_DATA        = 6;
    const unsigned long long CAPACITY_CHANNEL = 10;

    // 10 缓冲的 Channel
    Channel<int> data_channel(CAPACITY_CHANNEL);
    // 0 缓冲的 Channel
    // Channel<int> data_channel;

    std::thread producer(
        [&]()
        {
            for (int idx = 0; idx < SIZE_DATA; ++idx)
            {
                data_channel.send(idx);
                std::cout << "Send data: " << idx << " successfully.\n";
            }

            data_channel.close();
        });

    std::thread consumer(
        [&]()
        {
            // using sleep to Simulate delaying consumers from starting consumption
            std::this_thread::sleep_for(std::chrono::milliseconds(500));

            int val;
            while (data_channel.receive(val))
            {
                std::cout << "Receive data: " << val << " successfully\n";
            }
        });

    producer.join();
    consumer.join();

    return 0;
}
