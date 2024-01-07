/**
 * @file main.cpp
 * @author Wei Li (weili_yzzcq@163.com)
 * @brief Double buffer
 * @version 0.1
 * @date 2024-01-07
 * 
 * @copyright Copyright (c) 2024
 * 
 */

#include <cstddef>
#include <iostream>
#include <mutex>
#include <thread>
#include <vector>

/**
 * @brief Double buffer
 * 
 * Write a class that represents a buffer that could be written and read 
 * at the same time without the two operations colliding. A read operation must 
 * provide access to the old data while a write operation is in progress. 
 * Newly written data must be available for reading upon completion of the write operation.
 * 
 * The problem described here is a typical double buffering situation. 
 * Double buffering is the most common case of multiple buffering, 
 * which is a technique that allows a reader to see a complete version of the data 
 * and not a partially updated version produced by a writer. This is a common technique 
 * – especially in computer graphics – for avoiding flickering.
 * 
 * In order to implement the requested functionality, the buffer class that we should 
 * write must have two internal buffers: one that contains temporary data being written, 
 * and another one that contains completed (or committed) data. 
 * Upon the completion of a write operation, the content of the temporary buffer is written
 * in the primary buffer. For the internal buffers, the implementation below uses 
 * std::vector. When the write operation completes, instead of copying data from one buffer
 * to the other, we just swap the content of the two, which is a much faster operation. 
 * Access to the completed data is provided with either the read() function, 
 * which copies the content of the read buffer into a designated output, 
 * or with direct element access (overloaded operator[]). Access to the read buffer 
 * is synchronized with an std::mutex to make it safe to read from one thread 
 * while another is writing to the buffer:
 * 
 */

/**
 * @brief Solution:
------------------------------------------------------ */
template<typename T>
class DoubleBuffer
{
    typedef T        value_type;
    typedef T       &reference;
    typedef const T &const_reference;
    typedef T       *pointer;

public:
    explicit DoubleBuffer(const size_t size)
        : readBuf(size)
        , writeBuf(size)
    {
    }

    size_t size() const noexcept
    {
        return readBuf.size();
    }

    void write(const T *const ptr, const size_t size)
    {
        std::unique_lock<std::mutex> lock(mutex_);

        auto length = std::min(size, writeBuf.size());

        std::copy(ptr, ptr + length, std::begin(writeBuf));
        writeBuf.swap(readBuf);
    }

    template<class Output>
    void read(Output it) const
    {
        std::unique_lock<std::mutex> lock(mutex_);
        std::copy(std::cbegin(readBuf), std::cend(readBuf), it);
    }

    pointer data() const
    {
        std::unique_lock<std::mutex> lock(mutex_);
        return readBuf.data();
    }

    reference operator[](const size_t pos)
    {
        std::unique_lock<std::mutex> lock(mutex_);
        return readBuf[pos];
    }

    const_reference operator[](const size_t pos) const
    {
        std::unique_lock<std::mutex> lock(mutex_);
        return readBuf[pos];
    }

    void swap(DoubleBuffer other)
    {
        std::swap(readBuf, other.readBuf);
        std::swap(writeBuf, other.writeBuf);
    }

private:
    std::vector<T>     readBuf;
    std::vector<T>     writeBuf;
    mutable std::mutex mutex_;
};

template<typename T>
void printDoubleBuffer(const DoubleBuffer<T> &buf)
{
    buf.read(std::ostream_iterator<T>(std::cout, " "));
    std::cout << std::endl;
}

// ------------------------------
int main(int argc, char **argv)
{
    DoubleBuffer<int> buf(10);

    std::thread t(
        [&buf]()
        {
            for (int i = 1; i < 1000; i += 10)
            {
                int data[] = {i, i + 1, i + 2, i + 3, i + 4, i + 5, i + 6, i + 7, i + 8, i + 9};
                buf.write(data, 10);

                using namespace std::chrono_literals;
                std::this_thread::sleep_for(100ms);
            }
        });

    auto start = std::chrono::system_clock::now();
    do
    {
        printDoubleBuffer(buf);

        using namespace std::chrono_literals;
        std::this_thread::sleep_for(150ms);
    }
    while (std::chrono::duration_cast<std::chrono::seconds>(std::chrono::system_clock::now() - start).count() < 12);

    t.join();

    return 0;
}
