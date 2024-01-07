/**
 * @file main.cpp
 * @author Wei Li (weili_yzzcq@163.com)
 * @brief Circular buffer 循环缓存
 * @version 0.1
 * @date 2024-01-07
 * 
 * @copyright Copyright (c) 2024
 * 
 */

#include <cassert>
#include <cstddef>
#include <iostream>
#include <iterator>
#include <stdexcept>
#include <vector>

/**
 * @brief Circular buffer 循环缓存
 * 
 * Create a data structure that represents a circular buffer of a fixed size. 
 * A circular buffer overwrites existing elements when the buffer 
 * is being filled beyond its fixed size. 
 * 
 * The class you must write should:
 * 1. Prohibit default construction
 * 2. Support the creation of objects with a specified size
 * 3. Allow checking of the buffer capacity and status (empty(), full(), size(), capacity())
 * 4. Add a new element, an operation that could potentially overwrite the oldest element in the buffer
 * 5. Remove the oldest element from the buffer
 * 6. Support iteration through its elements
 * 
 * A circular buffer is a fixed-size container that behaves as if its two ends 
 * were connected to form a virtual circular memory layout. Its main benefit is 
 * that you don't need a large amount of memory to retain data, as older entries are 
 * overwritten by newer ones. Circular buffers are used in I/O buffering, 
 * bounded logging (when you only want to retain the last messages), 
 * buffers for asynchronous processing, and others.
 * 
 * We can differentiate between two situations:
 * 1. The number of elements added to the buffer has not reached its capacity
 *   (its user-1. defined fixed size). In this case, it behaves likes a regular container,
 *   such as a vector.
 * 2. The number of elements added to the buffer has reached and exceeded its capacity. 
 *   In this case, the buffer's memory is reused and older elements are being overwritten.
 *
 * We could represent such a structure using:
 * 1. A regular container with a pre-allocated number of elements
 * 2. A head pointer to indicate the position of the last inserted element
 * 3. A size counter to indicate the number of elements in the container, 
 *   which cannot exceed its capacity (since elements are being overwritten in this case)
 *
 * The two main operations with a circular buffer are:
 * 1. Adding a new element to the buffer. We always insert at the next position of 
 *    the head pointer (or index). This is the push() method shown below.
 * 2. Removing an existing element from the buffer. We always remove the oldest element. 
 *    That element is at position head - size (this must account for the circular nature of the index). This is the pop() method shown below.
 * 
 */

/**
 * @brief Solution:
------------------------------------------------------ */
template<typename T>
class CircularBuffer;

template<typename T>
class CircularBufferIterator
{
    typedef CircularBufferIterator          self_type;
    typedef T                               value_type;
    typedef T                              &reference;
    typedef const T                        &const_reference;
    typedef T                              *pointer;
    typedef std::random_access_iterator_tag iterator_category;
    typedef ptrdiff_t                       difference_type;

public:
    CircularBufferIterator(const CircularBuffer<T> &buf, const size_t pos, const bool last)
        : buffer_(buf)
        , index_(pos)
        , last_(last)
    {
    }

    self_type &operator++()
    {
        if (last_)
            throw std::out_of_range("Iterator cannot be incremented past the end of range.");

        index_ = (index_ + 1) % buffer_.data_.size();
        last_  = index_ == buffer_.next_pos();
        return *this;
    }

    self_type operator++(int)
    {
        self_type tmp = *this;
        ++*this;
        return tmp;
    }

    bool operator==(const self_type &other) const
    {
        assert(compatible(other));
        return index_ == other.index_ && last_ == other.last_;
    }

    bool operator!=(const self_type &other) const
    {
        return !(*this == other);
    }

    const_reference operator*() const
    {
        return buffer_.data_[index_];
    }

    const_reference operator->() const
    {
        return buffer_.data_[index_];
    }

private:
    bool compatible(const self_type &other) const
    {
        return &buffer_ == &other.buffer_;
    }

    const CircularBuffer<T> &buffer_;
    size_t                   index_;
    bool                     last_;
};

template<typename T>
class CircularBuffer
{
    typedef CircularBufferIterator<T> const_iterator;

    CircularBuffer() = delete;

public:
    explicit CircularBuffer(const size_t size)
        : data_(size)
    {
    }

    bool clear() noexcept
    {
        head_ = -1;
        size_ = 0;
    }

    bool empty() const noexcept
    {
        return size_ == 0;
    }

    bool full() const noexcept
    {
        return size_ == data_.size();
    }

    size_t capacity() const noexcept
    {
        return data_.size();
    }

    size_t size() const noexcept
    {
        return size_;
    }

    void push(const T item)
    {
        head_        = next_pos();
        data_[head_] = item;

        if (size_ < data_.size())
            ++size_;
    }

    T pop()
    {
        if (empty())
            throw std::runtime_error("empty buffer");

        auto pos = first_pos();
        --size_;
        return data_[pos];
    }

    const_iterator begin() const
    {
        return const_iterator(*this, first_pos(), empty());
    }

    const_iterator end() const
    {
        return const_iterator(*this, next_pos(), true);
    }

private:
    std::vector<T> data_;
    size_t         head_ = -1;
    size_t         size_ = 0;

    size_t next_pos() const noexcept
    {
        return size_ == 0 ? 0 : (head_ + 1) % data_.size();
    }

    size_t first_pos() const noexcept
    {
        return size_ == 0 ? 0 : (head_ + data_.size() - size_ + 1) % data_.size();
    }

    friend class CircularBufferIterator<T>;
};

template<typename T>
void printCircularBuffer(CircularBuffer<T> &buf)
{
    for (auto &e : buf)
    {
        std::cout << e << ' ';
    }

    std::cout << std::endl;
}

// ------------------------------
int main(int argc, char **argv)
{
    CircularBuffer<int> circular_buf(5);

    assert(circular_buf.empty());
    assert(!circular_buf.full());
    assert(circular_buf.size() == 0);
    printCircularBuffer(circular_buf);

    circular_buf.push(1);
    circular_buf.push(2);
    circular_buf.push(3);
    assert(!circular_buf.empty());
    assert(!circular_buf.full());
    assert(circular_buf.size() == 3);
    printCircularBuffer(circular_buf);

    auto item = circular_buf.pop();
    assert(item == 1);
    assert(!circular_buf.empty());
    assert(!circular_buf.full());
    assert(circular_buf.size() == 2);

    circular_buf.push(4);
    circular_buf.push(5);
    circular_buf.push(6);
    assert(!circular_buf.empty());
    assert(circular_buf.full());
    assert(circular_buf.size() == 5);
    printCircularBuffer(circular_buf);

    circular_buf.push(7);
    circular_buf.push(8);
    assert(!circular_buf.empty());
    assert(circular_buf.full());
    assert(circular_buf.size() == 5);
    printCircularBuffer(circular_buf);

    item = circular_buf.pop();
    assert(item == 4);
    item = circular_buf.pop();
    assert(item == 5);
    item = circular_buf.pop();
    assert(item == 6);

    assert(!circular_buf.empty());
    assert(!circular_buf.full());
    assert(circular_buf.size() == 2);
    printCircularBuffer(circular_buf);

    item = circular_buf.pop();
    assert(item == 7);
    item = circular_buf.pop();
    assert(item == 8);

    assert(circular_buf.empty());
    assert(!circular_buf.full());
    assert(circular_buf.size() == 0);
    printCircularBuffer(circular_buf);

    circular_buf.push(9);
    assert(!circular_buf.empty());
    assert(!circular_buf.full());
    assert(circular_buf.size() == 1);
    printCircularBuffer(circular_buf);

    return 0;
}
