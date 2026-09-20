/**
 * @file RingBuffer_FIFO.hpp
 * @author Wei Li (Ithaca) (weili_yzzca@163.com)
 * @brief 
 * @version 0.1
 * @date 2026-09-20
 * 
 * @copyright Copyright (c) 2026
 * 
 */

#pragma once

#include <array>
#include <atomic>
#include <cstddef>
#include <memory>

template<typename T, std::size_t Capacity, bool PlacementNew = true>
class RingBuffer
{
    static_assert(Capacity && !(Capacity & (Capacity - 1)), "Capacity must be 2^n");

public:
    template<typename U>
    bool push(U &&value)
    {
        const std::size_t w      = write_.load(std::memory_order_relaxed);
        const std::size_t next_w = (w + 1) & (Capacity - 1);

        if (next_w == read_.load(std::memory_order_acquire))
            return false;

        if constexpr (PlacementNew)
        {
            new (at(w)) T(std::forward<U>(value));
        }
        else
        {
            *at(w) = std::forward<U>(value);
        }

        write_.store(next_w, std::memory_order_release);
        return true;
    }

    bool pop(T &value)
    {
        const std::size_t r = read_.load(std::memory_order_relaxed);

        if (r == write_.load(std::memory_order_acquire))
            return false;

        if constexpr (PlacementNew)
        {
            T *ptr = at(r);
            value  = std::move(*ptr);
            ptr->~T();
        }
        else
        {
            value = std::move(*at(r));
        }

        read_.store((r + 1) & (Capacity - 1), std::memory_order_release);
        return true;
    }

    std::size_t Size() const
    {
        const std::size_t r = read_.load(std::memory_order_relaxed);
        const std::size_t w = write_.load(std::memory_order_relaxed);
        return (w >= r) ? (w - r) : (Capacity - r + w);
    }

private:
    alignas(64) std::atomic<std::size_t> read_;
    alignas(64) std::atomic<std::size_t> write_;

    struct ByteBuffer
    {
        alignas(alignof(T)) std::byte data[sizeof(T) * Capacity];
    };

    using Storage = std::conditional_t<PlacementNew, ByteBuffer, std::array<T, Capacity>>;

    T *at(std::size_t index)
    {
        if constexpr (PlacementNew)
        {
            return reinterpret_cast<T *>(storage_.data + index * sizeof(T));
        }
        else
        {
            return &storage_[index];
        }
    }

    alignas(64) Storage storage_;

public:
    RingBuffer()
        : read_(0)
        , write_(0)
    {
    }

    ~RingBuffer()
    {
        if constexpr (PlacementNew)
        {
            std::size_t r = read_.load(std::memory_order_relaxed);
            std::size_t w = write_.load(std::memory_order_relaxed);
            while (r != w)
            {
                at(r)->~T();
                r = (r + 1) & (Capacity - 1);
            }
        }
    }

    RingBuffer(const RingBuffer &)            = delete;
    RingBuffer &operator=(const RingBuffer &) = delete;
    RingBuffer(RingBuffer &&)                 = delete;
    RingBuffer &operator=(RingBuffer &&)      = delete;
};
