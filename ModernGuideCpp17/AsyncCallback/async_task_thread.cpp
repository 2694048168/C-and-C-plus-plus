/**
 * @file async_task_thread.cpp
 * @author Wei Li (Ithaca) (weili_yzzcq@163.com)
 * @brief Modern C++ Async thread task launch
 * @version 0.1
 * @date 2025-07-21
 * 
 * @copyright Copyright (c) 2025
 * 
 * g++ async_task_thread.cpp -std=c++20
 * clang++ async_task_thread.cpp -std=c++20
 * 
 */

// #include <Windows.h>
// #include <immintrin.h>

#include <chrono>
#include <future>
#include <iostream>
#include <thread>
#include <vector>

void testAsyncTask()
{
    std::vector<int> regionVec{1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
    // std::vector<int> regionVec{1, 2, 3, 4};
    std::vector<int> regionRet{};

    auto slotDetectionPipeline = [&](int idx)
    {
        std::cout << regionVec[idx] << " ";
        std::this_thread::sleep_for(std::chrono::milliseconds(123));
        return regionVec[idx];
    };

    auto showResult = [&]()
    {
        for (auto &&value : regionRet)
        {
            std::cout << value << " ";
        }
        std::cout << "\n\n";
    };

    // single thread to processing
    {
        auto startTime = std::chrono::high_resolution_clock::now();
        for (size_t i = 0; i < regionVec.size(); i++)
        {
            auto ret = slotDetectionPipeline(i) + 1;
            regionRet.emplace_back(ret);
        }
        auto endTime  = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime);
        std::cout << "\nsingle thread to processing Time: " << duration.count() << " ms\n";

        showResult();
    }

    // 获取系统支持的并发线程数
    unsigned int n_threads = std::thread::hardware_concurrency();
    std::cout << "Available concurrent threads: " << n_threads << "\n\n";

    // multi-thread async to processing
    {
        auto                          startTime = std::chrono::high_resolution_clock::now();
        std::vector<std::future<int>> futureVec; // 使用异步线程处理所有对象
        for (size_t i = 0; i < regionVec.size(); i++)
        {
            // 启动异步任务（确保使用引用传递对象）
            futureVec.emplace_back(std::async(std::launch::async, slotDetectionPipeline, i));
        }

        regionRet.clear();
        for (size_t i = 0; i < regionVec.size(); i++)
        {
            regionRet.emplace_back(futureVec[i].get()); // 等待所有异步任务完成(get 阻塞[同步]直到当前任务完成)
        }
        auto endTime  = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime);
        std::cout << "\nmulti-thread async to processing Time: " << duration.count() << " ms\n";

        showResult();
    }
}

// bool isSupportedAVX512()
// {
//     int cpuInfo[4]{};
//     __cpuid(cpuInfo, 1);
//     // 检查OSXSAVE和AVX
//     if ((cpuInfo[2] & (1 << 27)) && (cpuInfo[2] & (1 << 28)))
//     {
//         unsigned long long xcr0 = _xgetbv(0);
//         if ((xcr0 & 0xE6) != 0xE6)
//             return false;

//         __cpuidex(cpuInfo, 7, 0);
//         return (cpuInfo[1] & (1 << 16)) != 0;

//         // 检查AVX512F
//     }
//     return false;
// }

// // memcpy by byte via AVX512 vector acc.
// void *AVX512_memcpy(void *dest, void *src, size_t size)
// {
//     if (0 == size || dest == src || nullptr == dest || nullptr == src)
//         return dest;

//     auto *dest_ptr = static_cast<char *>(dest);
//     auto *src_ptr  = static_cast<const char *>(src);

//     // small block memory, such as 128 bytes
//     if (size < 128)
//     {
//         for (size_t i = 0; i < size; ++i)
//         {
//             dest_ptr[i] = src_ptr[i];
//         }
//         return dest;
//     }
//     // align to 64bytes: 处理目标地址的非对齐情况 && 确保主循环64字节对齐
//     constexpr size_t align_bytes = 64;
//     const size_t     alignOffset = (reinterpret_cast<uintptr_t>(dest_ptr) & (align_bytes - 1));
//     if (0 != alignOffset)
//     {
//         const size_t prefix = align_bytes - alignOffset;
//         // 对齐处理
//         for (size_t i = 0; i < prefix; ++i)
//         {
//             dest_ptr[i] = src_ptr[i];
//         }
//         dest_ptr += prefix;
//         src_ptr += prefix;
//         size -= prefix;
//     }

//     // AVX512 block for-range
//     const size_t chunks_num      = size / align_bytes;
//     const size_t remainder_bytes = size % align_bytes;
//     for (size_t idx{0}; idx < chunks_num; ++idx)
//     {
//         // 预取下一缓存行 cache-line: 提前2个缓存行预取数据 && 减少内存延迟影响
//         if (idx + 2 < chunks_num)
//             _mm_prefetch(src_ptr + align_bytes * (idx + 2), _MM_HINT_T0);

//         // 加载->存储
//         __m512i data = _mm512_loadu_si512(reinterpret_cast<const __m512i *>(src_ptr) + idx);
//         //_mm512_store_si512(reinterpret_cast<__m512i*>(dest_ptr) + idx, data);
//         // 非时态存储（可选优化）:适合大块不重复使用的内存 && 避免污染CPU缓存
//         _mm512_stream_si512(reinterpret_cast<__m512i *>(dest_ptr) + idx, data);
//     }
//     // 确保所有非时间存储完成
//     _mm_sfence();

//     // process the remainder data 处理尾部剩余数据 || 剩余部分：使用32/16字节SSE/AVX指令
//     if (remainder_bytes > 0)
//     {
//         const size_t offset = chunks_num * align_bytes;
//         for (size_t idx{0}; idx < remainder_bytes; ++idx)
//         {
//             dest_ptr[offset + idx] = src_ptr[offset + idx];
//         }
//     }
//     return dest_ptr;
// }

// // 性能测试函数
// void test_performance()
// {
//     constexpr size_t  size = 200 * 1024 * 1024; // 200 MB
//     std::vector<char> src(size, 'A');
//     std::vector<char> dest(size);

//     // 预热缓存
//     std::copy(src.begin(), src.end(), dest.begin());

//     // 测试AVX512
//     auto start = std::chrono::high_resolution_clock::now();
//     AVX512_memcpy(dest.data(), src.data(), size);
//     auto                          end     = std::chrono::high_resolution_clock::now();
//     std::chrono::duration<double> elapsed = end - start;
//     std::cout << "AVX512 memcpy: " << elapsed.count() * 1000 << " ms\n";

//     // 测试标准memcpy
//     start = std::chrono::high_resolution_clock::now();
//     memcpy(dest.data(), src.data(), size);
//     end     = std::chrono::high_resolution_clock::now();
//     elapsed = end - start;
//     std::cout << "Standard memcpy: " << elapsed.count() * 1000 << " ms\n";

//     // 验证结果
//     if (memcmp(src.data(), dest.data(), size) != 0)
//     {
//         std::cerr << "Validation failed!" << std::endl;
//     }
//     else
//     {
//         std::cout << "Validation passed!" << std::endl;
//     }
// }

//--------------------------------------
int main(int argc, char **argv)
{
    testAsyncTask();

    // std::cout << "The host is supported AVX512: " << std::boolalpha << isSupportedAVX512() << '\n';
    // if (isSupportedAVX512())
    // test_performance();

    system("pause");
    return 0;
}
