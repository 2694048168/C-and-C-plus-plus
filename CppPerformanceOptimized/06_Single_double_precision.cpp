/**
 * @file 06_Single_double_precision.cpp
 * @author Wei Li (weili_yzzcq@163.com)
 * @brief 
 * @version 0.1
 * @date 2024-12-10
 * 
 * @copyright Copyright (c) 2024
 * 
 */

#include "Stopwatch.hpp"

// -------------------------------------
int main(int argc, const char *argv[])
{
    constexpr unsigned NUM_REPEATED = 1000000000000;

    {
        Stopwatch{"====== Single precision ======"};

        for (size_t iter{0}; iter < NUM_REPEATED; ++iter)
        {
            float d, t, a = -9.8f, v0 = 0.0f, d0 = 100.0f;
            for (t = 0.0; t < 3.01f; t += 0.1f)
            {
                d = a * t * t + v0 * t + d0;
            }
        }
    }

    {
        Stopwatch{"====== Double precision ======"};

        for (size_t iter{0}; iter < NUM_REPEATED; ++iter)
        {
            double d, t, a = -9.8, v0 = 0.0, d0 = 100.0;
            for (t = 0.0; t < 3.01; t += 0.1)
            {
                d = a * t * t + v0 * t + d0;
            }
        }
    }

    return 0;
}
