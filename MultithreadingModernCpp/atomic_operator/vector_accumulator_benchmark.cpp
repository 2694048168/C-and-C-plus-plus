/**
 * @file vector_accumulator_benchmark.cpp
 * @author Wei Li (weili_yzzcq@163.com)
 * @brief 
 * @version 0.1
 * @date 2023-05-31
 * 
 * @copyright Copyright (c) 2023
 * 
 */


#include "benchmark/benchmark.h"
#include "vector_accumulator.hpp"

#include <atomic>
#include <iostream>
#include <mutex>
#include <random>
#include <thread>
#include <vector>


class Data
{
public:
    Data(unsigned long size)
    {
        v.resize(size);

        std::random_device                 rnd_device;
        // Specify the engine and distribution.
        std::mt19937                       mersenne_engine{rnd_device()}; // Generates random integers
        std::uniform_int_distribution<int> dist{1, 5};

        auto gen = [&dist, &mersenne_engine]()
        {
            return dist(mersenne_engine);
        };

        generate(std::begin(v), std::end(v), gen);
    }

    std::vector<unsigned long> v;
};

const unsigned long        g_size            = 1000000000;
const int                  g_max_num_threads = 15;
std::mutex                 VectorAccumulator::_my_mutex;
unsigned long              VectorAccumulator::_mutex_sum(0);
std::atomic<unsigned long> VectorAccumulator::_atomic_sum(0);

//-----------------------------------------------------

static void BM_AtomicAccumulator(benchmark::State &state)
{
    for (auto _ : state)
    {
        state.PauseTiming();
        unsigned long number_of_threads = state.range(0);
        Data          d(g_size);

        state.ResumeTiming();

        VectorAccumulator::Driver(d.v, number_of_threads, VectorAccumulator::AtomicAccumulator);
    }
}

//-----------------------------------------------------

static void BM_AtomicAccumulatorRelaxed(benchmark::State &state)
{
    for (auto _ : state)
    {
        state.PauseTiming();
        unsigned long number_of_threads = state.range(0);
        Data          d(g_size);

        state.ResumeTiming();

        VectorAccumulator::Driver(d.v, number_of_threads, VectorAccumulator::AtomicAccumulatorRelaxed);
    }
}

//-----------------------------------------------------

static void BM_AtomicAccumulatorPartitionRelaxed(benchmark::State &state)
{
    for (auto _ : state)
    {
        state.PauseTiming();
        unsigned long number_of_threads = state.range(0);
        Data          d(g_size);

        state.ResumeTiming();

        VectorAccumulator::Driver(d.v, number_of_threads, VectorAccumulator::AtomicAccumulatorPartitionRelaxed);
    }
}

//-----------------------------------------------------
static void BM_MutexAccumulatorPartition(benchmark::State &state)
{
    for (auto _ : state)
    {
        state.PauseTiming();
        unsigned long number_of_threads = state.range(0);
        Data          d(g_size);

        state.ResumeTiming();

        VectorAccumulator::Driver(d.v, number_of_threads, VectorAccumulator::MutexAccumulatorPartition);
    }
}

//-----------------------------------------------------

// BENCHMARK(BM_AtomicAccumulator)->DenseRange(1, 12, 1);
// BENCHMARK(BM_AtomicAccumulatorRelaxed)->DenseRange(1, 12, 1);
// BENCHMARK(BM_AtomicAccumulatorPartitionRelaxed)->DenseRange(1, 12, 1);

BENCHMARK(BM_MutexAccumulatorPartition)->DenseRange(1, 12, 1);
BENCHMARK_MAIN();
