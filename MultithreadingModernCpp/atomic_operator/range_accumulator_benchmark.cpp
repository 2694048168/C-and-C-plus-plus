/**
 * @file range_accumulator_benchmark.cpp
 * @author Wei Li (weili_yzzcq@163.com)
 * @brief 
 * @version 0.1
 * @date 2023-05-31
 * 
 * @copyright Copyright (c) 2023
 * 
 */

#include "benchmark/benchmark.h"
#include "range_accumulator.hpp"

#include <atomic>
#include <iostream>
#include <mutex>
#include <random>
#include <thread>
#include <vector>


const unsigned long        g_size = 10000000;
std::mutex                 RangeAccumulator::_my_mutex;
unsigned long              RangeAccumulator::_mutex_sum(0);
std::atomic<unsigned long> RangeAccumulator::_atomic_sum(0);

//-----------------------------------------------------
static void BM_AtomicAccumulator(benchmark::State &state)
{
    for (auto _ : state)
    {
        state.PauseTiming();
        unsigned long number_of_threads = state.range(0);

        state.ResumeTiming();

        RangeAccumulator::Driver(number_of_threads, RangeAccumulator::AtomicAccumulator, g_size);
    }
}

//-----------------------------------------------------

static void BM_AtomicAccumulatorRelaxed(benchmark::State &state)
{
    for (auto _ : state)
    {
        state.PauseTiming();
        unsigned long number_of_threads = state.range(0);

        state.ResumeTiming();

        RangeAccumulator::Driver(number_of_threads, RangeAccumulator::AtomicAccumulatorRelaxed, g_size);
    }
}

//-----------------------------------------------------
static void BM_AtomicAccumulatorPartition(benchmark::State &state)
{
    for (auto _ : state)
    {
        state.PauseTiming();
        unsigned long number_of_threads = state.range(0);

        state.ResumeTiming();

        RangeAccumulator::Driver(number_of_threads, RangeAccumulator::AtomicAccumulatorPartition, g_size);
    }
}

//-----------------------------------------------------
static void BM_AtomicAccumulatorPartitionRelaxed(benchmark::State &state)
{
    for (auto _ : state)
    {
        state.PauseTiming();
        unsigned long number_of_threads = state.range(0);

        state.ResumeTiming();

        RangeAccumulator::Driver(number_of_threads, RangeAccumulator::AtomicAccumulatorPartitionRelaxed, g_size);
    }
}

//-----------------------------------------------------
static void BM_MutexAccumulatorPartition(benchmark::State &state)
{
    for (auto _ : state)
    {
        state.PauseTiming();
        unsigned long number_of_threads = state.range(0);

        state.ResumeTiming();

        RangeAccumulator::Driver(number_of_threads, RangeAccumulator::MutexAccumulatorPartition, g_size);
    }
}

//-----------------------------------------------------

BENCHMARK(BM_AtomicAccumulator)->DenseRange(1, 12, 1);
BENCHMARK(BM_AtomicAccumulatorRelaxed)->DenseRange(1, 12, 1);

BENCHMARK(BM_AtomicAccumulatorPartition)->DenseRange(1, 12, 1);
BENCHMARK(BM_AtomicAccumulatorPartitionRelaxed)->DenseRange(1, 12, 1);

BENCHMARK(BM_MutexAccumulatorPartition)->DenseRange(1, 12, 1);
BENCHMARK_MAIN();