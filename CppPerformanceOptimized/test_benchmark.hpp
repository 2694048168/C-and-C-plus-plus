#ifndef __TEST_BENCHMARK__HPP__
#define __TEST_BENCHMARK__HPP__
// #pragma once

#include <functional>

// 使用typedef定义函数指针
// typedef int (*test_func)(int, unsigned long);

// 使用using定义函数指针
using test_func = int (*)(int, unsigned long);

void test_driver(test_func *func_list, int argc = 0, const char **argv = 0);
void test_driver(test_func f, int argc = 0, const char **argv = 0);

std::function<int (*)(int, unsigned long)> func;

#endif /* __TEST_BENCHMARK__HPP__ */
