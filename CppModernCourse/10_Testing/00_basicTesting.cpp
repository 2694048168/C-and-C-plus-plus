/**
 * @file 00_basicTesting.cpp
 * @author Wei Li (weili_yzzcq@163.com)
 * @brief 
 * @version 0.1
 * @date 2024-05-02
 * 
 * @copyright Copyright (c) 2024
 * 
 */

#include <cstdio>
#include <stdexcept>

/**
 * @brief Unit Tests 单元测试
 * 集中的, 有结合力的代码集合称为单元(如函数或类), 单元测试是验证单元完全符合程序员的预期的过程;
 * 好的单元测试将被测试的单元与其依赖项隔离开来, 有时这可能很难做到, 因为该单元可能依赖于其他单元;
 * 在这种情况下, 可以使用模拟对象(mock)来代替依赖项;
 * *模拟对象是仅在测试期间使用的假对象, 在测试期间对单元依赖项的行为方式进行细粒度控制;
 * 模拟对象还可以记录单元如何与它们交互, 因此可以测试单元是否按预期与其依赖项交互;
 * 使用模拟对象还可以模拟罕见事件, 例如系统内存不足(通过对它们进行编程以抛出异常).
 * 
 * ---- 集成测试 Integration Tests
 * 将一系列单元一起测试称为集成测试; 集成测试也可以指软件和硬件之间的交互测试;
 * 集成测试是单元测试之上的一种重要测试, 因为它可以确保编写的软件作为一个系统协同工作;
 * *这种测试是单元测试的补充, 无法取代单元测试;
 * 
 * ---- 验收测试 Acceptance Tests
 * 验收测试可确保软件满足客户的所有要求, 高性能软件团队可以使用验收测试来指导开发;
 * 一旦所有验收测试通过, 软件就是可交付的, 因为验收测试是代码库的一部分,
 * 所以内置了防止重构或功能回归的保护机制, 以防止在添加新功能的过程中破坏现有功能.
 * 
 * ---- 性能测试 Performance Tests
 * 性能测试评估软件是否满足效能要求, 例如执行速度或内存/功耗要求;
 * *优化代码基本上是一种经验练习,可以并且应该知道是代码的哪些部分导致了性能瓶颈,
 * 但除非进行测量, 否则无法确定; 此外,除非再次测量, 否则无法知道为优化而实施的代码更改是否能够提高性能;
 * 可以使用性能测试来检测代码并提供相关测量指标;
 * *仪器（instrumentation）是一种用于测量产品性能、检测错误和记录程序执行方式的技术;
 * 有时, 客户会提出严格的性能要求(例如, 计算时间不能超过 100ms, 系统不能分配超过 1MB 的内存）.
 * 
 * ?The process of writing a test that encodes a requirement before 
 * ?implementing the solution is the fundamental idea behind TDD.
 * ?TDD practitioners have a mantra: red, green, refactor. 
 * ?Red is the first step, and it means to implement a failing test.
 * 
 * !断言: 单元测试的基石; Assertions: The Building Blocks of Unit Tests
 * 单元测试最重要的组成部分是断言(assertion), 它检查是否满足某些条件, 如果不满足条件, 则封闭测试失败.
 * 
 */

constexpr void assert_that(bool statement, const char *message)
{
    if (!statement)
        throw std::runtime_error{message};
}

// ----------------------------------
int main(int argc, const char **argv)
{
    printf("the unit-test and assertion\n");

    assert_that(1 + 2 > 2, "Something is profoundly wrong with the universe.\n");

    static_assert(1 + 2 > 2, "something is error\n");

    assert_that(24 == 42, "This assertion will generate an exception.\n");

    return 0;
}
