/**
 * @file    : main.cpp
 * @brief   : Sum of naturals divisible by 3 and 5
 * @author  : Wei Li
 * @date    : 2021-05-07
*/

// 编写一个程序，计算并打印所有可被 3 或 5 整除的自然数之和，
// 直到用户输入的给定限制。

// Solution
// 1. 遍历从 3（1 和 2 不能被 3 整除，所以测试它们是没有意义的）到用户输入的限制的所有数字
// 2. 使用模运算检查一个数除以 3 或 5 的余数部分是否为 0
// 3. 求和到更大极限的诀窍是使用 long long 而不是 int 或 long 导致求和结果溢出

#include <iostream>

int main(int argc, char** argv)
{
  unsigned int limit = 0;
  std::cout << "Please input upper limit:";
  std::cin >> limit;

  // the trick to being able to sum up to a larger limit 
  // unsigned long long sum = 0;
  // unsigned long sum = 0;
  unsigned int sum = 0;
  for (unsigned int i = 3; i < limit; ++i)
  {
    // modulo operation to check 
    if (i % 3 == 0 || i % 5 == 0)
    {
      // std::cout << i << ' ';
      sum += i;
    }
  }
  // std::cout << std::endl;

  std::cout << "Sum of natural numbers satisfying conditions: " << sum << std::endl;

  return 0;
}

// ---------------------------------------------------
// TEST
// $ g++ main.cpp 
// $ ./a.exe
// Please input upper limit:32
// 3 5 6 9 10 12 15 18 20 21 24 25 27 30
// Sum of natural numbers satisfying conditions: 225
// ---------------------------------------------------
