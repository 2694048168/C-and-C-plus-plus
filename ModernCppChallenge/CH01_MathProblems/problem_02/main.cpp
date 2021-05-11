/**
 * @file    : main.cpp
 * @brief   : Greatest common divisor
 * @author  : Wei Li
 * @date    : 2021-05-07
*/

// 写一个程序，给定两个正整数，计算并打印出最大公约数。

// Solution
// The greatest common divisor (gcd) of two or more non-zero integers, 
// is the greatest positive integer that divides all of them. 
// an efficient method is Euclid's algorithm. 

// 两个或多个非零整数的最大公约数（简称 gcd），是将所有整数除的最大正整数。
// 一种有效的方法是欧几里德算法:
// gcd(a,0) = a
// gcd(a,b) = gcd(b, a mod b)

// Euclid's algorithm 也称之为 辗转相除法：
// 1. 当两个数相等时，其中任意一个就是它们的最大公约数，因为它们的余数为 0；
// 2. 当两个数不相等时，用较大数除以较小数，当余数不为 0 时，这时
// 使较小数作为被除数，余数作为除数，继续 2 的操作，
// 直至余数为 0,也就是这两个数相等时，其中任一数为最大公约数。

#include <iostream>
#include <numeric> // C++ 17 std::gcd()

// recursive function
unsigned int gcd(unsigned int const a, unsigned int const b)
{
  return b == 0 ? a : gcd(b, a % b);
}

// non-recursive function
// unsigned int gcd(unsigned int a, unsigned int b)
// {
//   while (b != 0)
//   {
//     unsigned int r = a % b;
//     a = b;
//     b = r;
//   }
//   return a;
// }

// In C++17 there is a constexpr function called gcd() in the header <numeric> 
// that computes the greatest common divisor of two numbers.
// https://en.cppreference.com/w/cpp/numeric/gcd

int main(int argc, char** argv)
{
  unsigned int a = 0;
  unsigned int b = 0;
  std::cout << "Please input two positive integers: ";
  std::cin >> a >> b;

  std::cout << gcd(a, b) << std::endl;
  // C++ 17 <numeric>
  std::cout << std::gcd(a, b) << std::endl;
  
  return 0;
}

// ------------------------------------------------
// TEST
// $ ./problem_02.exe 
// Please input two positive integers: 256 64
// 64
// 64

// $ ./problem_02.exe 
// Please input two positive integers: 278 33
// 1
// 1

// $ ./problem_02.exe 
// Please input two positive integers: 99 3
// 3
// 3
// ------------------------------------------------