#include <iostream>

int main(int argc, char** argv)
{
  std::cout << "This program will calculate Fibonacci Numbers at a time." << std::endl;
  std::cout << "Please enter the numbers of the Fibonacci Numbers: ";
  int numsToCalculate = 0;
  std::cin >> numsToCalculate;

  int fibonacci_one = 0, fibonacci_two = 1;
  char wantMore = '\0';
  std::cout << fibonacci_one << " " << fibonacci_two << " ";

  // 计算斐波纳契数列
  do
  {
    for (size_t i = 0; i < numsToCalculate; ++i)
    {
      std::cout << fibonacci_two + fibonacci_one << " ";
      int fibonacci_two_temp = fibonacci_two;
      fibonacci_two = fibonacci_two + fibonacci_one;
      fibonacci_one = fibonacci_two_temp;
    }
    std::cout << std::endl << "Do you want more numbers (y/n)? ";
    std::cin >> wantMore;
  } while (wantMore == 'y');
    
  std::cout << "Goodbye!" << std::endl;

  return 0;
}


// $ g++ -o main 6.17_test_fiboacci.cpp
// $ ./main

// This program will calculate Fibonacci Numbers at a time.    
// Please enter the numbers of the Fibonacci Numbers: 34
// 0 1 1 2 3 5 8 13 21 34 55 89 144 233 377 610 987 1597 2584 4181 6765 10946 17711 28657 46368 75025 121393 196418 317811 
// 514229 832040 1346269 2178309 3524578 5702887 9227465       
// Do you want more numbers (y/n)? y
// 14930352 24157817 39088169 63245986 102334155 165580141 267914296 433494437 701408733 1134903170 1836311903 -1323752223 
// 512559680 -811192543 -298632863 -1109825406 -1408458269 1776683621 368225352 2144908973 -1781832971 363076002 -1418756969 -1055680967 1820529360 764848393 -1709589543 -944741150 1640636603 695895453 -1958435240 -1262539787 1073992269 -188547518
// Do you want more numbers (y/n)? n
// Goodbye!


// 后面出现的异常数值表明溢出了
// 这也证明了 斐波纳契数列 的强大，累计或者复利的力量强大！！！