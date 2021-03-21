#include <iostream>

// 使用递归函数计算 斐波纳契数列
int GetFibNumber(int fibIndex)
{
  if (fibIndex < 2)
  {
    return fibIndex;
  }
  else
  {
    // using recursion if fibIndex >= 2
    return GetFibNumber(fibIndex - 1) + GetFibNumber(fibIndex - 2);
  }
}

int main(int argc, char** argv)
{
  std::cout << "Enter 0-based index of desired Fibonacci Number: ";
  int index = 0;
  std::cin >> index;

  std::cout << "Fibonacci number is: " << GetFibNumber(index) << std::endl;

  return 0;
}

// $ g++ -o main 7.4_recursion_function.cpp 
// $ ./main.exe 
// Enter 0-based index of desired Fibonacci Number: 33
// Fibonacci number is: 3524578