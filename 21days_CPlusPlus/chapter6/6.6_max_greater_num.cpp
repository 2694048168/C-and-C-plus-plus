#include <iostream>

int main(int argc, char** argv)
{
  std::cout << "Enter two numbers" << std::endl;
  int numOne = 0, numTwo = 0;
  std::cin >> numOne;
  std::cin >> numTwo;

  // 三目运算符，条件表达式
  int max = (numOne > numTwo) ? numOne : numTwo;

  std::cout << "The greater of " << numOne << " and " << numTwo << " is " << max << std::endl;

  return 0;
}

// $ g++ 6.6_max_greater_num.cpp -o main 
// $ ./main.exe 

// Enter two numbers
// 32
// 23
// The greater of 32 and 23 is 32