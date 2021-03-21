#include <iostream>

int main(int argc, char** argv)
{
  // 在 C++ 中，只有 零 才是 false
  // 其他任何非零数值，判断都为 true

  std::cout << "Enter two integers: " << std::endl;
  int numOne = 0, numTwo = 0;
  std::cin >> numOne;
  std::cin >> numTwo;

  std::cout << "Enter \'m\' to multiply, anything else to add: ";
  char userSeclection = '\0';
  std::cin >> userSeclection;

  int result = 0;
  
  // 分支流程控制
  if (userSeclection == 'm')
  {
    result = numOne * numTwo;
  }
  else
  {
    result = numOne + numTwo;
  }
  
  std::cout << "result is: " << result << std::endl; 

  return 0;
}

// $ g++ -o main 6.1_branch.cpp 
// $ ./main.exe
// Enter two integers: 
// 32
// 1
// Enter 'm' to multiply, anything else to add: m
// result is: 32

// $ ./main.exe 
// Enter two integers: 
// 32
// 23
// Enter 'm' to multiply, anything else to add: a
// result is: 55