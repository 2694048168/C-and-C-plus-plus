#include <iostream>

void MultiplyNumbers()
{
  std::cout << "Enter the first number: ";
  int firstNumber = 0;
  std::cin >> firstNumber;

  std::cout << "Enter the second number: ";
  int secondNumber = 0;
  std::cin >> secondNumber;

  // Multiply two numbers,store result in a variable
  int multiplicationResult = firstNumber * secondNumber;

  // Display result
  std::cout << firstNumber << " * " << secondNumber << " = " 
            << multiplicationResult << std::endl;
}

int main(int argc, char**argv)
{
  std::cout << "This program will help you multiply two numbers" << std::endl;

  // Multiply two numbers,store result in a variable
  MultiplyNumbers();

  // 局部变量，函数封装的
  // std::cout << firstNumber << " * " << secondNumber << " = " 
            // << multiplicationResult << std::endl;

  return 0;
}