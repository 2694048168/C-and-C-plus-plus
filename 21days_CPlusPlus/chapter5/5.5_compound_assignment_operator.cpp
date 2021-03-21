#include <iostream>

int main(int argc, char** argv)
{
  std::cout << "Enter a number: ";
  int value = 0;
  std::cin >> value;

  value += 8;
  std::cout << "After += 8, value = " << value << std::endl;

  value -= 2;
  std::cout << "After = 8, value = " << value << std::endl;

  value *= 4;
  std::cout << "After *= 8, value = " << value << std::endl;

  value /= 4;
  std::cout << "After /= 8, value = " << value << std::endl;

  value %= 1000;
  std::cout << "After %= 8, value = " << value << std::endl;

  // Note: henceforth assignment happens with cout
  std::cout << "After <<= 1, value = " << (value <<= 1) << std::endl;
  std::cout << "After >>= 2, value = " << (value >>= 2) << std::endl;

  std::cout << "After |= 0X55, value = " << (value |= 0X55) << std::endl;
  std::cout << "After ^= 0X55, value = " << (value ^= 0X55) << std::endl;
  std::cout << "After ~= 0X55, value = " << (value &= 0X55) << std::endl;

  // 复合赋值运算符
  // 使用运算符 sizeof 来计算变量所占用的内存大小
  // sizeof(variable) or sizeof(type)
  // sizeof 是一个运算符，而且不能重载

  std::cout << "==================================" << std::endl;
  std::cout << "Use sizeof to determine memory used by array" << std::endl;
  int numbers[100] = {0};

  std::cout << "Bytes used by an int: " << sizeof(int) << std::endl;
  std::cout << "Bytes used by numbers: " << sizeof(numbers) << std::endl;
  std::cout << "Bytes used by an element: " << sizeof(numbers[0]) << std::endl;

  // 所有运算符有两大特性
  // 优先级 和 结合性（左右结合性）
  // 建议使用 括号，便于理解

  return 0;
}

// $ g++ -o  main 5.5_compound_assignment_operator.cpp      
// $ ./main
// Enter a number: 16
// After += 8, value = 24
// After = 8, value = 22
// After *= 8, value = 88
// After /= 8, value = 22
// After %= 8, value = 22
// After <<= 1, value = 44
// After >>= 2, value = 11
// After |= 0X55, value = 95
// After ^= 0X55, value = 10
// After ~= 0X55, value = 0
// ==================================
// Use sizeof to determine memory used by array
// Bytes used by an int: 4
// Bytes used by numbers: 400
// Bytes used by an element: 4