#include <iostream>

int main(int argc, char** argv)
{
  std::cout << "Enter two integers: " << std::endl;
  int numOne = 0, numTwo = 0;
  std::cin >> numOne;
  std::cin >> numTwo;

  bool isEqual = (numOne == numTwo);
  std::cout << "Result of equality test: " << isEqual << std::endl;

  bool isUnequal = (numOne != numTwo);
  std::cout << "Result of inequality test: " << isUnequal << std::endl;

  bool isGreaterThan = (numOne > numTwo);
  std::cout << "Result of " << numOne << " > " << numTwo;
  std::cout << "test: " << isGreaterThan << std::endl;

  bool isLessThan = (numOne < numTwo);
  std::cout << "Result of " << numOne << " < " << numTwo;
  std::cout << "test: " << isLessThan << std::endl;

  bool isGreaterThanEquals = (numOne >= numTwo);
  std::cout << "Result of " << numOne << " >= " << numTwo;
  std::cout << "test: " << isGreaterThanEquals << std::endl;

  bool isLessThanEquals = (numOne > numTwo);
  std::cout << "Result of " << numOne << " <= " << numTwo;
  std::cout << "test: " << isLessThanEquals << std::endl;

  std::cout << "===========================" << std::endl;

  std::cout << "Result of equality test: " << std::boolalpha << isEqual << std::endl;
  std::cout << "Result of inequality test: " << std::boolalpha << isUnequal << std::endl;
  std::cout << "Result of " << numOne << " > " << numTwo;
  std::cout << "test: " << std::boolalpha << isGreaterThan << std::endl;
  std::cout << "Result of " << numOne << " < " << numTwo;
  std::cout << "test: " << std::boolalpha << isLessThan << std::endl;
  std::cout << "Result of " << numOne << " >= " << numTwo;
  std::cout << "test: " << std::boolalpha << isGreaterThanEquals << std::endl;
  std::cout << "Result of " << numOne << " <= " << numTwo;
  std::cout << "test: " << std::boolalpha << isLessThanEquals << std::endl;

  return 0;
}

// $ g++ -o main 5.2_relational_operator.cpp
// $ ./main.exe 
// Enter two integers: 
// 32
// 56
// Result of equality test: 0
// Result of inequality test: 1
// Result of 32 > 56test: 0
// Result of 32 < 56test: 1
// Result of 32 >= 56test: 0
// Result of 32 <= 56test: 0
// ===========================
// Result of equality test: false
// Result of inequality test: true
// Result of 32 > 56test: false
// Result of 32 < 56test: true
// Result of 32 >= 56test: false
// Result of 32 <= 56test: false