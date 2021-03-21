#include <iostream>
#include <algorithm>
#include <vector>

// 计算包含给定值或满足条件的元素数量
// size_t std::cout(start of range, end of range, element to count);
// size_t std::cout(start of range, end of range, unary predicate);
// unary predicate 可以是函数对象functor，也可是 lambda expression

// unary predicate for *_if functions.
template <typename elementType>
bool IsEven (const elementType& number)
{
  return ((number % 2) == 0);
}

int main(int argc, char** argv)
{
  std::vector<int> numInVec {2021, 3, 42, 35, 9, 0, 1010};

  size_t numZeros = std::count (numInVec.cbegin(), numInVec.cend(), 0);
  std::cout << "Number of instances of '0': " << numZeros << std::endl;
  std::cout << "========================================" << std::endl;

  size_t numEvenNums = std::count_if(numInVec.cbegin(), numInVec.cend(), IsEven<int>);
  std::cout << "Number of even elements: " << numEvenNums << std::endl;
  std::cout << "========================================" << std::endl;

  std::cout << "Number of odd elements: " << numInVec.size() - numEvenNums << std::endl;
  
  return 0;
}

// $ g++ -o main 23.2_count_algorithm.cpp 
// $ ./main.exe 

// Number of instances of '0': 1
// ========================================
// Number of even elements: 3
// ========================================
// Number of odd elements: 4
