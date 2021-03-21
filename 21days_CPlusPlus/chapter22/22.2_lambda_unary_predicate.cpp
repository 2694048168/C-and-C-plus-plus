#include <iostream>
#include <algorithm>
#include <vector>

int main(int argc, char** argv)
{
  std::vector<int> numInVec {101, -4, 500, 21, 42, -1};

  std::cout << "Display elements in a vector using a lambda expression: " << std::endl;
  std::cout << "=====================================================" << std::endl;
  // 一元函数对应的 lambda 表达式
  std::for_each(numInVec.cbegin(), numInVec.cend(),
                [] (const int& element) {std::cout << element << ' ';});
  std::cout << std::endl << "=====================================================" << std::endl;

  auto evenNum = std::find_if(numInVec.cbegin(), numInVec.cend(),
                              [] (const int& num) {return ((num % 2) == 0);});

  if (evenNum != numInVec.cend())
  {
    std::cout << "Even number in collection is: " << *evenNum << std::endl;
  }

  return 0;
}

// $ g++ -o main 22.2_lambda_unary_predicate.cpp 
// $ ./main.exe 

// Display elements in a vector using a lambda expression: 
// =====================================================   
// 101 -4 500 21 42 -1
// =====================================================   
// Even number in collection is: -4