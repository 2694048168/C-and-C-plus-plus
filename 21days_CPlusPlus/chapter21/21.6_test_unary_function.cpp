#include <iostream>
#include <algorithm>
#include <vector>

// a structure as a unary predicate. 
template <typename numberType>
struct Double
{
  int usageCount;
  Double () : usageCount(0) {};

  // unary predicate
  // void operator () (const numberType& element) const
  void operator () (const numberType& element)
  {
    ++usageCount;
    std::cout << element * 2 << ' ';
  }
};

template <typename elementType>
void DisplayContainer(elementType& container)
{
  for (auto element = container.begin(); element != container.end(); ++element)
  {
    std::cout << (*element) << ' ';
  }
  std::cout << std::endl;
}

int main(int argc, char** argv)
{
  std::vector<int> numInVec {42, 24, 66, 99, 4, 1};
  std::cout << "==============================" << std::endl;
  std::cout << "The origin integers: " << std::endl;
  DisplayContainer(numInVec);

  // display the array of integers.
  std::cout << "==============================" << std::endl;
  std::cout << "The double integers: " << std::endl;
  auto result = std::for_each(numInVec.begin(), numInVec.end(),
                Double<int>());

  std::cout << std::endl << "==============================" << std::endl;
  std::cout << "The unary predicate used : " << result.usageCount << std::endl;
  std::cout << "==============================" << std::endl;

  return 0;
}

// $ g++ -o main 21.6_test_unary_function.cpp 
// $ ./main.exe

// ==============================
// The origin integers:
// 42 24 66 99 4 1
// ==============================
// The double integers:
// 84 48 132 198 8 2
// ==============================
// The unary predicate used : 6
// ==============================