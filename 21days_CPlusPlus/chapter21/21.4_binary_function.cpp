#include <iostream>
#include <algorithm>
#include <vector>

// 使用二元函数将两个范围元素相乘
template <typename elementType>
class Multiply
{
public:
  elementType operator () (const elementType& element1, const elementType& element2)
  {
    return (element1 * element2);
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
  std::vector<int> multiplicands {0, 1, 2, 3, 4};
  std::vector<int> multipliers {100, 101, 102, 103, 104};
  // a third container that holds the result of multiplication.
  std::vector<int> vecResult;

  // make space for the result of the multiplication.
  vecResult.resize(multipliers.size());
  std::transform(multiplicands.begin(), multiplicands.end(), 
                 multipliers.begin(), 
                 vecResult.begin(), 
                 Multiply<int>());

  std::cout << "The contents of the first vector are: " << std::endl; 
  DisplayContainer(multiplicands);

  std::cout << "The contents of the second vector are: " << std::endl; 
  DisplayContainer(multipliers);

  std::cout << "The contents of the result vector are: " << std::endl; 
  DisplayContainer(vecResult);

  return 0;
}

// $ g++ -o main 21.4_binary_function.cpp 
// $ ./main.exe

// The contents of the first vector are: 
// 0 1 2 3 4 
// The contents of the second vector are:
// 100 101 102 103 104
// The contents of the result vector are:
// 0 101 204 309 416