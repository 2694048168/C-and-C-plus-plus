#include <iostream>
#include <algorithm>
#include <vector>

// 参考文件 21.4_binary_function.cpp
// 全部没有变化，只是将二元函数的函数对象，替换为 lambda 函数
// 代码简洁了很多

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
                 [] (int a, int b) {return a * b;});

  std::cout << "The contents of the first vector are: " << std::endl; 
  DisplayContainer(multiplicands);

  std::cout << "The contents of the second vector are: " << std::endl; 
  DisplayContainer(multipliers);

  std::cout << "The contents of the result vector are: " << std::endl; 
  DisplayContainer(vecResult);

  return 0;
}

// $ g++ -o main 22.4_lambda_binay_function.cpp 
// $ ./main.exe 

// The contents of the first vector are:  
// 0 1 2 3 4
// The contents of the second vector are: 
// 100 101 102 103 104
// The contents of the result vector are: 
// 0 101 204 309 416