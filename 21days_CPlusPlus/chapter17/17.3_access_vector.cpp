#include <iostream>
#include <vector>

// 访问 std::vector 里面的元素
// 1. 使用数组语法访问：下标运算符 []
// 2. 使用指针语法访问：迭代器 iterator

int main(int argc, char** argv)
{
  std::vector<int> ingeters {50, 2, 234, 2021, 2020};

  // 1. 下标运算符
  for (size_t i = 0; i < ingeters.size(); ++i)
  {
    std::cout << "Element[" << i << "] = " << ingeters[i] << std::endl; 
  }

  // change value of 3rd element.
  ingeters[2] = 42;
  std::cout << "After replacement: " << "Element[2] = " << ingeters[2] << std::endl;
  std::cout << "============================================" << std::endl;

  // 2. 迭代器
  // 声明一个迭代器
  std::vector<int>::const_iterator element = ingeters.cbegin();
  // auto element = ingeters.cbegin();

  while (element != ingeters.end())
  {
    // using std::distance to compute offset; 计算偏移量 
    size_t index = std::distance (ingeters.cbegin(), element);

    std::cout << "Element at position " << index << " is: " << *element << std::endl;

    // move to the next element.
    ++element;
  }
  
  return 0;
}

// $ g++ -o main 17.3_access_vector.cpp 
// $ ./main.exe 

// Element[0] = 50
// Element[1] = 2
// Element[2] = 234
// Element[3] = 2021
// Element[4] = 2020
// After replacement: Element[2] = 42
// ============================================
// Element at position 0 is: 50
// Element at position 1 is: 2
// Element at position 2 is: 42
// Element at position 3 is: 2021
// Element at position 4 is: 2020