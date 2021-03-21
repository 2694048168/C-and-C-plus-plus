#include <iostream>
#include <vector>

// 动态数组
// STL 通过 std::vector 提供解决方案
// std::vector<element_type> dynamic_array;
// std::vector<element_type>::const_iterator elementValue;
// std::vector<element_type>::iterator elementValue;

int main(int argc, char** argv)
{
  // 1. 构造函数初始化
  std::vector<int> integers;

  // vector initialized using C++11 list initialization.
  std::vector<int> initVector {202, 2021, -42, 42};

  // instance a vector with 10 elements(it can still grow)
  std::vector<int> tenElements (10);

  // instantiate a vector with 10 elements, each initialized to 90.
  std::vector<int> tenElementsInit (10, 90);

  // initialize vector to the contents of another
  std::vector<int> copyVector (tenElementsInit);

  // vctor initialized to 5 elements from another using iterators
  std::vector<int> partialCopy (tenElements.cbegin(), tenElements.cbegin() + 5);

  // 2. 使用成员函数 push_back() 在末尾插入元素 
  integers.push_back(20);
  integers.push_back(2021);
  integers.push_back(42);
  integers.push_back(24);
  integers.push_back(40);
  // size 函数，返回 vector 中的元素数量
  std::cout << "The vector contains " << integers.size() << " elements." << std::endl;
  
  return 0;
}

// $ touch 17.1_vector.cpp
// $ g++ -o main 17.1_vector.cpp 
// $ ./main.exe 
// The vector contains 5 elements.