#include <iostream>
#include <vector>

// 删除 std::vector 中的元素，使用成员函数 pop_back() 删除末尾元素
// push_back() and pop_back() 所需时间是固定的
template <typename T>
void DisplayVector(const std::vector<T>& inVec)
{
  for (auto element = inVec.cbegin(); element != inVec.cend(); ++element)
  {
    std::cout << *element << " ";
  }
  std::cout << std::endl;
}

int main(int argc, char** argv)
{
  std::vector<int> integers;

  integers.push_back (42);
  integers.push_back (2042);
  integers.push_back (2);
  integers.push_back (4);
  integers.push_back (24);

  std::cout << "Vector contains " << integers.size() << " elements: " << std::endl;
  DisplayVector(integers);

  // erase one element at the end.
  integers.pop_back();

  std::cout << "========================================" << std::endl;
  std::cout << "After a call to pop_back()" << std::endl;
  std::cout << "Vector contains " << integers.size() << " elements: " << std::endl;
  DisplayVector(integers);
  
  return 0;
}

// $ g++ -o main 17.4_pop_back_vector.cpp 
// $ ./main.exe 

// Vector contains 5 elements: 
// 42 2042 2 4 24
// ========================================
// After a call to pop_back()
// Vector contains 4 elements:
// 42 2042 2 4