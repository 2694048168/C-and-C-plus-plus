// 该函数也可采用另一种表现形式，实现在包含类或者结构中的 operator()
// 使用类或者结构体形式表现一元函数，可以记录储存状态的函数对象
// struct that can behave as a unary function
/*
template <typename elementType>
struct DisplayElement
{
  int count;

  DisplayElement(){ count = 0;}
  
  void operator () (const elementType& element)
  {
    std::cout << element << " ";
  }
};
*/

#include <iostream>
#include <algorithm>
#include <vector>
#include <list>

// struct that can behave as a unary function
template <typename elementType>
struct DisplayElement
{
  int count;

  DisplayElement() : count(0) {}

  void operator () (const elementType& element)
  {
    ++count;
    std::cout << element << " ";
  }
};

int main(int argc, char** argv)
{
  std::vector<int> numIntVec {0, 1, 2, 3, 4, 5, -1, 42};
  std::cout << "Vector of integers contains: " << std::endl;
  DisplayElement<int> result;
  result = std::for_each(numIntVec.begin(), numIntVec.end(), DisplayElement<int>());
  std::cout << std::endl << "Functor invoked " << result.count << " TIMES." << std::endl;

  return 0;
}

// $ g++ -o main 21.2_class_functor.cpp
// $ ./main.exe

// Vector of integers contains: 
// 0 1 2 3 4 5 -1 42       
// Functor invoked 8 TIMES.