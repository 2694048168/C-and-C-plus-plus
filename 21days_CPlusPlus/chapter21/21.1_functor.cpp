/**函数对象 functor
 * 概念上，函数对象是用作函数的对象；
 * 实现上，函数对象是实现 operator() 函数运算符的类的对象
 * C++ 常用于 STL 算法的函数对象有两种类型
 * 1. 一元谓词：接受一个参数的函数，返回布尔类型的值
 * 2. 二元谓词：接受两个参数的函数，返回布尔类型的值
 * 
 * 返回布尔类型的函数对象常用于需要进行判断的算法
 * 组合两个函数对象的函数对象称之为自适应函数对象
 */

// 只对一个参数进行操作的函数称之为一元函数
// a unary function
/*
template <typename elementType>
void FuncDisplayElement (const elementType& element)
{
  std::cout << element << " ";
};
*/

// 该函数也可采用另一种表现形式，实现在包含类或者结构中的 operator()
// struct that can behave as a unary function
/*
template <typename elementType>
struct DisplayElement
{
  void operator () (const elementType& element)
  {
    std::cout << element << " ";
  }
};

template <typename elementType>
class DisplayElement
{
public:
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

// a unary function
template <typename elementType>
void FuncDisplayElement (const elementType& element)
{
  std::cout << element << " ";
};

// struct that can behave as a unary function
template <typename elementType>
struct DisplayElement
{
  void operator () (const elementType& element)
  {
    std::cout << element << " ";
  }
};

int main(int argc, char** argv)
{
  std::vector<int> numIntVec {0, 1, 2, 3, 4, 5, -1, 42};
  std::cout << "Vector of integers contains: " << std::endl;

  // std::for_each(, ,); 将集合内容进行处理
  // 第一个参数集合的起点；第二参数集合的终点；第三参数 一元函数
  std::for_each(numIntVec.begin(), numIntVec.end(), DisplayElement<int>());
  std::cout << std::endl << "============================================" << std::endl;
  std::for_each(numIntVec.begin(), numIntVec.end(), FuncDisplayElement<int>);

  // display the list of characters.
  std::list<char> charInList {'a', 'z', 'k', 'd'};
  std::cout << std::endl << std::endl << "List of chars contains: " << std::endl;
  std::for_each(charInList.begin(), charInList.end(), DisplayElement<char>());
  std::cout << std::endl << "============================================" << std::endl;
  std::for_each(charInList.begin(), charInList.end(), FuncDisplayElement<char>);

  // C++11 lambda 表达式，即匿名函数对象，简化一元函数的声明
  std::cout << std::endl << std::endl << "Using lambda expression as the unary function." << std::endl;
  std::cout << "============================================" << std::endl;
  std::for_each(numIntVec.begin(), numIntVec.end(), [] (int& element) {std::cout << element << ' ';});
  
  return 0;
}

// $ g++ -o main 21.1_functor.cpp 
// $ ./main.exe

// Vector of integers contains: 
// 0 1 2 3 4 5 -1 42
// ============================================
// 0 1 2 3 4 5 -1 42

// List of chars contains:
// a z k d
// ============================================
// a z k d

// Using lambda expression as the unary function.
// ============================================
// 0 1 2 3 4 5 -1 42