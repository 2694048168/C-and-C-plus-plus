/**标准模板库 Standard Template Library, STL
 * 使用 auto 让编译器确定类型
 * auto 表明必须初始化,编译器才能根据初始值推断变量的类型
 * 
 * 选择正确的容器, 否则将导致性能问题和可扩展性瓶颈
 * 
 * STL 字符串类 std::basic_string<T>
 * std::string: 基于 char 的 std::basic_string 具体化,用于操作简单字符串
 * std::wstring: 基于 wchar_t 的 std::basic_string 具体化,用于操作宽字符串,通常用于储存支持 Ucicode 字符
 */

#include <iostream>
#include <algorithm>
#include <deque>

int main(int argc, char** argv)
{

  // A dynamic array of integers
  std::deque<int> intArray;

  // insert sample integers into the array
  intArray.push_back(50);
  intArray.push_back(42);
  intArray.push_back(21);
  intArray.push_back(2021);
  intArray.push_back(2020);

  // std::cout << "The contents of the vector are: " << std::endl;
  std::cout << "The contents of the deque are: " << std::endl;

  // walk the deque and read values using an iterator
  // 使用 auto 自动推断类型
  // std::deque<int>::iterator arrIterator = intArray.begin();
  auto arrIterator = intArray.begin();

  while (arrIterator != intArray.end())
  {
    // write the value to the screen
    std::cout << *arrIterator << std::endl;

    // increment the iterator to access the next element
    ++arrIterator;
  }

  // find an element(2021) using the 'find' algorithm
  auto eleFound = std::find(intArray.begin(), intArray.end(), 2021);

  // check if value was found
  if (eleFound != intArray.end())
  {
    // determine position of element using std::distance
    int elePos = std::distance(intArray.begin(), eleFound);
    std::cout << "Value " << *eleFound << " found in the vector at position: " << elePos << std::endl;
  }
  
  return 0;
}

// $ g++ -o main 15.2_auto_type.cpp 
// $ ./main.exe 
// The contents of the deque are: 
// 50
// 42
// 21
// 2021
// 2020
// Value 2021 found in the vector at position: 3