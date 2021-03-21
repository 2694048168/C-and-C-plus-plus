#include <iostream>
#include <list>
#include <vector>

// std::list 是一种泛型实现，需要实例化才能使用成员函数
// 迭代器 iterator
// std::list<T>::const_iterator elementList;
// std::list<T>::iterator elementList;
// 允许在开头和末尾插入元素

template <typename T>
void DisplayContents(const T& container)
{
  for (auto element = container.cbegin(); element != container.cend(); ++element)
  {
    std::cout << *element << " ";
  }
  std::cout << std::endl;
}

int main(int argc, char** argv)
{
  // 1. 利用构造函数的各个重载版本
  // instantiate an empty list.
  std::list<int> linkIist;

  // instantiate a list with 10 integers.
  std::list<int> listWithInteger(10);

  // instantiate a list with 4 integers , each value 99.
  std::list<int> listWithIntegerInit(10, 99);

  // create an exact copy of an existint list.
  std::list<int> listCopyAnother(listWithIntegerInit);

  // a vector with 10 integers, each 2021.
  std::vector<int> vecIntegers(10, 2021);

  // instantiate a list using values from another container.
  std::list<int> listContainCopyAnother(vecIntegers.cbegin(), vecIntegers.cend());

  // 2. 在开头和末尾插入元素
  // C++11 list initialization
  std::list<int> linkInt {102, 42};
  linkIist.push_front(10);
  linkIist.push_back(2020);
  linkIist.push_front(12);
  linkIist.push_back(2021);

  DisplayContents(linkIist);
  
  return 0;
}

// $ g++ -o main 18.1_list.cpp 
// $ ./main.exe 
// 12 10 2020 2021