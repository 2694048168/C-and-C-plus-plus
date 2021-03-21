#include <iostream>
#include <deque>
#include <algorithm>

// STL deque class
// std::deque 也是一个动态数组类，支持在开头和末尾插入和删除元素

int main(int argc, char** argv)
{
  // define a deque of integer.
  std::deque<int> intDeque;

  // insert integers at the bottom of the array.
  intDeque.push_back(3);
  intDeque.push_back(5);
  intDeque.push_back(42);

  // insert integers at the top of the array.
  intDeque.push_front(25);
  intDeque.push_front(2021);
  intDeque.push_front(2);

  std::cout << "The contents of the deque after inserting elements at the top and bottom are:" << std::endl;

  // diaplay contents on the screen
  for (size_t i = 0; i < intDeque.size(); ++i)
  {
    std::cout << "Element [" << i << "] = " << intDeque[i] << std::endl;
  }
  std::cout << std::endl;

  // erase an element at the top.
  intDeque.pop_front();

  // erase an element at the bottom.
  intDeque.pop_back();

  std::cout << "=================================================" << std::endl;
  std::cout << "The contents of the deque after erasing an element from the top and bottom are: " << std::endl;

  // display contents again with iterator.
  // auto == std::deque::iterator 
  for (auto element = intDeque.begin(); element != intDeque.end(); ++element)
  {
    // compute offset 计算偏移量
    size_t offset = std::distance(intDeque.begin(), element);
    std::cout << "Element [" << offset << "] = " << *element << std::endl;
  }
  
  return 0;
}

// $ g++ -o main 17.6_deque_class_STL.cpp 
// $ ./main.exe 

// The contents of the deque after inserting elements at the top and bottom are:
// Element [0] = 2
// Element [1] = 2021
// Element [2] = 25
// Element [3] = 3
// Element [4] = 5
// Element [5] = 42

// =================================================
// The contents of the deque after erasing an element from the top and bottom are:
// Element [0] = 2021
// Element [1] = 25
// Element [2] = 3
// Element [3] = 5