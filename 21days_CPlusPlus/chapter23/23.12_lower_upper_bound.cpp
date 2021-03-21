#include <iostream>
#include <algorithm>
#include <string>
#include <list>

// 在有序集合中插入元素
// std::lower_bound();
// std::upper_bound();

template <typename T>
void DisplayContainer(const T &container)
{
  for (auto element = container.cbegin(); element != container.cend(); ++element)
  {
    std::cout << *element << ' ';
  }
  std::cout << std::endl;
}

int main(int argc, char** argv)
{
  std::list<std::string> names { "John", "Brad", "jack", "sean", "Anna" }; 
  std::cout << "-----------------------------------------" << std::endl;
  std::cout << "Sorted contents of the list are: " << std::endl;
  names.sort();
  DisplayContainer(names);
  std::cout << "-----------------------------------------" << std::endl;

  std::cout << "Lowest index where \"Brad\" can be inserted is: ";
  auto minPos = std::lower_bound(names.begin(), names.end(), "Brad");
  std::cout << std::distance(names.begin(), minPos) << std::endl;

  std::cout << "The highest index where \"Brad\" can be inserted is: ";
  auto maxPos = std::upper_bound(names.begin(), names.end(), "Brad");
  std::cout << std::distance(names.begin(), maxPos) << std::endl;

  std::cout << "-----------------------------------------" << std::endl;
  std::cout << "List after inserting Brad in sorted order: " << std::endl;
  names.insert(minPos, "Brad");
  DisplayContainer(names);
  
  return 0;
}

// $ g++ -o main 23.12_lower_upper_bound.cpp 
// $ ./main.exe

// -----------------------------------------
// Sorted contents of the list are:
// Anna Brad John jack sean
// -----------------------------------------
// Lowest index where "Brad" can be inserted is: 1
// The highest index where "Brad" can be inserted is: 2
// -----------------------------------------
// List after inserting Brad in sorted order:
// Anna Brad Brad John jack sean
