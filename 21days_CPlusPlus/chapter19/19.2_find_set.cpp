#include <iostream>
#include <set>

// std::set; std::multiset; std::unordered_set; std::unordered_multiset
// 使用成员方法 set.find() 按键查询元素，multi的返回第一个匹配

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
  // 1. create set / multiset.
  std::set<int> setInt1 {2020, 2021, 42, 24, -1};
  std::cout << "Contents of the set: " << std::endl;
  DisplayContents(setInt1);

  std::multiset<int> setInt2 {20, 21, 42, -1, 24, -1};
  std::cout << "Contents of the multiset: " << std::endl;
  DisplayContents(setInt2);

  // try finding an element.
  auto elementFound = setInt1.find(-1);
  // check if found
  if (elementFound != setInt1.end())
  {
    std::cout << "Element in set " << *elementFound << " found." << std::endl;
  }
  else
  {
    std::cout << "Element not found in set." << std::endl;
  }

  // try finding an element.
  auto elementFound2 = setInt2.find(-1);
  // check if found
  if (elementFound2 != setInt2.end())
  {
    std::cout << "Element in multiset " << *elementFound2 << " found." << std::endl;
  }
  else
  {
    std::cout << "Element not found in multiset." << std::endl;
  }
  
  // finding another
  auto anotherFound = setInt1.find(12345);
  if (anotherFound != setInt1.end())
  {
    std::cout << "Element in the set " << *anotherFound << " found." << std::endl;
  }
  else
  {
    std::cout << "Element 12345 not found in set." << std::endl;
  }

  // finding another
  auto anotherFound3 = setInt2.find(12345);
  if (anotherFound3 != setInt2.end())
  {
    std::cout << "Element in the multiset" << *anotherFound3 << " found." << std::endl;
  }
  else
  {
    std::cout << "Element 12345 not found in multiset." << std::endl;
  }

  return 0;
}

// $ g++ -o main 19.2_find_set.cpp 
// $ ./main.exe 

// Contents of the set: 
// -1 24 42 2020 2021
// Contents of the multiset:
// -1 -1 20 21 24 42
// Element in set -1 found.
// Element in multiset -1 found.       
// Element 12345 not found in set.     
// Element 12345 not found in multiset.