#include <iostream>
#include <set>

// std::set; std::multiset; std::unordered_set; std::unordered_multiset
// 使用成员方法 set.erase() 按键删除元素

template <typename T>
void DisplayContents(const T& container)
{
  for (auto element = container.cbegin(); element != container.cend(); ++element)
  {
    std::cout << *element << " ";
  }
  std::cout << std::endl;
}

// typedef std::multiset<int> MSETINT;

int main(int argc, char** argv)
{
  // 1. create set / multiset.
  std::multiset<int> setInt2 {20, 21, 42, -1, 24, -1};
  std::cout << "Contents of the multiset: " << setInt2.size() << std::endl;
  DisplayContents(setInt2);

  std::cout << "Please enter a number to erase from the multiset: ";
  int input = 0;
  std::cin >> input;

  std::cout << "Erasing " << setInt2.count(input) << " instances of value " << input << std::endl;
  setInt2.erase(input);

  std::cout << "multiset now contains " << setInt2.size() << " elements: " << std::endl;
  DisplayContents(setInt2);

  return 0;
}

// $ g++ -o main 19.3_erase_set.cpp 
// $ ./main.exe 

// Contents of the multiset: 6
// -1 -1 20 21 24 42
// Please enter a number to erase from the multiset: -1
// Erasing 2 instances of value -1
// multiset now contains 4 elements:
// 20 21 24 42