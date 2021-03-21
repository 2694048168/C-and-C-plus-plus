#include <iostream>
#include <list>

// list 独特之处在于，指向元素的迭代器在 list 重新排列之后依然有效
// std::list 提供成员方法 sort() and reverse()
// STL Algorithm 提供了同样的方法

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
  std::list<int> linkInts {1, 3, 5, 42, 2020, 2021};

  std::cout << "Initial contents of list: " << std::endl;
  DisplayContents(linkInts);

  linkInts.reverse();

  std::cout << "Contents of list after using reverse() " << std::endl;
  DisplayContents(linkInts);

  return 0;
}

// $ g++ -o main 18.4_reverse_list.cpp 
// $ ./main.exe 

// Initial contents of list: 
// 1 3 5 42 2020 2021 
// Contents of list after using reverse() 
// 2021 2020 42 5 3 1 