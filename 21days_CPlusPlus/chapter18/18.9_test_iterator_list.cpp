#include <iostream>
#include <list>

template <typename T>
void DisplayAsContents(const T& container)
{
  for (auto element = container.cbegin(); element != container.cend(); ++element)
  {
    std::cout << *element << " ";
  }
  std::cout << std::endl;
}

int main(int argc, char** argv)
{
  std::list<int> listData {42, 24, 2020, 2021, 66, 99};
  DisplayAsContents(listData);

  std::list<int>::const_iterator elementV11 = listData.begin();
  std::cout << "The value of first iterator elementV11: " << *elementV11 << std::endl;

  listData.insert(listData.begin(), 2);
  DisplayAsContents(listData);

  std::cout << "The value of first iterator elementV11: " << *elementV11 << std::endl;
  
  return 0;
}

// $ g++ -o main 18.9_test_iterator_list.cpp
// $ ./main.exe

// 42 24 2020 2021 66 99 
// The value of first iterator elementV11: 42
// 2 42 24 2020 2021 66 99
// The value of first iterator elementV11: 42