#include <iostream>
#include <forward_list>

// std::forward_list 是一种单向链表，只允许一个方向遍历
// 与 std::list 类似，只能向一个方向移动迭代器
// 只有 push_back()

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
  std::forward_list<int> forwardListIntegers {3, 4, 2, 2, 0};
  forwardListIntegers.push_front(1);

  std::cout << "Contents of forward_list: " << std::endl;
  DisplayAsContents(forwardListIntegers);

  forwardListIntegers.remove(2);
  forwardListIntegers.sort();

  std::cout << "Contents after removing 2 and sorting: " << std::endl;
  DisplayAsContents(forwardListIntegers);
  
  return 0;
}

// $ g++ -o main 18.7_forward_list.cpp 
// $ ./main.exe 

// Contents of forward_list: 
// 1 3 4 2 2 0
// Contents after removing 2 and sorting:
// 0 1 3 4