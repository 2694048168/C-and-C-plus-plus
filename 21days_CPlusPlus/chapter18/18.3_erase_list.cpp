#include <iostream>
#include <list>

// std::list::erase() 有两个重载版本
// 1. 接受一个迭代器参数并删除迭代器指向的元素
// 2. 接受两个迭代器参数并删除指定范围的所有元素

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
  std::list<int> linkInts {4, 3, 5, 42, -32, 2021};
  // store an iterator obtained in using insert()
  auto val2 = linkInts.insert(linkInts.begin(), 2);
  std::cout << "Initial contents of the list:" << std::endl;
  DisplayContents(linkInts);

  std::cout << "After erasing element " << *val2 << ": " << std::endl;
  linkInts.erase(val2);
  DisplayContents(linkInts);

  linkInts.erase(linkInts.begin(), linkInts.end());
  // 调用成员函数 linkInts.clear(); 直接清空容器
  std::cout << "Numbers of elements after erasing range: " << linkInts.size() << std::endl;

  return 0;
}

// $ g++ -o main 18.3_erase_list.cpp 
// $ ./main.exe 

// Initial contents of the list:
// 2 4 3 5 42 -32 2021
// After erasing element 2:
// 4 3 5 42 -32 2021
// Numbers of elements after erasing range: 0