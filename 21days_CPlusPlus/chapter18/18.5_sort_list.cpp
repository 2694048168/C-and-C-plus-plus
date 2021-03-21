#include <iostream>
#include <list>

// list 独特之处在于，指向元素的迭代器在 list 重新排列之后依然有效
// std::list 提供成员方法 sort() and reverse()
// list.sort() 提供两个版本
// 1. 不需要参数
// 2. 接受一个二元谓词函数作为参数，指定排序标准

bool SortPredicate_Descending(const int& lhs, const int& rhs)
{
  // define criteria for list::sort: return true for desired order.
  return (lhs > rhs);
}

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
  std::list<int> linkInts {1, 3, 2021, 42, 2020, 0};

  std::cout << "Initial contents of list - " << std::endl;
  DisplayContents(linkInts);

  linkInts.sort();

  std::cout << "Order after sort() " << std::endl;
  // 默认升序排序
  DisplayContents(linkInts);

  // 二元谓词函数
  linkInts.sort(SortPredicate_Descending);
  std::cout << "Order after sort() with a predicate:" << std::endl; 
  DisplayContents(linkInts);

  return 0;
}

// $ g++ -o main 18.5_sort_list.cpp 
// $ ./main.exe 

// Initial contents of list - 
// 1 3 2021 42 2020 0 
// Order after sort() 
// 0 1 3 42 2020 2021 
// Order after sort() with a predicate:
// 2021 2020 42 3 1 0