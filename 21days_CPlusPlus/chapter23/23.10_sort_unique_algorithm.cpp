#include <iostream>
#include <algorithm>
#include <vector>
#include <string>

// 排序、在有序集合中搜索以及删除重复元素
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
  std::vector<std::string> vecNames {"John", "jack", "sean", "Anna"}; 

  // insert a depicate.
  vecNames.push_back("jack");

  std::cout << "-----------------------------------------" << std::endl;
  std::cout << "The initial contenst of the vector are: " << std::endl;
  DisplayContainer(vecNames);

  std::cout << "-----------------------------------------" << std::endl;
  std::cout << "The sorted vector contains name in the order:" << std::endl;
  std::sort(vecNames.begin(), vecNames.end());
  DisplayContainer(vecNames);

  std::cout << "-----------------------------------------" << std::endl;
  std::cout << "Searching for \"John\" using 'binary_search' :" << std::endl;
  bool elementFound = std::binary_search(vecNames.begin(), vecNames.end(), "John");
  if (elementFound)
  {
    std::cout << "Result: \"John\" was found in the vector." << std::endl;
  }
  else
  {
    std::cout << "Element not found." << std::endl;
  }

  // erase adjacent deplicates.
  // 一定要注意使用 erase 避免容器末尾包含未知的值
  auto newEnd = std::unique(vecNames.begin(), vecNames.end());
  vecNames.erase(newEnd, vecNames.end());

  std::cout << "-----------------------------------------" << std::endl;
  std::cout << "The contents of the vector after using 'unique' : " << std::endl;
  DisplayContainer(vecNames);
  
  return 0;
}

// $ g++ -o main 23.10_sort_unique_algorithm.cpp 
// $ ./main.exe

// -----------------------------------------
// The initial contenst of the vector are:  
// John jack sean Anna jack
// -----------------------------------------
// The sorted vector contains name in the order:
// Anna John jack jack sean 
// -----------------------------------------
// Searching for "John" using 'binary_search' :
// Result: "John" was found in the vector.
// -----------------------------------------
// The contents of the vector after using 'unique' :
// Anna John jack sean
