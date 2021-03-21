#include <iostream>
#include <unordered_set>

// STL 散列集合实现 std::unordered_set ; std::unordered_multiset;
// 基于散列实现，使用散列函数来计算排序索引
// C++11 
// unordered_set 的一个重要特征是，有一个负责确定排列顺序的散列函数：
// unordered_set<int>::hasher HFn = usetInt.hash_function();

template <typename T>
void DisplayContents(const T& container)
{
  for (auto element = container.cbegin(); element != container.cend(); ++element)
  {
    std::cout << *element << ' ';
  }
  std::cout << std::endl;

  std::cout << "Number of elements, size() = " << container.size() << std::endl;
  std::cout << "Bucket count = " << container.bucket_count() << std::endl;
  std::cout << "Max load factor = " << container.max_load_factor() << std::endl;
  std::cout << "Load factor: " << container.load_factor() << std::endl << std::endl;
}

int main(int argc, char** argv)
{
  std::unordered_set<int> usetInt {1, 2, 2021, 2020, -1, -9, 99, 300};
  DisplayContents(usetInt);
  usetInt.insert(999);
  DisplayContents(usetInt);

  std::cout << "Please enter int you want to check for existence in set: ";
  int input = 0;
  std::cin >> input;
  
  auto elementFound = usetInt.find(input);
  if (elementFound != usetInt.end())
  {
    std::cout << *elementFound << " found in set." << std::endl;
  }
  else
  {
    std::cout << input << " not available in set." << std::endl;
  }
  
  return 0;
}

// $ g++ -o main 19.5_hash_unordered.cpp 
// $ ./main.exe 

// 300 99 -1 -9 2020 2021 2 1 
// Number of elements, size() = 8
// Bucket count = 11
// Max load factor = 1
// Load factor: 0.727273

// 999 300 99 -1 -9 2020 2021 2 1
// Number of elements, size() = 9
// Bucket count = 11
// Max load factor = 1
// Load factor: 0.818182

// Please enter int you want to check for existence in set: -300  
// -300 not available in set.

// $ ./main.exe 
// 300 99 -1 -9 2020 2021 2 1 
// Number of elements, size() = 8
// Bucket count = 11
// Max load factor = 1
// Load factor: 0.727273

// 999 300 99 -1 -9 2020 2021 2 1
// Number of elements, size() = 9
// Bucket count = 11
// Max load factor = 1
// Load factor: 0.818182

// Please enter int you want to check for existence in set: 2021  
// 2021 found in set.