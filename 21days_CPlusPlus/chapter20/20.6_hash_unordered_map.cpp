/**基于散列表的 STL 键值对容器
 * C++11 <unordered_map>
 * std::unordered_map<keyType>;
 * std::unordered_multimap<keyType>;
 * 
 * 哈希函数/散列函数，提高查找的效率，数据结构里面的知识
 * 利用哈希函数计算键的索引，可能发生冲突
 */

#include <iostream>
#include <string>
#include <unordered_map>

template <typename T1, typename T2>
void DisplayUnorderedMap(std::unordered_map<T1, T2>& container)
{
  std::cout << "Unoredered Map contains: " << std::endl;
  for (auto element = container.cbegin(); element != container.cend(); ++element)
  {
    std::cout << element->first << " -> " << element->second << std::endl;
  }

  std::cout << "Number of pairs, size(): " << container.size() << std::endl;
  std::cout << "Bucket count = " << container.bucket_count() << std::endl;
  std::cout << "Current load factor: " << container.load_factor() << std::endl;
  std::cout << "Max load factor: " << container.max_load_factor() << std::endl;
}

int main(int argc, char** argv)
{
  std::unordered_map<int, std::string> umapIntToStr;

  umapIntToStr.insert(std::make_pair(1, "One"));
  umapIntToStr.insert(std::make_pair(45, "Forty Five"));
  umapIntToStr.insert(std::make_pair(1001, "Thousand One"));
  umapIntToStr.insert(std::make_pair(-2, "Minus Two"));
  umapIntToStr.insert(std::make_pair(-1000, "Minus One Thousand"));
  umapIntToStr.insert(std::make_pair(100, "One Hundred"));
  umapIntToStr.insert(std::make_pair(12, "Twelve"));
  umapIntToStr.insert(std::make_pair(-100, "Minus One Hundred"));

  DisplayUnorderedMap<int, std::string> (umapIntToStr);

  std::cout << "Inserting one more element" << std::endl;
  umapIntToStr.insert(std::make_pair(300, "Three Hundred"));
  DisplayUnorderedMap<int, std::string>(umapIntToStr);

  std::cout << "Please enter key to find for: ";
  int key = 0;
  std::cin >> key;

  auto element = umapIntToStr.find(key);
  if (element != umapIntToStr.end())
  {
    std::cout << "Found, key pairs with value " << element->second << std::endl;
  }
  else
  {
    std::cout << "Key has no corresponding pair value." << std::endl;
  }
  
  return 0;
}

// $ g++ -o main 20.6_hash_unordered_map.cpp 
// $ ./main.exe 

// Unoredered Map contains:   
// -100 -> Minus One Hundred  
// 12 -> Twelve
// 100 -> One Hundred
// -1000 -> Minus One Thousand
// 1001 -> Thousand One       
// 45 -> Forty Five
// -2 -> Minus Two
// 1 -> One
// Number of pairs, size(): 8
// Bucket count = 13
// Current load factor: 0.615385
// Max load factor: 1
// Inserting one more element
// Unoredered Map contains:
// -100 -> Minus One Hundred
// 12 -> Twelve
// 100 -> One Hundred
// -1000 -> Minus One Thousand
// 1001 -> Thousand One
// 45 -> Forty Five
// 300 -> Three Hundred
// -2 -> Minus Two
// 1 -> One
// Number of pairs, size(): 9
// Bucket count = 13
// Current load factor: 0.692308
// Max load factor: 1
// Please enter key to find for: 300
// Found, key pairs with value Three Hundred