#include <iostream>
#include <map>
#include <string>

// 查找 键值对 元素
// 成员函数 map.find(); 
// multimap.find() 由相同的键，返回第一个，使得迭代器进行递增，访问相邻的值。
// map<int, string>::const_iterator pairFound = map.find(key);
// if (pairFound != map.end())

template <typename T>
void DisplayContents (const T& container)
{
  for (auto element = container.begin(); element != container.end(); ++element)
  {
    std::cout << element->first << " -> " << element->second << std::endl;
  }
  std::cout << std::endl;
}

int main(int argc, char** argv)
{
  std::map<int, std::string> mapIntToString;

  // insert a pair using function make_pair.
  mapIntToString.insert(std::make_pair(3, "Three"));
  mapIntToString.insert(std::make_pair(45, "Forty Five"));
  mapIntToString.insert(std::make_pair(-1, "Minus One"));
  mapIntToString.insert(std::make_pair(1000, "Thousand"));

  std::cout << "The multimap contains " << mapIntToString.size() << " key-value pairs." << std::endl;
  // 默认排序 升序方式输出
  DisplayContents(mapIntToString);

  std::cout << "Please enter the key you wish to find: ";
  int key = 0;
  std::cin >> key;

  auto pairFound = mapIntToString.find(key);
  if (pairFound != mapIntToString.end())
  {
    std::cout << "Key " << pairFound->first << " points to value: " << pairFound->second << std::endl;
  }
  else
  {
    std::cout << "sorry, pair with key " << key << " not in map." << std::endl;
  }

  std::cout << "==============================================" << std::endl;
  std::multimap<int, std::string> mmapIntToString (mapIntToString.begin(), mapIntToString.end());
  mmapIntToString.insert(std::make_pair(45, "Forty Five"));
  mmapIntToString.insert(std::make_pair(45, "Forty Five"));
  mmapIntToString.insert(std::make_pair(45, "Forty Five"));

  std::cout << "The multimap contains " << mmapIntToString.size() << " key-value pairs." << std::endl;
  DisplayContents(mmapIntToString);

  auto mpairFound = mmapIntToString.find(45);
  // check if find() succeeded.
  if (mpairFound != mmapIntToString.end())
  {
    // find the number of pairs that have the same supplied key.
    size_t numPairInMap = mmapIntToString.count(45);

    for (size_t i = 0; i < numPairInMap; ++i)
    {
      std::cout << "Key: " << mpairFound->first << ", value [" << i << "] = " << mpairFound->second << std::endl;
      // 查找相邻的下一个
      ++mpairFound;
    }
  }
  else
  {
    std::cout << "Element not found in the multimap." << std::endl;
  }

  return 0;
}

// $ g++ -o main 20.3_find_map.cpp 
// $ ./main.exe

// The multimap contains 4 key-value pairs.
// -1 -> Minus One
// 3 -> Three
// 45 -> Forty Five
// 1000 -> Thousand

// Please enter the key you wish to find: 3
// Key 3 points to value: Three
// ==============================================
// The multimap contains 7 key-value pairs.
// -1 -> Minus One
// 3 -> Three
// 45 -> Forty Five
// 45 -> Forty Five
// 45 -> Forty Five
// 45 -> Forty Five
// 1000 -> Thousand

// Key: 45, value [0] = Forty Five
// Key: 45, value [1] = Forty Five
// Key: 45, value [2] = Forty Five
// Key: 45, value [3] = Forty Five