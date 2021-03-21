#include <iostream>
#include <map>
#include <string>

// 插入 键值对 元素
// std::map.insert(std::make_pair(key, value));
// std::multimap.insert(std::pair<int, std::string>(key, value));
// std::map[key] = value;

// type-define the map and multimap definition for easy readability.
typedef std::map<int, std::string> MAP_INT_STRING;
typedef std::multimap<int, std::string> MMAP_INT_STRING;

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
  MAP_INT_STRING mapIntToString;

  // insert key-value pairs into the map using value_type.
  mapIntToString.insert(MAP_INT_STRING::value_type(3, "Three"));

  // insert a pair using function make_pair.
  mapIntToString.insert(std::make_pair(-1, "Minus One"));

  // insert a pair object directly.
  mapIntToString.insert(std::pair<int, std::string>(1000, "One Thousand"));

  // using an array-like syntax for inserting key-value pairs.
  mapIntToString[1000000] = "One Million";

  // diaplay
  std::cout << "The map contains " << mapIntToString.size() << " key-value pairs." << std::endl;
  // 默认排序 升序方式输出
  DisplayContents(mapIntToString);

  // instantiate a multimap that is a copy of a map.
  MMAP_INT_STRING mmapIntToString (mapIntToString.cbegin(), mapIntToString.cend());

  // the insert function works the same way for multimap too.
  // a multimap can store duplicates - insert a duplicate.
  mmapIntToString.insert(std::make_pair(1000, "Thousand"));

  std::cout << "==============================================" << std::endl;
  std::cout << "The multimap contains " << mmapIntToString.size() << " key-value pairs." << std::endl;
  DisplayContents(mmapIntToString);

  // the multimap can return number of pairs with same key.
  std::cout << "===============================================" << std::endl;
  std::cout << "The number of pairs in the multimap with 1000 as their key: " 
            << mmapIntToString.count(1000) << std::endl;

  return 0;
}

// $ g++ -o main 20.2_insert_map.cpp 
// $ ./main.exe 

// The map contains 4 key-value pairs.
// -1 -> Minus One
// 3 -> Three
// 1000 -> One Thousand
// 1000000 -> One Million

// ==============================================
// The multimap contains 5 key-value pairs.
// -1 -> Minus One
// 3 -> Three
// 1000 -> One Thousand
// 1000 -> Thousand
// 1000000 -> One Million

// ===============================================
// The number of pairs in the multimap with 1000 as their key: 2
