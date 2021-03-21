#include <iostream>
#include <map>
#include <string>

// 删除 键值对 元素
// 成员函数 map.erase(key); 

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
  std::multimap<int, std::string> mmapIntToString;

  // insert a pair using function make_pair.
  mmapIntToString.insert(std::make_pair(3, "Three"));
  mmapIntToString.insert(std::make_pair(45, "Forty Five"));
  mmapIntToString.insert(std::make_pair(-1, "Minus One"));
  mmapIntToString.insert(std::make_pair(1000, "Thousand"));

  // insert duplicates into the multimap.
  mmapIntToString.insert(std::make_pair(-1, "Minus One"));
  mmapIntToString.insert(std::make_pair(1000, "Thousand"));

  std::cout << "The multimap contains " << mmapIntToString.size() << " key-value pairs." << std::endl;
  DisplayContents(mmapIntToString);

  // erasing an element with key as -1 from the multimap.
  auto numPairsErased = mmapIntToString.erase(-1);
  std::cout << "Erased " << numPairsErased << " pairs with -1 as key." << std::endl;

  // erase an element given an iterator from the multimap.
  auto pair = mmapIntToString.find(45);
  if (pair != mmapIntToString.end())
  {
    mmapIntToString.erase(pair);
    std::cout << "Erased a pair with 45 as key using an iterator." << std::endl;
  }

  // erase a range from the multimap
  std::cout << "Erasing the range of pairs with 10000 as key." << std::endl;
  mmapIntToString.erase(mmapIntToString.lower_bound(1000), mmapIntToString.upper_bound(1000));

  std::cout << "The multimap now contains " << mmapIntToString.size() << " key-value pairs." << std::endl;
  DisplayContents(mmapIntToString);

  return 0;
}

// $ g++ -o main 20.4_erase_map.cpp 
// $ ./main.exe 

// The multimap contains 6 key-value pairs.       
// -1 -> Minus One
// -1 -> Minus One
// 3 -> Three
// 45 -> Forty Five
// 1000 -> Thousand
// 1000 -> Thousand

// Erased 2 pairs with -1 as key.
// Erased a pair with 45 as key using an iterator.
// Erasing the range of pairs with 10000 as key.  
// The multimap now contains 1 key-value pairs.   
// 3 -> Three