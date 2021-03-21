#include <iostream>
#include <map>

// 实例化 map 和 multimap 类
// 指定键类型和值类型，以及可选的谓词
// std::map<keyType, valueType, predicate=std::less<keyType>> mapObj;
// std::multimap<keyType, valueType, predicate=std::less<keyType>> mmapObj;

template <typename keyType>
struct ReverseSort
{
  bool operator () (const keyType& key1, const keyType& key2) 
  {
    return (key1 > key2);
  }
};

int main(int argc, char** argv)
{
  // map and multimap key of type int to value of type string.
  std::map<int, std::string> mapIntToString;
  std::multimap<int, std::string> mmapIntToString;

  // map and multimap constructored as a copy of another.
  std::map<int, std::string> mapIntToStr(mapIntToString);
  std::multimap<int, std::string> mmapIntToStr(mmapIntToString);

  // map and multimap constructord given a part of another map or multimap.
  std::map<int, std::string> mapIntToStr2(mapIntToString.cbegin(), mapIntToString.cend());
  std::multimap<int, std::string> mmapIntToStr2(mmapIntToString.cbegin(), mmapIntToString.cend());

  // map and multimap with a predicate that inverse sort order.
  std::map<int, std::string, ReverseSort<int>> mapIntToStr3(mapIntToString.cbegin(), mapIntToString.cend());
  std::multimap<int, std::string, ReverseSort<int>> mmapIntToStr3(mmapIntToString.cbegin(), mmapIntToString.cend());

  return 0;
}
