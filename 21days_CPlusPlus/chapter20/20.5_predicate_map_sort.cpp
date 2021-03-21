/**提供自定义的排序谓词
 * 实例化 map/multimap 时候，提供排序谓词参数，保证 map 能够正常工作
 * 默认排序谓词 std::less<keyType> , 该谓词使用 < 运算符来比较两个对象
 * 要提供不同的排序标准，可以编写一个二元谓词：实现 operator() 的类和结构
 * template <typename KeyType>
 * struct Predicate
 * {
 *   bool operator() (const KeyType& key1, const KeyType& key2)
 *   {
 *     // you sort priority logic here.
 *   }
 * };
 * 
 * 使用谓词来定制 map 的行为，表明 键 可以是任何类型的
 * 谓词实现了 运算符() 的结构，这种作为函数对象被称之为函数对象
 */

#include <iostream>
#include <string>
#include <map>
#include <algorithm>

template <typename keyType>
void DisplayContents (const keyType& container)
{
  for (auto element = container.cbegin(); element != container.cend(); ++element)
  {
    // std::cout << element->first << " -> " << element->second << std::endl;
    std::cout << (*element).first << " -> " << (*element).second << std::endl;
  }
  std::cout << std::endl;
}

struct PredicateIgoreCase
{
  bool operator() (const std::string& str1, const std::string& str2) const
  {
    std::string str1NoCase(str1);
    std::string str2NoCase(str2);
    // 全部转换为小写，忽略大小写
    std::transform(str1.begin(), str1.end(), str1NoCase.begin(), ::tolower);
    std::transform(str2.begin(), str2.end(), str2NoCase.begin(), ::tolower);

    return (str1NoCase < str2NoCase);
  }
};

// 简化变量类型，更加易于阅读
typedef std::map<std::string, std::string> DIR_WITH_CASE;
typedef std::map<std::string, std::string, PredicateIgoreCase> DIR_NOCASE;

int main(int argc, char** argv)
{
  // case-sensitive directorycase of string-key plays no role.
  DIR_WITH_CASE dirWithCase;

  dirWithCase.insert(std::make_pair("John", "2345764"));
  dirWithCase.insert(std::make_pair("JOHN", "2345764"));
  dirWithCase.insert(std::make_pair("Sara", "42367236"));
  dirWithCase.insert(std::make_pair("Jack", "32435348"));

  std::cout << "Displaying contents of the case-sensitive map: " << std::endl;
  DisplayContents(dirWithCase);

  // case-insensitive mapcase of string-key affects insertion & search.
  DIR_NOCASE dirNoCase(dirWithCase.begin(), dirWithCase.end());

  std::cout << "Displaying contents of the case-insenstive map: " << std::endl;
  DisplayContents(dirNoCase);

  // search for a name in the two maps and display result.
  std::cout << "Please enter a name to search" << std::endl << "> ";
  std::string name;
  std::cin >> name;

  auto pairWithCase = dirWithCase.find(name);
  if (pairWithCase != dirWithCase.end())
  {
    std::cout << "Num in case-sens. dir: " << pairWithCase->second << std::endl;
  }
  else
  {
    std::cout << "Num not found in case-sensitive dir" << std::endl;
  }
  
  auto pairNoCase = dirNoCase.find(name);
  if (pairWithCase != dirNoCase.end())
  {
    std::cout << "Num found in CI dir: " << pairNoCase->second << std::endl;
  }
  else
  {
    std::cout << "Num not found in case-insensitive directory" << std::endl;
  }
  
  return 0;
}

// $ g++ -o main 20.5_predicate_map_sort.cpp 
// $ ./main.exe 

// Displaying contents of the case-sensitive map:  
// JOHN -> 2345764
// Jack -> 32435348
// John -> 2345764
// Sara -> 42367236

// Displaying contents of the case-insenstive map: 
// Jack -> 32435348
// JOHN -> 2345764
// Sara -> 42367236

// Please enter a name to search
// > jack
// Num not found in case-sensitive dir
// Num found in CI dir: 32435348
