#include <iostream>
#include <algorithm>
#include <string>
#include <vector>

// 接受两个参数的函数对象称之为二元函数
// 返回类型为布尔类型的二元函数称之为二元谓词
// STL 容器中的元素执行算数运算，常用二元函数
// STL 算法判断，常用二元谓词
// std::unique() 删除相邻重复元素
// std::sort() 排序算法
// std::stable_sort() 排序算法并保持相对顺序
// std::transform() 对两个范围进行操作的算法

class CompareStringNoCase
{
public:
  // binary predicate
  bool operator() (const std::string& str1, const std::string& str2)
  {
    std::string str1LowerCase;

    // assign space.
    str1LowerCase.resize(str1.size());

    // convert every character to the lower case.
    std::transform(str1.begin(), str1.end(), str1LowerCase.begin(), ::tolower);

    std::string str2LowerCase;
    // assign space.
    str2LowerCase.resize(str2.size());
    // convert every character to the lower case.
    std::transform(str2.begin(), str2.end(), str2LowerCase.begin(), ::tolower);

    return (str1LowerCase < str2LowerCase);
  }
};

template <typename T>
void DisplayContainer(const T& container)
{
  for (auto element = container.begin(); element != container.end(); ++element)
  {
    std::cout << *element << std::endl;
  }
  std::cout << std::endl;
}

int main(int argc, char** argv)
{
  // define a vector of string to hold names.
  std::vector<std::string> names;

  // insert some sample names in to the vector.
  names.push_back("jim");
  names.push_back("Jack");
  names.push_back("Sam");
  names.push_back("Anna");

  std::cout << "The names in vector in order of insertion: " << std::endl;
  DisplayContainer(names);
  std::cout << "==============================================" << std::endl;

  std::cout << "Names after sorting using default std::less<> : " << std::endl;
  std::sort(names.begin(), names.end());
  DisplayContainer(names);
  std::cout << "==============================================" << std::endl;

  std::cout << "Names after sorting using binary predicate that ignore case: " << std::endl;
  std::sort(names.begin(), names.end(), CompareStringNoCase());
  DisplayContainer(names);
  std::cout << "==============================================" << std::endl;
  
  return 0;
}

// $ g++ -o main 21.5_binary_predicate.cpp
// $ ./main.exe

// The names in vector in order of insertion:    
// jim
// Jack
// Sam
// Anna

// ==============================================
// Names after sorting using default std::less<> :
// Anna
// Jack
// Sam
// jim

// ==============================================
// Names after sorting using binary predicate that ignore case:   
// Anna
// Jack
// jim
// Sam

// ==============================================