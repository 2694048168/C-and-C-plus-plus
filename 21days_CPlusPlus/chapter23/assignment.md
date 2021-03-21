## Assignment 作业
- 作业包括测验和练习，前者帮助读者加深对所学知识的理解，后者提供了使用新学知识的机会。
- 请尽量先完成测验和练习题，继续学习下一章前，请务必弄懂这些答案。

### 23.6.1 测验
1. 要将 list 中满足特定条件的元素删除，应使用 std::remove_if() 还是 list::remove_if()？
- 使用 std::list::remove_if()
- 因为确保了指向 list 元素的现有迭代器依然有效

2. 假设有一个包含 ContactItem 对象的 list，在没有显式指定二元谓词时， 函数 list::sort() 将如何对这些元素排序？
- using default functor: std::less<>
- 即使用运算符 < 对集合中的元素对象进行排序处理

3. STL 算法 generate() 将调用函数 generator()？
- 对指定范围的每一个元素调用一次
- container.size()

4. std::transform() 与 std::for_each() 之间的区别何在？
- std::for_each() 接受一元谓词，并返回用于包含状态信息的函数对象；
- std::transform() 接受一元谓词或者二元谓词，可用处理两个输入范围的重载版本


### 23.6.2 练习
1. 编写一个二元谓词，接受字符串作为输入参数，并根据不区分大小写的比较结果返回一个值。
- 参考文件 23.13_test_predicate.cpp

```C++
#include <iostream>
#include <algorithm>
#include <string>

// binary predicate
struct CaseInsensitiveCompare
{
  bool operator () (const std::string& str1, const std::string& str2)
  {
    std::string str1Copy (str1);
    std::string str2Copy (str2);

    std::transform(str1Copy.begin(), str1Copy.end(), str1Copy.begin(), tolower);
    std::transform(str2Copy.begin(), str2Copy.end(), str2Copy.begin(), tolower);

    return (str1Copy < str2Copy);
  }
};

int main(int argc, char** argv)
{
  std::string str1 {"LiWei"};
  std::string str2 {"Jxufe Software"};

  std::cout << "The first string: " << str1 << std::endl;
  std::cout << "The second string: " << str2 << std::endl;

  std::cout << "-------------------------------" << std::endl;
  std::cout << std::boolalpha << (str1 < str2) << std::endl;
  std::cout << std::boolalpha << (str2 < str1) << std::endl;

  std::cout << "-------------------------------" << std::endl;
  auto element = CaseInsensitiveCompare();
  std::cout << std::boolalpha << element(str1, str2) << std::endl;
  std::cout << std::boolalpha << element(str2, str1) << std::endl;

  return 0;
}
// TEST failure
// $ g++ -o main 23.13_test_predicate.cpp 
// $ ./main.exe

// The first string: LiWei
// The second string: Jxufe Software
// -------------------------------
// false
// true
// -------------------------------
// false
// true
```

2. 演示 STL 算法（如 copy() )如何使用迭代器实现其功能——复制两个类型不同的容器储存的序列，而无需知道目标集合的特征。
- 参考文件 23.14_test_copy_achieve.cpp

```C++
#include <iostream>
#include <algorithm>
#include <list>
#include <string>
#include <vector>

template <typename T>
void DisplayContainer(const T &container)
{
  for (auto element = container.cbegin(); element != container.cend(); ++element)
  {
    std::cout << *element << ' ';
  }
  std::cout << std::endl;
}

int main(int argc, char **argv)
{
  std::list<std::string> listNames;
  listNames.push_back("Jack");
  listNames.push_back("John");
  listNames.push_back("Anna");
  listNames.push_back("Skate");

  std::list<std::string>::const_iterator iListNames;
  for (iListNames = listNames.begin(); iListNames != listNames.end(); ++iListNames)
  {
    std::cout << *iListNames << ' ';
  }

  std::cout << std::endl << "-----------------------" << std::endl;
  std::vector<std::string> vecNames(4);
  std::copy(listNames.begin(), listNames.end(), vecNames.begin());

  std::vector<std::string>::const_iterator iNames;
  for (iNames = vecNames.begin(); iNames != vecNames.end(); ++iNames)
  {
    std::cout << *iNames << ' ';
  }

  return 0;
}

// $ g++ -o main 23.14_test_copy_achieve.cpp
// $ ./main.exe

// Jack John Anna Skate 
// -----------------------
// Jack John Anna Skate
```

3. 您正在编写一个应用程序，它按星星在地平线上升起的顺序记录它们的特点。在天文学中，星球的大小很重要，其升起和落下的相对顺序亦如此。如果根据星星的大小对这个集合进行排序，应使用 std::sort() 还是 std::stable_sort()?
- 按照题述要求，同时结合 std::sort() and std::stable_sort() 特点区别
- 应该选择 std::stable_sort()
