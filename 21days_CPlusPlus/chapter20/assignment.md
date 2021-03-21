## Assignment 作业
- 作业包括测验和练习，前者帮助读者加深对所学知识的理解，后者提供了使用新学知识的机会。
- 请尽量先完成测验和练习题，继续学习下一章前，请务必弄懂这些答案。

### 20.7.1 测验
1. 使用 map<int> 声明整型 map 时，排序标准将由那个函数提供？
- 默认使用 std::less<> 提供, 使用 运算符< 来比较两个整数，返回 bool 值
- 也可以自定义 排序谓词 来覆盖默认的

2. 在 multimap 中，重复的值以什么方式出现？
- multimap 在插入值时会进行排序，重复的值一定在一起，彼此相邻
- 使用成员函数 multiset.count(value) 返回重复的值的个数

3. map 和 multimap 的那个成员函数指出容器包含多少个元素？
- map.size() ; multimap.size() 该成员函数返回容器的包含元素个数

4. 在 map 中的什么地方可以找到重复的值？
- map 中不能储存重复的值
- 只有 multimap 才能储存重复的值


### 20.7.2 练习
1. 编写一个应用程序来实现电话簿，他不要求人名唯一的，应该选择那种容器？写出容器的定义。
- 选择可以包含重复元素的关联容器，std::multimap;

```C++
std::multimap<std::string, std::string> mapNamesToNumbers;
```

2. 下面是电话簿应用程序中的一个 map 的定义，其中 WordProperty 是一个结构, 请定义一个二元谓词 fProdicate，用于帮助该 map 根据 WordProperty 键包含的 string 属性对元素进行排序。

```C++
map <wordProperty, string, fPredicate> mapWordDefineition;

struct WordProperty
{
  string word;
  bool isLatinBass;
};

// 二元谓词
struct fPredicate
{
  bool operator < (const WordProperty& lsh, const WordProperty& rsh )
  {
    return (lsh.word < rsh.word);
  }
};
```

3. 通过一个简单程序演示 map 不接受重复的元素，而 multimap 接受。
- 参考文件 20.7_test_repeat_element.cpp

```C++
#include <iostream>
#include <map>

template <typename T>
void DisplayContent (const T& container)
{
  for (auto element = container.begin(); element != container.end(); ++element)
  {
    // std::cout << element->first << " ---> " << element->second << std::endl;
    std::cout << (*element).first << " ---> " << (*element).second << std::endl;
  }
  std::cout << std::endl;
}

int main(int argc, char** argv)
{
  std::multimap<int, int> mmapIntegers;
  mmapIntegers.insert(std::make_pair(5, 42));
  mmapIntegers.insert(std::make_pair(5, 42));
  mmapIntegers.insert(std::make_pair(5, 42));
  mmapIntegers.insert(std::make_pair(5, 42));
  mmapIntegers.insert(std::make_pair(5, 42));

  std::map<int, int> mapIntegers;
  mapIntegers.insert(std::make_pair(5, 24));
  mapIntegers.insert(std::make_pair(5, 24));
  mapIntegers.insert(std::make_pair(5, 24));
  mapIntegers.insert(std::make_pair(5, 24));
  mapIntegers.insert(std::make_pair(5, 24));

  std::cout << "Displaying the contents of the multimap: " << std::endl;
  DisplayContent(mmapIntegers);
  std::cout << "The size of the multimap is: " << mmapIntegers.size() << std::endl;
  std::cout << "=========================================" << std::endl;

  std::cout << "Displaying the contents of the map: " << std::endl;
  DisplayContent(mapIntegers);
  std::cout << "The size of the map is: " << mapIntegers.size() << std::endl;
  std::cout << "=========================================" << std::endl;
  
  return 0;
}

// $ g++ -o main 20.7_test_repeat_element.cpp 
// $ ./main.exe

// Displaying the contents of the multimap: 
// 5 ---> 42
// 5 ---> 42
// 5 ---> 42
// 5 ---> 42
// 5 ---> 42

// The size of the multimap is: 5
// =========================================
// Displaying the contents of the map:
// 5 ---> 24

// The size of the map is: 1
// =========================================
```