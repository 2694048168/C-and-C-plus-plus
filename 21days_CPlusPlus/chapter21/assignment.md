## Assignment 作业
- 作业包括测验和练习，前者帮助读者加深对所学知识的理解，后者提供了使用新学知识的机会。
- 请尽量先完成测验和练习题，继续学习下一章前，请务必弄懂这些答案。

### 21.5.1 测验
1. 返回布尔值的一元函数称之为什么？
- 对 operator() 函数运算符重载的类或者结构，称之为函数对象
- 只操作一个参数的函数对象，称之为一元函数；操作两个参数的函数对象，称之为二元函数
- 返回类型为布尔类型的一元函数称之为一元谓词；返回类型为布尔类型的二元函数称之为二元谓词

2. 不修改数据也不返回布尔类型的函数对象有什么用？请通过示例阐述您的观点。
- 可以显示数据或者统计元素个数

3. 函数对象这一术语的定义是什么？
- 在应用程序运行阶段存在的所有实体都是对象，因此结构和类也可用作函数，称之为函数对象。
- 函数也可以通过指针来调用，指针函数，故此指针函数也是函数对象。


### 21.5.2 练习
1. 编写一个一元函数，它可供 std::for_each() 用来显示输入参数的两倍。
- 参考文件 21.6_test_unary_function.cpp

```C++
#include <iostream>
#include <algorithm>
#include <vector>

// a structure as a unary predicate. 
template <typename numberType>
struct Double
{
  int usageCount;
  Double () : usageCount(0) {};

  // unary predicate
  // void operator () (const numberType& element) const
  void operator () (const numberType& element)
  {
    ++usageCount;
    std::cout << element * 2 << ' ';
  }
};

template <typename elementType>
void DisplayContainer(elementType& container)
{
  for (auto element = container.begin(); element != container.end(); ++element)
  {
    std::cout << (*element) << ' ';
  }
  std::cout << std::endl;
}

int main(int argc, char** argv)
{
  std::vector<int> numInVec {42, 24, 66, 99, 4, 1};
  std::cout << "==============================" << std::endl;
  std::cout << "The origin integers: " << std::endl;
  DisplayContainer(numInVec);

  // display the array of integers.
  std::cout << "==============================" << std::endl;
  std::cout << "The double integers: " << std::endl;
  auto result = std::for_each(numInVec.begin(), numInVec.end(),
                Double<int>());

  std::cout << std::endl << "==============================" << std::endl;
  std::cout << "The unary predicate used : " << result.usageCount << std::endl;
  std::cout << "==============================" << std::endl;

  return 0;
}

// $ g++ -o main 21.6_test_unary_function.cpp 
// $ ./main.exe

// ==============================
// The origin integers:
// 42 24 66 99 4 1
// ==============================
// The double integers:
// 84 48 132 198 8 2
// ==============================
// The unary predicate used : 6
// ==============================
```

2. 进一步扩展上述谓词，使其能够记录它被调用的次数。
- 参考文件 21.6_test_unary_function.cpp

3. 编写一个用于降序排序的二元谓词。
- 参考文件 21.7_test_binary_predicate.cpp

```C++
#include <iostream>
#include <vector>
#include <algorithm>

template <typename elementType>
class SortAscending
{
public:
  bool operator () (const elementType& num1, const elementType& num2)
  {
    return (num1 < num2);
  }
};

template <typename elementType>
void DisplayContainer(elementType& container)
{
  for (auto element = container.begin(); element != container.end(); ++element)
  {
    std::cout << (*element) << ' ';
  }
  std::cout << std::endl;
}

int main(int argc, char** argv)
{
  std::vector<int> numInVec;
  // insert sample integers
  for (size_t i = 10; i > 0; --i)
  {
    numInVec.push_back(i * 10);
  }

  std::cout << "==============================" << std::endl;
  std::cout << "The origin integers: " << std::endl;
  DisplayContainer(numInVec);

  std::sort(numInVec.begin(), numInVec.end(), SortAscending<int>());
  std::cout << "==============================" << std::endl;
  std::cout << "The after sorting: " << std::endl;
  DisplayContainer(numInVec);
  std::cout << "==============================" << std::endl;

  std::sort(numInVec.begin(), numInVec.end());
  std::cout << "The after sorting using std::less<> : " << std::endl;
  DisplayContainer(numInVec);
  std::cout << "==============================" << std::endl;
  
  return 0;
}

// $ g++ -o main 21.7_test_binary_predicate.cpp 
// $ ./main.exe

// ==============================
// The origin integers:
// 100 90 80 70 60 50 40 30 20 10
// ==============================
// The after sorting:
// 10 20 30 40 50 60 70 80 90 100
// ==============================
// The after sorting using std::less<> :
// 10 20 30 40 50 60 70 80 90 100
// ==============================
```