## Assignment 作业
- 作业包括测验和练习，前者帮助读者加深对所学知识的理解，后者提供了使用新学知识的机会。
- 请尽量先完成测验和练习题，继续学习下一章前，请务必弄懂这些答案。

### 22.11.1 测验
1. 编译器如何确定 lambda 表达式的起始位置？
- 捕获列表 []

2. 如何将状态变量传递给 lambda 表达式？
- 通过捕获列表，进行状态变量参数的处理
- 是否需要修改 mutable；是否需要**引用方式**传递参数

3. 如何指定 lambda 表达式的返回类型？
- 尾置返回类型方式
- -> returnType


### 22.11.2 练习
1. 编写一个可用的二元谓词的 lambda 表达式，帮助将元素按降序排列。
- 参考文件 22.6_test_binary_predicate_sort.cpp

```C++
#include <iostream>
#include <algorithm>
#include <vector>

template <typename T>
void DisplayContainer(const T &container)
{
  for (auto element = container.begin(); element != container.end(); ++element)
  {
    std::cout << *element << ' ';
  }
  std::cout << std::endl;
}

int main(int argc, char** argv)
{
  std::vector<int> vecNumbers {25, -5, 122, 2021, 2020};
  DisplayContainer(vecNumbers);

  std::sort(vecNumbers.begin(), vecNumbers.end());
  DisplayContainer(vecNumbers);

  std::sort(vecNumbers.begin(), vecNumbers.end(),
            [] (int num1, int num2) { return num1 > num2;});
  DisplayContainer(vecNumbers);            

  std::cout << "================================================" << std::endl;
  std::cout << "Number you wish to add to all elements: ";
  int numcontainer = 0;
  std::cin >> numcontainer;
  std::for_each(vecNumbers.begin(), vecNumbers.end(),
                [=] (int& element) {element += numcontainer;});
  DisplayContainer(vecNumbers);                
  
  return 0;
}

// $ g++ -o main 22.6_test_binary_predicate_sort.cpp 
// $ ./main.exe

// 25 -5 122 2021 2020 
// -5 25 122 2020 2021
// 2021 2020 122 25 -5
// ================================================
// Number you wish to add to all elements: 5
// 2026 2025 127 30 0 
```

2. 编写一个这样的 lambda 表达式，即用于 for_each() 时，给 vector 等容器中的元素加上用户指定的值。
- 参考文件 22.6_test_binary_predicate_sort.cpp
