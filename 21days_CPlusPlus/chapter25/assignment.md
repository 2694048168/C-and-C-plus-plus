## Assignment 作业
- 作业包括测验和练习，前者帮助读者加深对所学知识的理解，后者提供了使用新学知识的机会。
- 请尽量先完成测验和练习题，继续学习下一章前，请务必弄懂这些答案。

### 25.6.1 测验
1. bitset 能否扩展其内部缓冲区以储存可变的元素数？
- 不可以
- std::bitset 在编译阶段就决定了长度，不可扩展 

2. 为什么 bitset 不属于 STL 容器类
- 因为 std::bitset 不支持动态调整长度
- 因为 std::bitset 不支持迭代器

3. 您会使用 std::vecetor 来储存位数在编译阶段就知道的固定位数吗？
- 不会
- 此情况下，最适合选择是 std::bitset<n>


### 25.6.2 练习
1. 创建一个长 4 位的 bitset 对象，并使用一个数字来初始化它，然后显式结果并将其与另一个 bitset 对象相加（注意：bitset 不支持语法 bitsetA = bitsetX + bitsetY）。
- 参考文件 25.5_test_bitset.cpp

```C++
#include <iostream>
#include <bitset>
#include <string>

int main(int argc, char** argv)
{
  // 4 bits initialized to 1010
  std::bitset<4> fourBits1 (10);
  std::cout << "Initial contents of fourBits: " << fourBits1 << std::endl;

  // 4 bits initialized to 0010
  std::bitset<4> fourBits2 (2);
  std::cout << "Initial contents of fourBits: " << fourBits2 << std::endl;

  // add bitset.
  std::bitset<4> addResult(fourBits1.to_ulong() + fourBits2.to_ulong());
  std::cout << "Initial contents of add result: " << addResult << std::endl;

  return 0;
}

// $ g++ -o main 25.5_test_bitset.cpp 
// $ ./main.exe 

// Initial contents of fourBits: 1010  
// Initial contents of fourBits: 0010  
// Initial contents of add result: 1100
```

2. 请演示如何将 bitset 对象中的位取反。
- 参考文件 25.6_test_flip.cpp

```C++
#include <iostream>
#include <bitset>
#include <vector>

template <typename T>
void DisplayContainer(const T& container)
{
  for (auto element = container.cbegin(); element != container.cend(); ++element)
  {
    std::cout << *element << ' ';
  }
  std::cout << std::endl;
}

int main(int argc, char** argv)
{
  // 4 bits initialized to 1010
  std::bitset<4> fourBits1 (10);
  std::cout << "Initial contents of fourBits: " << fourBits1 << std::endl;
  std::cout << "-----------------------------------" << std::endl;
  std::cout << "Initial contents of fourBits with flip: " << fourBits1.flip() << std::endl;

  // 4 bits initialized to 0010
  std::vector<bool> fourBits2 {false, false, true, false};
  std::cout << "Initial contents of fourBits: " << std::endl;
  DisplayContainer(fourBits2);
  std::cout << "-----------------------------------" << std::endl;
  fourBits2.flip();
  std::cout << "Initial contents of fourBits with flip: " << std::endl;
  DisplayContainer(fourBits2);

  return 0;
}

// $ g++ -o main 25.6_test_flip.cpp 
// $ ./main.exe 

// Initial contents of fourBits: 1010
// -----------------------------------
// Initial contents of fourBits with flip: 0101
// Initial contents of fourBits: 
// 0 0 1 0 
// -----------------------------------
// Initial contents of fourBits with flip:
// 1 1 0 1
```