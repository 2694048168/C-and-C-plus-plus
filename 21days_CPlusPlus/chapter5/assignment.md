## Assignment 作业
- 作业包括测验和练习，前者帮助读者加深对所学知识的理解，后者提供了使用新学知识的机会。
- 请尽量先完成测验和练习题，继续学习下一章前，请务必弄懂这些答案。

### 5.6.1 测验
1. 编写将两个数相除的应用程序时，将变量声明为那种数据类型更合适，int 还是 float？
- 建议使用 float，因为 int 可能会损失精度，达不到用户需求

2. 32/7 的结果是多少？
- 4 编译器认为整数相除只能向下取整

3. 32.0/7 的结果是多少？
- 4.571 编译器认定为浮点数运算

4. sizeof 时函数吗？
- sizeof 不是函数，而是一个运算符，特殊的运算符，而且不能进行运算符重载 

5. 我需要将一个数翻倍，再加上 5，然后再翻倍，下面的代码是否正确？
```C++
int result = number << 1 + 5 << 1;
// 修改为
int result = ( ( (number << 1) + 5) << 1);
```
- 加法 优先级高于 移位运算

6. 如果两个操作数的值都为 true，对其执行 XOR 运算的结果是什么？
- 异或运算 XOR，相同为假，不同为真
- 结果为 false

### 5.6.2 练习
1. 使用括号改善测验题 5 中的代码，使其更加清晰。
- 参考测试文件 5.6_test_priority.cpp
```C++
#include <iostream>

int main(int argc, char** argv)
{
  int number = 6;
  int result_false = number << 1 + 5 << 1;
  // 修改为
  int result_true = ( ( (number << 1) + 5) << 1);

  std::cout << "False: " << result_false << std::endl;
  std::cout << "True: " << result_true << std::endl;

  return 0;
}

// $ g++ -o main 5.6_test_priority.cpp 
// $ ./main.exe
// False: 768
// True: 34
```

2. 下述代码导致 result 的值为多少？
```C++
int result = number << 1 + 5 << 1;
```
- 首先计算优先级高的加法 1 + 5 = 6
- 再计算移位运算，左结合性，先计算左边的移位运算，再计算右边的移位运算
- （ number * 2^6 ）* 2
- 例如 number = 6，and result = 768

3. 编写一个程序，让用户输入两个布尔值，并显示对其执行各种按位运算的结果？
- 参考文件 5.7_test_bitr.cpp

```C++
#include <iostream>
#include <bitset>

int main(int argc, char** argv)
{
  std::cout << "Enter bool value ( 0 or 1 ): ";
  bool inputNum = false;
  std::cin >> inputNum;

  std::cout << "Enter another bool value ( 0 or 1 ): ";
  bool inpuValue = false;
  std::cin >> inpuValue;

  std::cout << "==============" << std::endl;

  bool bitwiseNOT = (~inputNum);
  std::cout << "Logical NOT ~ " << std::endl;
  std::cout << "~" << inputNum << " = " << bitwiseNOT << std::endl;
  std::cout << "==============" << std::endl;

  std::cout << "Logical AND & with inpuValue" << std::endl;
  bool bitwiseAND = (inpuValue & inputNum);
  std::cout << "inpuValue & inputNum = " << bitwiseAND << std::endl;
  std::cout << "==============" << std::endl;

  std::cout << "Logical OR | with inpuValue" << std::endl;
  bool bitwiseOR = (inpuValue | inputNum);
  std::cout << "inpuValue | inputNum = " << bitwiseOR << std::endl;
  std::cout << "==============" << std::endl;

  std::cout << "Logical XOR & with inpuValue" << std::endl;
  bool bitwiseXOR = (inpuValue & inputNum);
  std::cout << "inpuValue ^ inputNum = " << bitwiseXOR << std::endl;

  return 0;
}

// $ g++ -o main 5.7_test_bitr.cpp 
// $ ./main.exe
// Enter bool value ( 0 or 1 ): 1
// Enter another bool value ( 0 or 1 ): 0
// ==============
// Logical NOT ~
// ~1 = 1
// ==============
// Logical AND & with inpuValue
// inpuValue & inputNum = 0
// ==============
// Logical OR | with inpuValue
// inpuValue | inputNum = 1
// ==============
// Logical XOR & with inpuValue
// inpuValue ^ inputNum = 0
```