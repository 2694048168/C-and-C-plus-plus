## Assignment 作业
- 作业包括测验和练习，前者帮助读者加深对所学知识的理解，后者提供了使用新学知识的机会。
- 请尽量先完成测验和练习题，继续学习下一章前，请务必弄懂这些答案。

### 14.7.1 测验
1. 什么是多次包含防范(inclusion guard)？
- 这是一个预编译器结构，用于避免多次或递归包含头文件

2. 如果使用参数 20 调用下面的宏,结果是多少?
```C++
#define SPLIT(x) x / 5
// 建议修改, 虽然在这里不影响结果,但是规范还是要的
// #define SPLIT(x) (x / 5)
```
- 4

3. 如果使用 10 + 10 调用测验 2 中的 SPLIT 宏,结果是多少?
- 10 + 10 / 5 = 12
- 没有达到预期的结果

4. 任何修改 SPLIT 宏以免得到错误的结果?
```C++
#define SPLIT(x) (x / 5)
```

### 14.7.2 练习
1. 编写一个将两个数相乘的宏.
- 参考文件 14.10_test.cpp

```C++
#define MULTIPLE(a, b) (a * b)
```

2. 编写一个模板,实现练习 4 中宏的功能
- 参考文件 14.10_test.cpp

```C++
template <typename T>
T QUARTER(const T value)
{
  return ((value) / 4);
}
```

3. 实现模板函数 swap, 交换两个变量的值.
- 参考文件 14.10_test.cpp

```C++
#include <iostream>

// test 练习 1
#define MULTIPLE(a, b) ((a) * (b))

// 使用模板实现该宏功能
// #define QUARTER(x) (x / 4)
template <typename T>
T QUARTER(const T value)
{
  return ((value) / 4);
}

// 实现模板函数 swap 交换两个变量的值
template <typename T1, typename T2=T1>
void swap(T1& value1, T2& value2)
{
  T1 temp = value1;
  value1 = value2;
  value2 = temp;
}


int main(int argc, char** argv)
{
  // test 1.
  float num1 = 3, num2 = 8;
  std::cout << "The result of multiple " << MULTIPLE(num1, num2) << std::endl;
  std::cout << "=========================" << std::endl;

  // test 2.
  double value = 16;
  std::cout << "The result of QUARTER: " << QUARTER(value) << std::endl;

  // test swap templeate function  
  std::cout << "=========================" << std::endl;
  int value1 = 2020, value2 = 2021;
  std::cout << "Before swap, the value of number1 and number2: " 
            << value1 << " " << value2 << std::endl; 
  // swap template function
  swap(value1, value2);
  std::cout << "After swap, the value of number1 and number2: " 
            << value1 << " " << value2 << std::endl;
  
  return 0;
}

// $ g++ -o main 14.10_test.cpp 
// $ ./main.exe
// The result of multiple 24
// =========================
// The result of QUARTER: 4
// =========================
// Before swap, the value of number1 and number2: 2020 2021
// After swap, the value of number1 and number2: 2021 2020
```

4. 查错: 您将如何改进下面的宏使得计算输入值的 1/4 ?
```C++
// #define QUARTER(x) (x / 4)
// 修改后
#define QUARTER(x) ((x) / 4)
```

5. 编写一个简单的模板类, 储存两个数组, 数组的类型是通过模板参数列表指定的. 数组包含 10 个元素,模板类应包含存取函数,可用于操作数组元素.
- 参考文件 14.11_test_template.cpp

```C++
#ifndef TEST_TEMPLATE
#define TEST_TEMPLATE

template <typename T1, typename T2>
class TwoArray
{
public:
  T1& GetT1Element(int index) 
  {
    return T1 arrayOne[index];
  }

  T2& GetT2Element(int index) 
  {
    return T2 arrayOne[index];
  }

private:
  T1 arrayOne [10];
  T2 arrayTwo [10];
};

#endif  // TEST_TEMPLATE
```

6. 编写模板函数 Display(), 可以使用不同数量和类型的参数调用,并将所有的参数都显示出来
- 参考文件 14.12_test_param_num.cpp

```C++
#include <iostream>

// 参数数量可变的模板
// ... C++ 中使用省略号来表明参数数量可变，可以接受任意参数数量，而且参数类型任意
// C++14 标准才支持，提供新的运算符 sizeof...() 来计算可变参数数量模板传递了多少个参数
void Display()
{
}

template <typename First, typename ...Last> void Display(First value1, Last... valueN)
{
  std::cout << value1 << std::endl;
  Display(valueN...);
}

int main(int argc, char **argv)
{
  Display('a');
  Display(3.14);
  Display('a', 3.14);
  Display('z', 3.14567, "The power of variadic templates!");

  return 0;
}

// $ g++ -std=c++14 -o main 14.12_test_param_num.cpp      
// $ ./main.exe
// a
// 3.14
// a
// 3.14
// z
// 3.14567
// The power of variadic templates!
```