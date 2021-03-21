## Assignment 作业
- 作业包括测验和练习，前者帮助读者加深对所学知识的理解，后者提供了使用新学知识的机会。
- 请尽量先完成测验和练习题，继续学习下一章前，请务必弄懂这些答案。

### 12.10.1 测验
1. 可用像下面这样，编写两个版本的下标运算符，一个的返回值类型 const，另一个为非 const 吗？
```C++
const Type& operator [] (int index);
Type& operator [] (int index);

// 修改后这样写，意思就变了，C++ 编译器支持这样的 const 函数和 非 const 函数
Type& operator [] (int index); const
Type& operator [] (int index); 
```
- 不可以
- 目的不同，可用有 const 版本和非 const 版本的函数

2. 可以将复制构造函数或者复制赋值运算符声明为私有的吗？
- 可以
- 导致禁止赋值和复制，例如设计模式里面的单例模式

3. 给 Date 类实现移动构造函数和移动赋值运算符有意义吗？
- 没有意义
- 只有动态分配内存等资源的才会导致赋值运算符和复制构造函数进行不必要的内存分配和释放，在这种情况下实现移动构造函数和移动赋值运算符才有意义


### 12.10.2 练习
1. 为 Date 类编写一个转换运算符，将其储存的日期转换为整数。
- 参考文件 12.10_test_conver.cpp
```C++
#include <iostream>

// 转换运算符
class Date
{
public:
  // constructor using list initialization
  Date(int inputMonth, int inputDay, int inputYear)
      : month(inputMonth), day(inputDay), year(inputYear)
  {

  }

  // 使用 explicit 要求使用强制类型转换来确认转换意图
  explicit operator int()
  {
    return (year * 365) + (month * 30) + day;
  }

private:
  int day, month, year;
};

int main(int argc, char** argv)
{
  Date hoilday (3, 24, 2021);
  // 要求程序员使用强制类型转换来确认转换意图
  std::cout << "hoilday is: " << (int)hoilday << " days."<< std::endl;
  
  return 0;
}

// $ g++ -o main 12.10_test_conver.cpp 
// $ ./main.exe 
// hoilday is: 737779 days.
```

2. DynIntegers 类 以 int* 私有成员的方式封装了一个动态分配的数组，请给它编写移动构造函数和移动赋值运算符。
- 参考文件 12.11_test_move.cpp
```C++
#include <iostream>

// 用于高性能编程的移动构造函数和移动赋值运算符
class DynIntegers
{
public:
  // default constructor.
  // 参考文件 12.8_move_constructor_assignment.cpp

  // move constructor.
  DynIntegers(DynIntegers&& moveSrc)
  {
    std::cout << "Move constructor moves: " << moveSrc.arrayNums << std::endl;
    if (moveSrc.arrayNums != NULL)
    {
      arrayNums = moveSrc.arrayNums;  // take ownershiop i.e. 'move'
      moveSrc.arrayNums = NULL; // free move source
    }
  }

  // move assignment operator
  DynIntegers& operator = (DynIntegers&& moveSrc)
  {
    std::cout << "Move assignment operator moves: " << moveSrc.arrayNums << std::endl;
    if ((moveSrc.arrayNums != NULL) && (this != &moveSrc))
    {
      delete [] arrayNums; // release own arrayNums

      arrayNums = moveSrc.arrayNums;  // take ownershiop i.e. 'move'
      moveSrc.arrayNums = NULL;  // free move source
    }

    return *this;
  }

  // copy constructor
  // 参考文件 12.8_move_constructor_assignment.cpp

  // copy assignment operator.
  // 参考文件 12.8_move_constructor_assignment.cpp

  // destructor
  ~DynIntegers()
  {
    // if (arrayNums != NULL)
    if (!arrayNums)
    {
      delete [] arrayNums;
    }
  }

private:
  int* arrayNums;
};

// 移动 避免不必要的复制和内存分配，节省处理时间，提高性能
int main(int argc, char** argv)
{
  // 参考文件 12.8_move_constructor_assignment.cpp
  return 0;
}
```