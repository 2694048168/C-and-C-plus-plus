## Assignment 作业
- 作业包括测验和练习，前者帮助读者加深对所学知识的理解，后者提供了使用新学知识的机会。
- 请尽量先完成测验和练习题，继续学习下一章前，请务必弄懂这些答案。

### 8.8.1 测验
1. 为何不能将 const 引用赋值非 const 引用？
```C++
int original = 30;
const int& constRef = original;
constRef = 40; // Not allowed: constRef can’ t change value in original
int& ref2 = constRef; // Not allowed: ref2 is not const
const int& constRef2 = constRef; // OK
```

- C++ 比起 C 语言而言，最大改变就是安全，C 语言就不是很安全，可以暗度陈仓

2. new 和 delete 是函数吗？
- 不是；是运算符

3. 指针变量包含的值有何特征？
- 指针本质就是内存地址，当然指针本身也需要内存地址储存，所有有了指针的指针

4. 要访问指针指向的数据，应该使用声明运算符？
- 解引用运算符 *


### 8.8.2 练习
1. 下面的语句显示是什么？
```C++
int number = 3;
int* pNum1 = &number;
_*pNum1 = 20;
int* pNum2 = pNum1;
number *= 2;
cout << *pNum2;
```
- 3 <—— 20 <—— 20 * 2 = 40
- 40

2. 下面三个重载的函数有何相同和不同之处？
```C++
int DoSomething(int num1, int num2);
int DoSomething(int& num1, int& num2);
int DoSomething(int* pNum1, int* pNum2);
```

- 第一个，实参将被拷贝复制给形参 
- 第二个，不会复制，形参是实参的引用
- 第三个，使用指针，指针可能为NULL或者无效，需要检验有效性

3. 要让练习 1 中第 3 行的赋值非法，应该如何修改第 1 行中的 pNum1 的声明？
```C++
int number = 3;
const int* pNum1 = &number;
_*pNum1 = 20; // Not allowed
```

4. 查错：下述代码有什么错误？
```C++
#include <iostream>
using namespace std;
int main()
{
  int *pointToAnInt = new int;
  pointToAnInt = 9;
  // *pointToAnInt = 9;
  cout << "The value at pointToAnInt: " << *pointToAnInt;
  delete pointToAnInt;
  return 0;
}
```
- 注释的一句是正确的修改

5. 查错：下述代码有什么错误？
```C++
#include <iostream>
using namespace std;
int main()
{
  int pointToAnInt = new int;
  int* pNumberCopy = pointToAnInt;
  *pNumberCopy = 30;
  cout << *pointToAnInt;
  delete pNumberCopy;
  // delete pointToAnInt;
  return 0;
}
```

- 注释掉 即可
- 30