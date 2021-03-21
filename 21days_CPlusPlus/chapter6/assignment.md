## Assignment 作业
- 作业包括测验和练习，前者帮助读者加深对所学知识的理解，后者提供了使用新学知识的机会。
- 请尽量先完成测验和练习题，继续学习下一章前，请务必弄懂这些答案。

### 6.7.1 测验
1. 既然不缩进也能通过编译，为何需要缩进语句块，嵌套if语句和嵌套循环？
- 便于阅读和理解源代码
- 代码规范也是很重要的一项技能

2. 使用 goto 可快速解决问题，为何要避免使用呢？
- 使用 goto 导致代码不直观和难以维护
- 建议避免使用 goto

3. 可编写计数器递减的for循环吗？这样的 for 循环时什么样子的？
- 可以的, 如倒序问数据元素
```C++
for (size_t i = length -1; i > 0; --i)
{

}
```

4. 下面的循环有何问题？
```C++
for (int counter = 0; counter == 10; ++counter)
    std::count << counter << " ";
```
- for 循环的条件一次都不满足，循环不会执行 


### 6.7.2 练习
1. 编写一个 for 循环，以倒序方式访问数组的元素。
- 参考测试文件 6.15_test_reverse.cpp
```C++
#include <iostream>

int main(int argc, char** argv)
{
  const int ARRAY_LEG_ONE = 5;

  int numOne[ARRAY_LEG_ONE] = {24, 20, -1, 20, -1};

  for (int index = ARRAY_LEG_ONE - 1; index >= 0; --index)
  {
    std::cout << "numOne[" << index << "] = " << numOne[index] << std::endl;
  }

  std::cout << std::endl;

  return 0;
}

// $ g++ -o main 6.15_test_reverse.cpp   
// $ ./main.exe
// numOne[4] = -1
// numOne[3] = 20
// numOne[2] = -1
// numOne[1] = 20
// numOne[0] = 24
```

2. 编写一个类似程序清单 6.14 的嵌套 for 循环，但是以倒序方式使得两个数组中元素相加。
- 参考测试文件 6.16_test_reverse_add.cpp
```C++
#include <iostream>

int main(int argc, char** argv)
{
  const int ARRAY_LEG_ONE = 3;
  const int ARRAY_LEG_TWO = 2;

  int numOne[ARRAY_LEG_ONE] = {24, 20, -1};
  int numTwo[ARRAY_LEG_TWO] = {20, -1};

  std::cout << "Adding each int in numOne by each in numTwo: " << std::endl;

  for (int i = ARRAY_LEG_ONE - 1; i >= 0; --i)
  {
    for (int j = ARRAY_LEG_TWO - 1; j >= 0; --j)
    {
      std::cout << numOne[i] << " * " << numTwo[j] << " = " << numTwo[j] * numOne[i] << std::endl;
    }
  }

  return 0;
}

// $ g++ -o main 6.16_test_reverse_add.cpp
// $ ./main.exe
// Adding each int in numOne by each in numTwo:
// -1 * -1 = 1
// -1 * 20 = -20
// 20 * -1 = -20
// 20 * 20 = 400
// 24 * -1 = -24
// 24 * 20 = 480
```

3. 编写一个程序，像程序清单 6.16 那样显示斐波纳契数列，但让用户指定每次显示多少个。
- 参考文件 6.17_test_fiboacci.cpp

```C++
#include <iostream>

int main(int argc, char** argv)
{
  std::cout << "This program will calculate Fibonacci Numbers at a time." << std::endl;
  std::cout << "Please enter the numbers of the Fibonacci Numbers: ";
  int numsToCalculate = 0;
  std::cin >> numsToCalculate;

  int fibonacci_one = 0, fibonacci_two = 1;
  char wantMore = '\0';
  std::cout << fibonacci_one << " " << fibonacci_two << " ";

  // 计算斐波纳契数列
  do
  {
    for (size_t i = 0; i < numsToCalculate; ++i)
    {
      std::cout << fibonacci_two + fibonacci_one << " ";
      int fibonacci_two_temp = fibonacci_two;
      fibonacci_two = fibonacci_two + fibonacci_one;
      fibonacci_one = fibonacci_two_temp;
    }
    std::cout << std::endl << "Do you want more numbers (y/n)? ";
    std::cin >> wantMore;
  } while (wantMore == 'y');
    
  std::cout << "Goodbye!" << std::endl;

  return 0;
}

// $ g++ -o main 6.17_test_fiboacci.cpp
// $ ./main

// This program will calculate Fibonacci Numbers at a time.    
// Please enter the numbers of the Fibonacci Numbers: 34
// 0 1 1 2 3 5 8 13 21 34 55 89 144 233 377 610 987 1597 2584 4181 6765 10946 17711 28657 46368 75025 121393 196418 317811 
// 514229 832040 1346269 2178309 3524578 5702887 9227465       
// Do you want more numbers (y/n)? y
// 14930352 24157817 39088169 63245986 102334155 165580141 267914296 433494437 701408733 1134903170 1836311903 -1323752223 
// 512559680 -811192543 -298632863 -1109825406 -1408458269 1776683621 368225352 2144908973 -1781832971 363076002 -1418756969 -1055680967 1820529360 764848393 -1709589543 -944741150 1640636603 695895453 -1958435240 -1262539787 1073992269 -188547518
// Do you want more numbers (y/n)? n
// Goodbye!

// 后面出现的异常数值表明溢出了
// 这也证明了 斐波纳契数列 的强大，累计或者复利的力量强大！！！
```

4. 编写一个 switch-case 结构，指出用户选择的颜色是否出现在彩虹中。请使用枚举常量。
- 参考测试文件 6.18_test_rain_colors.cpp
```C++
#include <iostream>

int main(int argc, char **argv)
{
  enum AvailableColors
  {
    Violet = 0,
    Indigo,
    Blue,
    Green,
    Yellow,
    Orange,
    Red,
    Crimson,
    Beige,
    Brown,
    Peach,
    Pink,
    White,
  };

  std::cout << "Here are the available colors: " << std::endl; 
  std::cout << "============================== " << std::endl; 
  std::cout << "Violet: " << Violet << std::endl; 
  std::cout << "Indigo: " << Indigo << std::endl; 
  std::cout << "Blue: " << Blue << std::endl; 
  std::cout << "Green: " << Green << std::endl; 
  std::cout << "Yellow: " << Yellow << std::endl; 
  std::cout << "Orange: " << Orange << std::endl; 
  std::cout << "Red: " << Red << std::endl; 
  std::cout << "Crimson: " << Crimson << std::endl; 
  std::cout << "Beige: " << Beige << std::endl; 
  std::cout << "Brown: " << Brown << std::endl; 
  std::cout << "Peach: " << Peach << std::endl; 
  std::cout << "Pink: " << Pink << std::endl; 
  std::cout << "White: " << White << std::endl;
  std::cout << "============================== " << std::endl; 

  std::cout << "Please choose one color as Rainbow colors by entering code: "; 
  int userChoose = Blue;
  std::cin >> userChoose;

  // switch 这种执行效果，值得注意！！！
  switch (userChoose)
  {
  case Violet:
  case Indigo:
  case Blue:
  case Green:
  case Yellow:
  case Orange:
  case Red:
    std::cout << "Bingo, your choice is a Rainbow color." << std::endl;
    break;
  
  default:
    std::cout << "The color you chose is not in the rainbow." << std::endl;
    break;
  }

  return 0;
}
```

5. 查错：下面的代码有何错误？
```C++
for (int counter=0; counter=10; ++counter)
    cout << counter << " ";
```

- for 循环中条件判断部分成了赋值，无法判断条件进行循环

6. 查错：下面的代码有何错误？
```C++
int loopCounter = 0;
while(loopCounter <5);
{
  cout << loopCounter << " ";
  loopCounter++;
}
```

- while 后面多了一个 分号 ；导致多了一条空语句，循环无法执行

7. 查错：下面的代码有何错误？
```C++
cout << "Enter a number between 0 and 4" << endl;
int input = 0;
cin >> input;
switch (input)
{
case 0:
case 1:
case 2:
case 3:
case 4:
  cout << "Valid input" << endl;
default:
  cout << "Invalid input" << endl;
}
```

- 缺失 break，导致 default 部分总会执行