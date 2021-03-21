## Assignment 作业
- 作业包括测验和练习，前者帮助读者加深对所学知识的理解，后者提供了使用新学知识的机会。
- 请尽量先完成测验和练习题，继续学习下一章前，请务必弄懂这些答案。

### 7.6.1 测验
1. 在函数原型中声明的变量的作用域是什么？
- 本函数内部

2. 传递给下述函数的值有何特征？
```C++
int Func(int &someNUmber);
```
- 通过 引用 进行函数参数的传递
- 形参和实参之间不是直接拷贝复制值，而是通过指针指向

3. 自己调用自己的函数称之为什么？
- 递归函数

4. 我声明了两个函数，他们的名称和返回类型相同，但是参数列表不同，这个称之为什么？
- 函数重载 function overload

5. 栈指针指向栈的顶部，中间，还是底部？
- 栈指针指向栈的**顶部**

### 7.6.2 练习
1. 编写两个重载函数，他们分别使用下述公式计算球和圆柱体的体积
- 参考测试文件 7.12_test_overload.cpp
```C++
Volume of sphere = (4 * Pi * radius * radius * radius) / 3
Volume of a cylinder = Pi * radius * radius * height
```

```C++
#include <iostream>

const double PI = 3.14159265;

// Volume of sphere = (4 * Pi * radius * radius * radius) / 3
// Volume of a cylinder = Pi * radius * radius * height

// Function Overloading
double ComputeVolume(double radius)
{
  return (4 * PI * radius * radius * radius) / 3;
}

double ComputeVolume(double radius, double height)
{
  return PI * radius * radius * height;
}


int main(int argc, char** argv)
{
  std::cout << "Enter s for sphere, c for Cylinder: ";
  char userSelection = 's';
  std::cin >> userSelection;

  std::cout << "Enter radius: ";
  double radius = 0;
  std::cin >> radius;

  if (userSelection == 'c')
  {
    std::cout << "Enter height: ";
    double height = 0;
    std::cin >> height;

   // Invoke overloaded variant of Area for Cyclinder 
   std::cout << "Area of cylinder is: " << ComputeVolume (radius, height) << std::endl;
  }
  else
    std::cout << "Area of sphere is: " << ComputeVolume (radius) << std::endl;

  return 0;
}

```

2. 编写一个函数，将一个 double 数组作为参数。
```C++
void DisplayElement(double array[], int length)
{
  for (int i = 0; i < length; ++i>)
  {
    std::cout << "Element of array: " << array[i] << " ";
  }
  std::cout << std::endl;
}
```

3. 查错：下述代码有什么错误？
```C++
#include <iostream>
using namespace std;
const double Pi = 3.1416;
void Area(double radius, double result)
{
  result = Pi * radius * radius;
}
int main()
{
  cout << "Enter radius: ";
  double radius = 0;
  cin >> radius;
  double areaFetched = 0;
  Area(radius, areaFetched);
  cout << "The area is: " << areaFetched << endl;
  return 0;
}
```

- 未使用 引用 进行参数传递，不能获取计算后的值

4. 查错：下述代码有什么错误？
```C++
double Area(double Pi = 3.14, double radius);
// 修改后
double Area(double radius, double Pi = 3.14);
```

- 参数默认初始化，后面的一定要全部初始化

5. 编写一个返回类型为 void 的函数，在提供了半径情况下，能够帮助调用者计算圆的周长和面积。
- 参考文件 7.13_test_default_para.cpp
```C++
#include <iostream>

const double PI = 3.14159265;

// Function Declaration
void ComputeArea(const double radius, double &result_area);
void ComputeCircumference(const double radius, double &result_circumference);

int main(int argc, char** argv)
{
  std::cout << "Enter radius: ";
  double radius = 0;
  double result_area = 0;
  double result_circumference = 0;
  std::cin >> radius;

  // Call function to compute.
  ComputeArea(radius, result_area);
  ComputeCircumference(radius, result_circumference);

  std::cout << "Area is: " << result_area << std::endl;
  std::cout << "Circumference is: " << result_circumference << std::endl;

  return 0;
}

// Function Definitions
void ComputeArea(const double radius, double &result_area)
{
  result_area =  PI * radius * radius;
}

void ComputeCircumference(const double radius, double &result_circumference)
{
  result_circumference = 2 * PI * radius;
}

// $ g++ -o main 7.13_test_default_para.cpp 
// admin@weili /d/VSCode/workspace/21days_CPlusPlus/chapter7
// $ ./main.exe 
// Enter radius: 4
// Area is: 50.2655
// Circumference is: 25.1327
```