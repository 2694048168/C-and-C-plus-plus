## Assignment 作业
- 作业包括测验和练习，前者帮助读者加深对所学知识的理解，后者提供了使用新学知识的机会。
- 请尽量先完成测验和练习题，继续学习下一章前，请务必弄懂这些答案。

### 11.8.1 测验
1. 假设您要模拟形状——圆和三角形，并要求每个形状都必须实现函数Area()和Print()。您该怎么办？
- 声明抽象基类ABC shape类，并在shape类中将函数Area()和Print()声明为纯虚函数，
- 从而要求派生类 Circle和Triangle必须实现这些函数

2. 编译器为每一个类都创建虚函数表吗？
- 编译器只为包含 虚函数 的类创建虚函数表 VFT

3. 我编写了一个 Fish 类，它有两个公有方法，一个纯虚函数和几个成员属性。这个类是抽象基类吗？
- 是抽象基类，不能实例化
- 只要类至少包含一个纯虚函数，编译器就认为是抽象基类ABC


### 11.8.2 练习
1. 创建一个继承层次结构，实现检测 1 中 Circle 和 Trangle 类。
- 参考文件 11.10_test_ABC.cpp
```C++
#include <iostream>

// 抽象基类 ABC
class Shape
{
public:
  virtual double Area() = 0;
  virtual void Print() = 0;
};

// 派生类，必须实现抽象基类中的纯虚函数
class Circle : public Shape
{
public:
  // 利用构造函数进行必要参数的初始化
  Circle(double radius, const double PI) : radius(radius), PI(PI)
  {

  }

  double Area() override
  {
    return PI * radius * radius;
  }

  void Print() override
  {
    std::cout << "This is a Ciecle." << std::endl;
  }

private:
  double radius;
  const double PI = 314159265;
};

// 派生类，必须实现抽象基类中的纯虚函数
class Triangle : public Shape
{
public:
  // 利用构造函数进行必要参数的初始化
  Triangle(const double bottom, const double heigth) : bottom(bottom), heigth(heigth)
  {

  }

  double Area() override
  {
    return 0.5 * bottom * heigth;
  }

  void Print() override
  {
    std::cout << "This is a Triangle." << std::endl;
  }

private:
  const double bottom;
  const double heigth;
};

int main(int argc, char** argv)
{
  double radius = 0;
  double bottom = 0;
  double heigth = 0;
  double PI = 3.1415;

  std::cout << "Please enter the radius of circle: ";
  std::cin >> radius;
  std::cout << "Please enter the value of  PI: ";
  std::cin >> PI;
  std::cout << "Please enter the value of bottom for triangle: ";
  std::cin >> bottom;
  std::cout << "Please enter the value of  height for triangle: ";
  std::cin >> heigth;

  Circle myCircle(radius, PI);
  Triangle myTriangle(bottom, heigth);

  myCircle.Print();
  std::cout << "The area of circle: " << myCircle.Area() << std::endl;

  myTriangle.Print();
  std::cout << "The area of triangle: " << myTriangle.Area() << std::endl;

  return 0;
}

// $ g++ -o main 11.10_test_ABC.cpp
// $ ./main.exe
// Please enter the radius of circle: 3
// Please enter the value of  PI: 3.14159265
// Please enter the value of bottom for triangle: 3
// Please enter the value of  height for triangle: 4
// This is a Ciecle.
// The area of circle: 28.2743
// This is a Triangle.        
// The area of triangle: 6
```

2. 查错：下面的代码有何问题？
```C++
class Vehicle
{
public:
  Vehicle() {}
  ~Vehicle(){}
  // virtual ~Vehicle(){}
};
class Car: public Vehicle
{
public:
  Car() {}
  ~Car() {}
};
```
- 基类缺少虚析构函数，只有默认析构函数
- 注释哪一行才是正确的修改

3. 给定练习 2 所示的（错误）代码，像下面这样创建并销毁 Car 实例时，将按什么样的顺序执行构造函数和析构函数？
```C++
Vehicle* pMyRacer = new Car;
delete pMyRacer;
```

- 参考文件：11.11.test_virtual_destructor.cpp
- 派生类实例化 Car，释放对象资源
- 构造函数：先构造基类的默认构造函数，然后再构造派生类的默认构造函数
- 析构函数：只析构了基类的析构函数，资源和内存泄漏等重大问题

```C++
#include <iostream>

class Vehicle
{
public:
  Vehicle() 
  {
    std::cout << "Base Class Vehicle constructor." << std::endl;
  }
  // ~Vehicle()
  // {
  //   std::cout << "Base Class Vehicle destructor." << std::endl;
  // }
  
  virtual ~Vehicle()
  {
    std::cout << "Base Class Vehicle destructor." << std::endl;
  }
};

class Car: public Vehicle
{
public:
  Car() 
  {
    std::cout << "Derive Class Car constructor." << std::endl;
  }
  ~Car() 
  {
    std::cout << "Derive Class Car destructor." << std::endl;
  }
};


int main(int argc, char** argv)
{
  Vehicle* pMyRacer = new Car;
  delete pMyRacer;
  
  return 0;
}

// $ g++ -o main 11.11.test_virtual_destructor.cpp 
// $ ./main.exe 
// Base Class Vehicle constructor.
// Derive Class Car constructor.  
// Base Class Vehicle destructor. 

// 将基类析构函数修改为虚析构函数结果如下：

// $ g++ -o main 11.11.test_virtual_destructor.cpp 
// $ ./main.exe
// Base Class Vehicle constructor.
// Derive Class Car constructor.  
// Derive Class Car destructor.   
// Base Class Vehicle destructor.
```