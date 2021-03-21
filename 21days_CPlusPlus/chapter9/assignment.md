## Assignment 作业
- 作业包括测验和练习，前者帮助读者加深对所学知识的理解，后者提供了使用新学知识的机会。
- 请尽量先完成测验和练习题，继续学习下一章前，请务必弄懂这些答案。

### 9.15.1 测验
1. 使用 new 创建类实例时，将在声明创建它？
- 在自由存储区
- 使用 new 创建 int 类型变量一样的

2. 我的类包含一个原始指针 int*，指向一个动态分配的 int 数组。请问将 sizeof 用于这个类的对象时，结果是否取决于该动态数组包含的元素数？
- 不
- sizeof 根据声明的数据成员计算类的大小
- 将 sizeof 用于指针时，结果与指向的数据量无关，因此计算的结果时固定的

3. 假设有一个类，其所有成员都是私有的，而且没有友元类和友元函数。请问谁能访问这些成员？
- 除了该类的成员方法之外，在其他任何地方都不能访问

4. 可以在一个类成员方法中调用另一个成员方法吗？
- 可以，通过友元 friend 即可

5. 构造函数适合做什么？
- 构造函数适合做初始化数据成员和资源

6. 析构函数适合做什么？
- 析构函数适合做内存和资源的释放


### 9.15.2 练习
1. 查错：下面的类声明有什么错误？
```C++
Class Human
{
  int age;
  string name;
public: 
  Human() {}
}
```
- 关键字 class 不能大写
- 最后缺少一个分号 ;

2. 练习1 所示类的用户任何访问成员 Human::age ？
```C++
class Human
{
  int age;
  string name;
public: 
  Human() {}
};
```
- 由于 age 数据属于私有属性
- 而且该类没有公有的存取函数 get方法和set方法
- 所有这个类的用户无法访问 age

3. 对练习 1 中类进行修改，在构造函数中使用初始化列表对所有参数进行初始化。
```C++
class Human
{
  int age;
  std::string name;
public: 
  Human() {}
  Human(int& initAge, std::string initName) : age(initAge), name(initName) {}
};
```

4. 编写一个 Circle 类，它根据实例化时提供的半径计算面积和周长。将 PI 包含在一个私有成员常量中，该常量不能再类外访问。
- 参考文件 9.18_test_class.cpp
```C++
#include <iostream>

class CalcCircle
{
public:
  CalcCircle (double radius) : radius(radius), PI(3.14159265) {}

  double CalcCircumference()
  {
    return 2 * PI * radius;
  }

  double CalcArea()
  {
    return PI * radius * radius;
  }

private:
  const double PI;
  double radius;
};

int main()
{
  std::cout << "Please enter the radius of circle: ";
  double radius = 0;
  std::cin >> radius;

  CalcCircle myCircle(radius);
  std::cout << "Circumference fo the circle = " << myCircle.CalcCircumference() << std::endl;
  std::cout << "Area fo the circle = " << myCircle.CalcArea() << std::endl;

  return 0;
}

// $ g++ -o mian 9.18_test_class.cpp 
// $ ./mian.exe
// Please enter the radius of circle: 44
// Circumference fo the circle = 276.46
// Area fo the circle = 6082.12
```