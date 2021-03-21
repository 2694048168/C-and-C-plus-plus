## Assignment 作业
- 作业包括测验和练习，前者帮助读者加深对所学知识的理解，后者提供了使用新学知识的机会。
- 请尽量先完成测验和练习题，继续学习下一章前，请务必弄懂这些答案。

### 13.7.1 测验
1. 您有一个基类对象指针 objBase，要确定它指向的是否是 Derived1 或 Derived2 对象，应使用哪
种类型转换？
- dynamic_cast

2. 假设您有一个指向对象的 const 引用，并试图通过它调用一个您编写的公有成员函数，但编译
器不允许您这样做，因为该函数不是 const 成员。您将修改这个函数还是使用 const_cast?
- 如果允许并能够修改，那就修改
- 是在不允许或者不能修改，在使用的时候就是用 const_cast

3. 判断对错：仅在不能使用 static_cast 时才应使用 reinterpret_cast，这种类型转换是必须和安全的。
- 对

4. 判断对错：优秀的编译器将自动执行很多基于 static_cast 的类型转换，尤其是简单数据类型之
间的转换。
- 对

### 13.7.2 练习
1. 查错：下述代码有何问题？
```C++
void DoSomething(Base* objBase)
{
  Derived* objDer = dynamic_cast <Derived*>(objBase);
  objDer->DerivedClassMethod();
}
```
- 总是需要检查动态转换的结果，是否有效
- 下面修改后正确的

```C++
void DoSomething(Base* objBase)
{
  Derived* objDer = dynamic_cast <Derived*>(objBase);
  if (objDer) // check for validity
    objDer->DerivedClassMethod();
}
```

2. 假设有一个 Fish 指针（ objFish），它指向一个 Tuna 对象：
```C++
Fish* objFish = new Tuna;
Tuna* pTuna = <what cast?>objFish;
```
要让一个 Tuna 指针指向该指针指向的 Tuna 对象， 应使用哪种类型转换？请使用代码证明您的
看法。

- 参考文件 13.2_test_stctic_cast.cpp
```C++
#include <iostream>

// test
class Fish
{
public:
  virtual void Swim()
  {
    std::cout << "Fish swims ini water." << std::endl;
  }

  // base class should always have virtual destructor
  virtual ~Fish() {}
};

class Tuna : public Fish
{
public:
  void Swim()
  {
    std::cout << "Tuna swims real fast in the sea." << std::endl;
  }

  void BecomeDinner()
  {
    std::cout << "Tuna become dinner in Sushi." << std::endl;
  }
};

class Carp : public Fish
{
public:
  void Swim()
  {
    std::cout << "Carp swims real slow in the lake." << std::endl;
  }

  void Talk()
  {
    std::cout << "Carp talked Carp!" << std::endl;
  }
};

void DetectFishType(Fish* objFish)
{
  Tuna* objTuna = dynamic_cast<Tuna*>(objFish);
  // if (objTuna) to check success of cast
  if (objTuna)
  {
    std::cout << "Detected Tuna. Making Tuna dinner: " << std::endl;
    objTuna->BecomeDinner();
  }

  Carp* objCarp = dynamic_cast<Carp*>(objFish);
  // if (objTuna) to check success of cast
  if (objCarp)
  {
    std::cout << "Detected Carp. Making Tuna dinner: " << std::endl;
    objCarp->Talk();
  }

  std::cout << "Verifying type using virtual Fish::Swim: " << std::endl;
  objFish->Swim(); // calling virtual function
}

int main(int argc, char** argv)
{
  Fish* ptrFish = new Tuna;
  // 使用 static_cast 进行类型转换
  Tuna* ptrTuna = static_cast<Tuna*> (ptrFish);

  // Tuna::BecomeDinner will work only using valid Tuna*
  ptrTuna->BecomeDinner();

  // virtual destructor in Fish ensures invocation of ~Tuna()
  delete ptrFish;
  
  return 0;
}

// $ g++ -o main 13.2_test_stctic_cast.cpp 
// $ ./main.exe 
// Tuna become dinner in Sushi
```