#include <iostream>

/**1. 表明覆盖意图的限定符 override
 * std=C++11 可以使用限定符 override 来核实被覆盖的函数在基类中是否被声明为虚的
 * override 提供强大途径，能够明确表达对基类的虚函数进行覆盖的意图，进而让编译器做如下检查：
 * -- 基类函数是否虚函数
 * -- 基类中相应的虚函数的特征标是否与派生类中被声明的 override 函数完全相同
 * 
 * 下面注释的代码能够说明出现的问题！！！
 */
/*
class Fish
{
public:
  virtual void Swim()
  {
    std::cout << "Fish swims!" << std::endl;
  }
};

class Tuna:public Fish
{
public:
  void Swim() const
  {
    std::cout << "Tuna swims!" << std::endl;
  }
};

class Tuna:public Fish
{
public:
  void Swim() const override // Error: no virtual fn with this sig in Fish
  {
    std::cout << "Tuna swims!" << std::endl; 
  }
};
*/

/**2. 使用 final 来禁止覆盖函数
 * std=C++11 引入限定符 final 
 * 被声明为 final 的类不能作为基类
 * 被声明为 final 的虚函数，不能在派生类中进行覆盖
 * 
 * 下面注释的代码说了这个问题！！！
 */
/*
class Tuna:public Fish
{
public:
  // override Fish::Swim and make this final
  void Swim() override final
  {
    std::cout << "Tuna swims!" << std::endl;
  }
};

class BluefinTuna final:public Tuna
{
public:
  void Swim() // Error: Swim() was final in Tuna, cannot override
  { }
};
*/

/**3. 将复制构造函数声明为虚函数吗？
 * 不能，C++ 不支持将拷贝构造函数声明为虚函数，虽然可以很美好的解决深拷贝问题
 * C++ 不允许使用 虚拷贝构造函数
 * 但是存在一种不错的解决方案，就是定义自己的克隆函数来实现
 * 
 * 下面注释代码可以说明这个问题！！！
 */
/*
// Tuna, Carp and Trout are classes that inherit public from base class Fish
Fish* pFishes[3];
Fishes[0] = new Tuna();
Fishes[1] = new Carp();
Fishes[2] = new Trout();

class Fish
{
public:
  virtual Fish* Clone() const = 0; // pure virtual function
};
class Tuna:public Fish
{
  // ... other members
public:
  Tuna * Clone() const // virtual clone function
  {
    return new Tuna(*this); // return new Tuna that is a copy of this
  }
};
*/

// 虚函数 Clone 模拟了 虚复制构造函数，需要显式调用
class Fish
{
public:
  virtual Fish* Clone() = 0; 
  virtual void Swim() = 0;
  virtual ~Fish() {};
};

class Tuna: public Fish
{
public:
  Fish* Clone() override
  {
    return new Tuna (*this);
  }

  void Swim() override final
  {
    std::cout << "Tuna swims fast in the sea" << std::endl;
  }
};

class BluefinTuna final:public Tuna
{
public:
  Fish* Clone() override
  {
    return new BluefinTuna(*this);
  }

  // Cannot override Tuna::Swim as it is "final" in Tuna
};

class Carp final: public Fish
{
  Fish* Clone() override
  {
    return new Carp(*this);
  }
  void Swim() override final
  {
    std::cout << "Carp swims slow in the lake" << std::endl;
  }
};


int main(int argc, char** argv)
{
  const int ARRAY_SIZE = 4;

  Fish* myFishes[ARRAY_SIZE] = {NULL};
  myFishes[0] = new Tuna();
  myFishes[1] = new Carp();
  myFishes[2] = new BluefinTuna();
  myFishes[3] = new Carp();

  Fish* myNewFishes[ARRAY_SIZE];
  for (int index = 0; index < ARRAY_SIZE; ++index)
  {
    myNewFishes[index] = myFishes[index]->Clone();
  }

  // invoke a virtual method to check
  for (int index = 0; index < ARRAY_SIZE; ++index)
  {
    myNewFishes[index]->Swim();
  }

  // memory cleanup
  for (int index = 0; index < ARRAY_SIZE; ++index)
  {
    delete myFishes[index];
    delete myNewFishes[index];
  }

  return 0;
}

// $ g++ -o main 11.9_override_final_virtual_copy_con.cpp      
// $ ./main.exe 
// Tuna swims fast in the sea
// Carp swims slow in the lake
// Tuna swims fast in the sea
// Carp swims slow in the lake