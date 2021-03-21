#include <iostream>

/**基类初始化——向基类传递参数
 * 如果基类包含重载的构造函数，需要在实例化时提供参数，创建派生类任何实例化基类呢？
 * 解决方法是使用列表初始化，并通过派生类的构造函数调用合适的基类构造函数
 * 
 * 为了最大限度的提高安全性，对于派生类不需要访问的基类属性，使用 protected 权限
 */

class Fish
{
public:
  // overload constructor
  Fish(bool isFreshWater) : isFreshWateFish(isFreshWater)
  {

  }

  void Swim()
  {
    if (isFreshWateFish)
    {
      std::cout << "Swims in lake." << std::endl;
    }
    else
    {
      std::cout << "Swims in sea." << std::endl;
    }
  }

protected:
  // accessible only to derived classes
  bool isFreshWateFish;
};

// 公有继承
class Tuna : public Fish
{
public:
  // constructor initializes base
  Tuna() : Fish(false)
  {

  }
};

// 公有继承
class Carp : public Fish
{
public:
  // constructor initializes base
  Carp() : Fish(true)
  {

  }
};


int main(int argc, char** argv)
{
  Carp myLunch;
  Tuna myDinner;

  std::cout << "About my food: " << std::endl;

  std::cout << "Lunch: ";
  myLunch.Swim();

  std::cout << "Dinner: ";
  myDinner.Swim();

  return 0;
}

// $ g++ -o main 10.3_list_base_class.cpp 
// $ ./main.exe 
// About my food:       
// Lunch: Swims in lake.
// Dinner: Swims in sea.