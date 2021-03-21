/**4. 使用构造函数进行类型转换
 * 通过重载构造函数，实现类型转换
 * 为了避免隐式转换，在声明构造函数使用 explicit 关键字
 */

#include <iostream>

class Human
{
private:
  int age;

public:
  // explicit constructor blocks implicit conversions
  explicit Human(int humanAge) : age(humanAge) 
  {

  }
};

void DoSomething(Human person)
{
  std::cout << "Human sent did something work." << std::endl;

  return;
}

int main(int argc, char** argv)
{
  // explicit conversion is OK.
  Human kid(10);
  Human antherKid = Human(11);
  DoSomething(kid);

  // failure: implicit conversion not OK
  // Human anotherKid = 11;
  // DoSomething(10);  // implicit conversion
  
  return 0;
}
