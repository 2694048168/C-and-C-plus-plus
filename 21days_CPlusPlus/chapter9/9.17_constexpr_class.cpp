#include <iostream>

// 将 constexpr 用于类和对象，改善 C++ 性能的方式
class Human
{
private:
  int age;

public:
  constexpr Human(int humanAge) : age(humanAge)  
  {

  }

  constexpr int GetAge() const 
  {
    return age;
  }
};


int main(int argc, char** argv)
{
  constexpr Human somePerson(15);
  const int hisAge = somePerson.GetAge();

  // Human anotherPerson()  is not constant expression.
  Human anotherPerson(45);

  return 0 ;
}