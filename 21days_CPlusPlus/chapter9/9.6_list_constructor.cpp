#include <iostream>
#include <string>

class Human
{
private:
  std::string name;
  int age;
public:
  // 可以将 constexpr 用于构造函数，定义为常量表达式，有助于提高性能
  // two parameterst to initialize menbers age and name
  Human(std::string humanName = "Adam", int humenAge = 24)
       :name(humanName), age(humenAge) // list initialization
  {
    std::cout << "Constructed a human called " << name << " , " << age << " years old." << std::endl;
  }
};


int main(int argc, char** argv)
{
  Human adam;

  Human firstMan("Eve", 23);

  return 0;
}

// $ g++ -o main 9.6_list_constructor.cpp 
// $ ./main.exe

// Constructed a human called Adam , 24 years old.
// Constructed a human called Eve , 23 years old.