#include <iostream>
#include <string>

class Human
{
private:
  std::string name;
  int age;
public:
  // constructor overload
  Human(std::string humanName, int humenAge)
  {
    name = humanName;
    age = humenAge;
    std::cout << "Overloaded constructor creates " << name << " of " << age << "years" << std::endl;
  }

  // constructor overload
  // Human(std::string humanName, int humenAge = 25)
  // {
  //   name = humanName;
  //   age = humenAge;
  //   std::cout << "Overloaded constructor creates " << name << " of " << age << "years" << std::endl;
  // }

  void IntroduceSelf()
  {
    std::cout << "I am " + name << " and am " << age << " years old" << std::endl;
  }
};


int main(int argc, char** argv)
{
  // using overloaded constructor
  Human firstWoman("Eve", 23);
  Human firstMan("Adam", 33);

  firstWoman.IntroduceSelf();
  firstMan.IntroduceSelf();

  return 0;
}

// $ g++ -o main 9.5_default_constructor.cpp 
// $ ./main.exe 

// Overloaded constructor creates Eve of 23years
// Overloaded constructor creates Adam of 33years
// I am Eve and am 23 years old
// I am Adam and am 33 years old