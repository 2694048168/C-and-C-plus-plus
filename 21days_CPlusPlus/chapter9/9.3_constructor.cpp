#include <iostream>
#include <string>

class Human
{
private:
  std::string name;
  int age;
public:
  // constructor of class
  Human()
  {
    age = 1;  // initialization
    std::cout << "Constructed an instance of class Human" << std::endl;
  }

  void SetName(std::string hunmanName)
  {
    name = hunmanName;
  }

  void SetAge(int humanAge)
  {
    age = humanAge;
  }

  void IntroduceSelf()
  {
    std::cout << "I am " + name << " and am " << age << " years old" << std::endl;
  }
};


int main(int argc, char** argv)
{
  Human firstMan;
  firstMan.SetName("Eve");
  firstMan.SetAge(23);
  firstMan.IntroduceSelf();

  return 0;
}

// $ g++ -o main 9.3_constructor.cpp 
// $ ./main.exe 
// Constructed an instance of class Human
// I am Eve and am 23 years old