#include <iostream>
#include <string>

class Human
{
public:
  std::string name;
  int age = 0;

  void IntroduceSelf()
  {
    std::cout << "I am " + name << " and am " << age << " years old" << std::endl;
  }
};


int main(int argc, char** argv)
{
  // An object of class Human with attribute name as "Adam"
  Human firstMan;
  firstMan.name = "Adam";
  firstMan.age = 30;

  // An object of  class Human attribute name as "Eve"
  Human firstWoman;
  firstMan.name = "Eve";
  firstWoman.age = 26;

  firstMan.IntroduceSelf();
  firstWoman.IntroduceSelf(); 

  return 0;
}
