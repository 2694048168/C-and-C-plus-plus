#include <iostream>
#include <string>

class Human
{
private:
  std::string name;
  int age;
public:
  // constructor overload
  Human()
  {
    age = 0;  // initialized to ensure no junk value
    std::cout << "Default constructor: name and age not set." << std::endl;
  }

  Human(std::string humanName, int humenAge)
  {
    name = humanName;
    age = humenAge;
    std::cout << "Overloaded constructor creates " << name << " of " << age << "years" << std::endl;
  }
};


int main(int argc, char** argv)
{
  // using default constructor
  Human firstMan;

  // using overloaded constructor
  Human firstWoman("Eve", 23);

  return 0;
}

// $ g++ -o main 9.4_overload_constructor.cpp 
// $ ./main.exe 
// Default constructor: name and age not set.
// Overloaded constructor creates Eve of 23years