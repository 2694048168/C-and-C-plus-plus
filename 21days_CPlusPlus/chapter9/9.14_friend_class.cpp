#include <iostream>
#include <string>

// 不能从类的外部访问类的私有成员和方法
// 但是这条规则不适合友元类和友元函数
class Human
{
private:
  // 声明外部函数为该类的友元函数，允许其访问私有数据以及方法
  friend void DisplayAge(const Human& person);

  // 声明外部类 Utility 为该类的友元类，允许其访问私有数据以及方法
  friend class Utility;

  std::string name;
  int age;

public:
  Human(std::string humanName, int humanAge)
  {
    name = humanName;
    age = humanAge;
  }
};

void DisplayAge(const Human& person)
{
  std::cout << person.age << std::endl;
}

class Utility
{
public:
  static void DisplayAge(const Human& person)
  {
    std::cout << person.age << std::endl;
  }
};


int main(int argc, char** argv)
{
  Human firstMan("Adam", 25);

  std::cout << "Accessing private menber age via friend function: ";
  DisplayAge(firstMan);

  std::cout << "===============================================" << std::endl;
  std::cout << "Accessing private menber age via friend class: ";
  Utility::DisplayAge(firstMan);

  return 0;
}


// $ g++ -o main 9.14_friend_class.cpp 
// $ ./main.exe
// Accessing private menber age via friend function: 25
// ===============================================
// Accessing private menber age via friend class: 25