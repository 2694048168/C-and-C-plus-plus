#include <iostream>

class Human
{
private:
  int age = 0;

public:
  void SetAge(int inputAge)
  {
    age = inputAge;
  }

  // Human lies about his / her age (if over 30)
  int GetAge()
  {
    if (age > 30)
    {
      return (age - 2);
    }
    else
    {
      return age;
    }
  }
};


int main(int argc, char** argv)
{
  Human firstMan;
  firstMan.SetAge(35);

  Human firstWoman;
  firstWoman.SetAge(22);

  // 使用 句点运算符 .  访问成员
  // 指针运算符 -> 访问成员 等价于 使用间接运算符 （*） 来获取对象，再使用 句点运算符来访问
  // Human* firstMan = new Human();
  // (*firstMan).IntroductionSelf();
  // firstMan->IntroductionSelf();
  std::cout << "Age of firstMan " << firstMan.GetAge() << std::endl;
  std::cout << "Age of firstWoman " << firstWoman.GetAge() << std::endl;

  return 0;
}