#include <iostream>

class Mammal
{
public:
  void FeedBabyMilk()
  {
    std::cout << "Mammal1: Baby says glug!" << std::endl;
  }
};

class Reptile
{
public:
  void SpitVenom()
  {
    std::cout << "Reptile: shoo enemy! Spits venom!" << std::endl; 
  }
};

class Bird
{
public:
  void LayEggs()
  {
    std::cout << "Bird: Laid my eggs, am lighter now!" << std::endl;
  }
};

// 多继承
// 使用 关键字 final 来禁止类的继承
// 被声明为 final 的类不能作为基类
// class Platypus final : public Mammal, public Bird, public Reptile {};
// final 关键字用于成员函数控制多态行为
class Platypus : public Mammal, public Bird, public Reptile
{
public:
  void Swim()
  {
    std::cout << "Platypus: Voila, I can swim!" << std::endl;
  }
};


int main(int argc, char** argv)
{
  Platypus realFreak;
  realFreak.LayEggs();
  realFreak.FeedBabyMilk();
  realFreak.SpitVenom();
  realFreak.Swim();

  return 0;
}

// $ g++ -o main 10.9_multiple_inheritance.cpp 
// $ ./main.exe 
// Bird: Laid my eggs, am lighter now!
// Mammal1: Baby says glug!
// Reptile: shoo enemy! Spits venom!
// Platypus: Voila, I can swim!