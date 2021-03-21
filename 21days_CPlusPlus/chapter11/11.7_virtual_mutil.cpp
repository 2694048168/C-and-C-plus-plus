#include <iostream>

class Animal
{
public:
  Animal()
  {
    std::cout << "Animal constructor." << std::endl;
  }

  // sample menber
  int age;
};

class Mammal: public Animal {};
class Bird: public Animal {};
class Reptile: public Animal {};

class Platypus: public Mammal, public Bird, public Reptile
{
public:
  Platypus()
  {
    std::cout << "Platypus constructor." << std::endl;
  }
};


int main(int argc, char** argv)
{
  Platypus duckBilledP;

  // uncomment next line to see compile failure
  // age is ambiguous as there are three instances of base Animal
  // duckBilledP.age = 25;

  // 测试结果显示一个很可笑的做法，多继承这样的作法
  // 不仅占用更多的内存，而且某些操作引起编译错误，引发二义性
  // 解决方法就是使用虚继承，即派生类可能被用作基类，则派生它是最好使用虚继承

  return 0;
}


// $ g++ -o main 11.7_virtual_mutil.cpp 
// $ ./main.exe 
// Animal constructor.  
// Animal constructor.  
// Animal constructor.  
// Platypus constructor.
