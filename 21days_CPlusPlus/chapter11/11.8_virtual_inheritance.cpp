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

class Mammal: public virtual Animal {};
class Bird: public virtual Animal {};
class Reptile: public virtual Animal {};

// final 表明该类不能作为基类
class Platypus final : public Mammal, public Bird, public Reptile
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
  // 解决方法就是使用虚继承，即派生类可能被用作基类，则派生它是最好使用虚继承
  // 两个结果一对比就知道，优势就体现出来了
  // 在继承层次结构中，继承多个从同一个类派生而来的基类时，如果这些基类没有采用虚继承，
  // 将导致二义性，这中二义性被称之为菱形问题(Diamond Problem)
//   $ g++ -o main 11.8_virtual_inheritance.cpp 
// $ ./main.exe 
// Animal constructor.
// Platypus constructor

  return 0;
}


// $ g++ -o main 11.7_virtual_mutil.cpp 
// $ ./main.exe 
// Animal constructor.  
// Animal constructor.  
// Animal constructor.  
// Platypus constructor.
