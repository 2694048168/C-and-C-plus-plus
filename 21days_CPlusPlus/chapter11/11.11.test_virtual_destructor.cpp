#include <iostream>

class Vehicle
{
public:
  Vehicle() 
  {
    std::cout << "Base Class Vehicle constructor." << std::endl;
  }
  // ~Vehicle()
  // {
  //   std::cout << "Base Class Vehicle destructor." << std::endl;
  // }
  
  virtual ~Vehicle()
  {
    std::cout << "Base Class Vehicle destructor." << std::endl;
  }
};

class Car: public Vehicle
{
public:
  Car() 
  {
    std::cout << "Derive Class Car constructor." << std::endl;
  }
  ~Car() 
  {
    std::cout << "Derive Class Car destructor." << std::endl;
  }
};


int main(int argc, char** argv)
{
  Vehicle* pMyRacer = new Car;
  delete pMyRacer;
  
  return 0;
}


// $ g++ -o main 11.11.test_virtual_destructor.cpp 
// $ ./main.exe 
// Base Class Vehicle constructor.
// Derive Class Car constructor.  
// Base Class Vehicle destructor. 

// 将基类析构函数修改为虚析构函数结果如下：

// $ g++ -o main 11.11.test_virtual_destructor.cpp 
// $ ./main.exe
// Base Class Vehicle constructor.
// Derive Class Car constructor.  
// Derive Class Car destructor.   
// Base Class Vehicle destructor.

