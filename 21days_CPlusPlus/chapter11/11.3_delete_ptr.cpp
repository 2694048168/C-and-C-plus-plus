#include <iostream>

// 为什么需要虚函数
// 构造和析构的顺序，看看编译器调用构造和析构函数
// 分析程序结果可以知道，存在问题
// $ g++ -o main 11.3_delete_ptr.cpp 
// $ ./main.exe 
// Allocating a Tuna on the free store:
// Constructor Fish.
// Constructor Tuna.
// Deleting the Tuna:
// Destructor Fish.
// ===================================
// Instantiating a Tuna on the stack:
// Constructor Fish.
// Constructor Tuna.
// Automatic destructoion as it goes out of scope: 
// Destructor Tuna.
// Destructor Fish.

// 可能会导致资源未释放，内存泄漏等严重问题
// 需要将析构函数声明为虚函数
class Fish
{
public:
  Fish()
  {
    std::cout << "Constructor Fish." << std::endl;
  }
  ~Fish()
  {
    std::cout << "Destructor Fish." << std::endl;
  }
};

class Tuna: public Fish
{
public:
  Tuna()
  {
    std::cout << "Constructor Tuna." << std::endl;
  }
  ~Tuna()
  {
    std::cout << "Destructor Tuna." << std::endl;
  }
};

void DeleteFishMemory(Fish* ptrFish)
{
  delete ptrFish;
}


int main(int argc, char** argv)
{
  std::cout << "Allocating a Tuna on the free store: " << std::endl;
  Tuna* ptrTuna = new Tuna;
  std::cout << "Deleting the Tuna: " << std::endl;
  DeleteFishMemory(ptrTuna);

  std::cout << "===================================" << std::endl;
  std::cout << "Instantiating a Tuna on the stack: " << std::endl;
  Tuna myDnner;
  std::cout << "Automatic destructoion as it goes out of scope: " << std::endl;

  return 0;
}
