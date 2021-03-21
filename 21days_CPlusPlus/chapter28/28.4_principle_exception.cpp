#include <iostream>

// 异常处理的工作原理
// 调用栈
// 引发异常时将对局部对象调用析构函数，
// 如果因出现异常而被调用的析构函数也引发异常，将导致程序异常终止。
struct StructA
{
  StructA() { std::cout << "Strucnt A construnctor" << std::endl; }
  ~StructA() { std::cout << "Strucnt A destrunctor" << std::endl; }
};

struct StructB
{
  StructB() { std::cout << "Strucnt B construnctor" << std::endl; }
  ~StructB() { std::cout << "Strucnt B destrunctor" << std::endl; }
};

void FuncB()
{
  std::cout << "In func B" << std::endl;
  StructA objA;
  StructB objB;
  std::cout << "About to throw up." << std::endl;
  throw "Throwing for the heck of it.";
}

void FuncA()
{
  try
  {
    std::cout << "In func A" << std::endl;
    StructA objA;
    StructB objB;
    FuncB();
    std::cout << "FuncA: returning to caller." << std::endl;
  }
  catch (const char* &excep)
  {
    std::cout << "FuncA: Caught exception: " << excep << std::endl;
    std::cout << "Handld it, will not throw to caller" << std::endl;
  }
}

int main(int argc, char **argv)
{
  std::cout << "Main(): Started execution" << std::endl;
  try
  {
    FuncA();
  }
  catch(const char* excep)
  {
    std::cerr << "Exception: " << excep << '\n';
  }
  std::cout << "Main(): exiting gracefully." << std::endl;

  return 0;
}

// $ g++ -o main 28.4_principle_exception.cpp 
// $ ./main.exe 

// Main(): Started execution
// In func A
// Strucnt A construnctor
// Strucnt B construnctor
// In func B
// Strucnt A construnctor
// Strucnt B construnctor
// About to throw up.
// Strucnt B destrunctor
// Strucnt A destrunctor
// Strucnt B destrunctor
// Strucnt A destrunctor
// FuncA: Caught exception: Throwing for the heck of it.
// Handld it, will not throw to caller
// Main(): exiting gracefully