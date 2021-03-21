#include <iostream>

// 使用 throw 引发特定类型的异常
double Divide (double divided, double divisor)
{
  if (divisor == 0)
  {
    throw "Dividing by 0 is a crime.";
  }
  return (divided / divisor);
}

int main(int argc, char** argv)
{
  std::cout << "Please enter dividend: ";
  double dividend = 0;
  std::cin >> dividend;

  std::cout << "Please enter dividsor: ";
  double dividsor = 0;
  std::cin >> dividsor;

  try
  {
    std::cout << "Result is: " << Divide(dividend,dividsor) << std::endl;
  }
  catch(const char*& excep)
  {
    std::cout << "Exception: " << excep << std::endl;
    std::cerr << "Sorry, can't continue." << '\n';
  }
  
  return 0;
}

// $ g++ -o main 28.3_throw_exception.cpp 
// $ ./main.exe 

// Please enter dividend: 42
// Please enter dividsor: 2
// Result is: 21

// $ ./main.exe 

// Please enter dividend: 42
// Please enter dividsor: 0
// Result is: Exception: Dividing by 0 is a crime.
// Sorry, can't continue.
