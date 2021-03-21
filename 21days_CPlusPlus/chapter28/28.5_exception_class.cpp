#include <iostream>
#include <exception>
#include <string>

// std::exception class <exception>
// 1. std::bad_alloc：使用 new 申请内存失败时引发异常
// 2. std::bad_cast: 试图使用 dynamic_cast 转换错误类型时引发异常
// 3. std::ios_base:failure: 由 iostream 库中的函数和非法引起异常
// std::exception 是基类，定义虚方法 what()

class CustomException : public std::exception
{
public:
  CustomException(const char* why) : reason(why) {}

  // redefining virtual function to return 'reason'.
  virtual const char* what() const throw()
  {
    return reason.c_str();
  }

private:
  std::string reason;
};

double Divide(double dividend, double divisor)
{
  if (divisor == 0)
  {
    throw CustomException("CustomException: Dividing by 0 is a crime.");
  }
  return (dividend / divisor);
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
  catch(const std::exception& excep)
  {
    std::cerr << excep.what() << '\n';
    std::cerr << "Sorry, can't continue." << '\n';
  }
  
  return 0;
}

// $ g++ -o main 28.5_exception_class.cpp 
// $ ./main.exe 
// Please enter dividend: 42
// Please enter dividsor: 2
// Result is: 21

// $ ./main.exe 
// Please enter dividend: 42
// Please enter dividsor: 0
// Result is: CustomException: Dividing by 0 is a crime.
// Sorry, can't continue.