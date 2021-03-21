#include <iostream>
#include <exception>  // to catch exception bad_allco

// 使用 catch 捕获特定类型的异常
int main(int argc, char** argv)
{
  std::cout << "Please enter number of integers you wish to reserve: ";
  try
  {
    int input = 0;
    std::cin >> input;

    // request memory space and then return it.
    int* numArray = new int[input];
    delete[] numArray;
  }
  catch(std::bad_alloc& exp)  // 捕获特定异常，内存申请失败
  {
    std::cout << "Exception encountered: " << exp.what() << std::endl;
    std::cout << "Go to end, sorry." << std::endl;
  }
  catch(...)  // 捕获所有异常类型
  {
    std::cerr << "Exception occurred. Go to end, sorry." << std::endl;
  }
  
  return 0;
}

// $ g++ -o main 28.2_catch_exception.cpp 
// $ ./main.exe 

// Please enter number of integers you wish to reserve: -1
// Exception encountered: std::bad_array_new_length
// Go to end, sorry.