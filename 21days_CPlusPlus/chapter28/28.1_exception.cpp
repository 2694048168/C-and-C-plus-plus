/**异常处理
 * 异常：打断程序正常运行的特殊情况
 * 异常原因：外部因素和内部因素
 */

#include <iostream>

// 使用 try-catch 捕获异常
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
  catch(...)  // 捕获所有异常类型
  {
    std::cerr << "Exception occurred. Go to end, sorry." << std::endl;
  }
  
  return 0;
}

// $ g++ -o main 28.1_exception.cpp 
// $ ./main.exe 

// Please enter number of integers you wish to reserve: -1
// Exception occurred. Go to end, sorry.