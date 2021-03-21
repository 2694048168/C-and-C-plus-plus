#include <iostream>

int main(int argc, char** argv)
{
  int number = 6;
  int result_false = number << 1 + 5 << 1;
  // 修改为
  int result_true = ( ( (number << 1) + 5) << 1);

  std::cout << "False: " << result_false << std::endl;
  std::cout << "True: " << result_true << std::endl;

  return 0;
}

// $ g++ -o main 5.6_test_priority.cpp 
// $ ./main.exe
// False: 768
// True: 34