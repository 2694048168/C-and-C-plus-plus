#include <iostream>

int main(int argc, char**argv)
{
  int x = 8;
  int y = 6;
  std::cout << std::endl;
  std::cout << x - y << " " << x * y << " " << x + y;
  std::cout << std::endl;

  return 0;
}


/**Note
 * 编译命令：g++ -o hello 1.2_compute_math.cpp
 */