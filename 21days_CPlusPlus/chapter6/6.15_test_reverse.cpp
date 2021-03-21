#include <iostream>

int main(int argc, char** argv)
{
  const int ARRAY_LEG_ONE = 5;

  int numOne[ARRAY_LEG_ONE] = {24, 20, -1, 20, -1};

  for (int index = ARRAY_LEG_ONE - 1; index >= 0; --index)
  {
    std::cout << "numOne[" << index << "] = " << numOne[index] << std::endl;
  }

  std::cout << std::endl;

  return 0;
}

// $ g++ -o main 6.15_test_reverse.cpp   
// $ ./main.exe
// numOne[4] = -1
// numOne[3] = 20
// numOne[2] = -1
// numOne[1] = 20
// numOne[0] = 24
