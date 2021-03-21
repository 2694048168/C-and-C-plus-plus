#include <iostream>

int main(int argc, char** argv)
{
  const int ARRAY_LEG_ONE = 3;
  const int ARRAY_LEG_TWO = 2;

  int numOne[ARRAY_LEG_ONE] = {24, 20, -1};
  int numTwo[ARRAY_LEG_TWO] = {20, -1};

  std::cout << "Adding each int in numOne by each in numTwo: " << std::endl;

  for (int i = ARRAY_LEG_ONE - 1; i >= 0; --i)
  {
    for (int j = ARRAY_LEG_TWO - 1; j >= 0; --j)
    {
      std::cout << numOne[i] << " * " << numTwo[j] << " = " << numTwo[j] * numOne[i] << std::endl;
    }
  }

  return 0;
}

// $ g++ -o main 6.16_test_reverse_add.cpp
// $ ./main.exe
// Adding each int in numOne by each in numTwo:
// -1 * -1 = 1
// -1 * 20 = -20
// 20 * -1 = -20
// 20 * 20 = 400
// 24 * -1 = -24
// 24 * 20 = 480