#include <iostream>

int main(int argc, char** argv)
{
  const int ARRAY_LEG_ONE = 3;
  const int ARRAY_LEG_TWO = 2;

  const int NUM_ROWS = 3;
  const int NUM_COLUMNS = 4;


  int numOne[ARRAY_LEG_ONE] = {24, 20, -1};
  int numTwo[ARRAY_LEG_TWO] = {20, -1};

  // 2D array of integers
  int MyInts[NUM_ROWS][NUM_COLUMNS] = { {34, -1, 879, 22},
                                        {24, 365, -101, -1},
                                        {-20, 40, 90, 97} };

  std::cout << "Multiplying each int in numOne by each in numTwo: " << std::endl;

  for (size_t i = 0; i < ARRAY_LEG_ONE; ++i)
  {
    for (size_t j = 0; j < ARRAY_LEG_TWO; ++j)
    {
      std::cout << numOne[i] << " * " << numTwo[j] << " = " << numTwo[j] * numOne[i] << std::endl;
    }
  }

  std::cout << "=====================================" << std::endl;

  // inteate rows, each array of int
  for (size_t row = 0; row < NUM_ROWS; ++row)
  {
    // iterate integers in each row (columns)
    for (size_t column = 0; column < NUM_COLUMNS; ++column)
    {
      std::cout << "Integer[" << row << "] [" << column << "] = " << MyInts[row][column] << std::endl; 
    }
  }

  return 0;
}

// $ g++ -o main 6.13_nested_for.cpp 
// $ ./main.exe 

// Multiplying each int in numOne by each in numTwo: 
// 24 * 20 = 480
// 24 * -1 = -24
// 20 * 20 = 400
// 20 * -1 = -20
// -1 * 20 = -20
// -1 * -1 = 1
// =====================================
// Integer[0] [0] = 34
// Integer[0] [1] = -1
// Integer[0] [2] = 879
// Integer[0] [3] = 22
// Integer[1] [0] = 24
// Integer[1] [1] = 365
// Integer[1] [2] = -101
// Integer[1] [3] = -1
// Integer[2] [0] = -20
// Integer[2] [1] = 40
// Integer[2] [2] = 90
// Integer[2] [3] = 97