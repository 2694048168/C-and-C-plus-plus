#include <iostream>

int main(int argc, char**argv)
{
  int threeRowsThreeColumns[3][3] = {{-501, 205, 2018},
                                      {989, 101, 206},
                                      {303, 456, 596}};

  std::cout << "Row 0: " << threeRowsThreeColumns[0][0] << " "
                         << threeRowsThreeColumns[0][1] << " "
                         << threeRowsThreeColumns[0][2] << std::endl;

  std::cout << "Row 1: " << threeRowsThreeColumns[1][0] << " "
                         << threeRowsThreeColumns[1][1] << " "
                         << threeRowsThreeColumns[1][2] << std::endl;

  std::cout << "Row 2: " << threeRowsThreeColumns[2][0] << " "
                         << threeRowsThreeColumns[2][1] << " "
                         << threeRowsThreeColumns[2][2] << std::endl;

  std::cout << "==========================" << std::endl;

  for (size_t i = 0; i < 3; ++i)
  {
    std::cout << "Row " << i << " ";
    for (size_t j = 0; j < 3; ++j)
    {
      std::cout << threeRowsThreeColumns[i][j] << " ";
    }
    std::cout << std::endl;
  }
  
  return 0;
}


// $ g++ -o main 4.2_multidimensional_array.cpp 
// $ ./main.exe 
// Row 0: -501 205 2018      
// Row 1: 989 101 206        
// Row 2: 303 456 596        
// ==========================
// Row 0 -501 205 2018       
// Row 1 989 101 206
// Row 2 303 456 596
