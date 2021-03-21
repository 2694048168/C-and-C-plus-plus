#include <iostream>
#include <bitset>

int main(int argc, char** argv)
{
  // 4 bits initialized to 1010
  std::bitset<4> fourBits1 (10);
  std::cout << "Initial contents of fourBits: " << fourBits1 << std::endl;

  // 4 bits initialized to 0010
  std::bitset<4> fourBits2 (2);
  std::cout << "Initial contents of fourBits: " << fourBits2 << std::endl;

  // add bitset.
  std::bitset<4> addResult(fourBits1.to_ulong() + fourBits2.to_ulong());
  std::cout << "Initial contents of add result: " << addResult << std::endl;

  return 0;
}

// $ g++ -o main 25.5_test_bitset.cpp 
// $ ./main.exe 

// Initial contents of fourBits: 1010  
// Initial contents of fourBits: 0010  
// Initial contents of add result: 1100