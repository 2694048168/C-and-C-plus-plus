/**使用 STL 位标志
 * std::bitset 用于处理位和位标志表示的信息
 * 实用类，针对处理长度在编译阶段已知的位序列进行优化
 * <bitset>
 */

#include <iostream>
#include <bitset>
#include <string>

int main(int argc, char** argv)
{
  // 4 bits initialized to 0000.
  std::bitset<4> fourBits;
  std::cout << "Initial contents of fourBits: " << fourBits << std::endl;

  // 5 bits 10101.
  std::bitset<5> fiveBits ("10101");
  std::cout << "Initial contents of fiveBits: " << fiveBits << std::endl;

  // C++14 binary literal
  std::bitset<6> sixBits(0b100001);
  std::cout << "Initial contents of sixBits: " << sixBits << std::endl;

  // 8 bits initialized to long int 255.
  std::bitset<8> eightBits (255);
  std::cout << "Initial contents of eightBits: " << eightBits << std::endl;

  // instantiate one bitset as a copy of anohter.
  std::bitset<8> eightBitsCopy(eightBits);
  
  return 0;
}

// $ touch 25.1_bitset.cpp
// $ g++ -o main 25.1_bitset.cpp 
// $ ./main.exe 

// Initial contents of fourBits: 0000     
// Initial contents of fiveBits: 10101    
// Initial contents of sixBits: 100001    
// Initial contents of eightBits: 11111111