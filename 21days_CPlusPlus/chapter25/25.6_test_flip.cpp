#include <iostream>
#include <bitset>
#include <vector>

template <typename T>
void DisplayContainer(const T& container)
{
  for (auto element = container.cbegin(); element != container.cend(); ++element)
  {
    std::cout << *element << ' ';
  }
  std::cout << std::endl;
}

int main(int argc, char** argv)
{
  // 4 bits initialized to 1010
  std::bitset<4> fourBits1 (10);
  std::cout << "Initial contents of fourBits: " << fourBits1 << std::endl;
  std::cout << "-----------------------------------" << std::endl;
  std::cout << "Initial contents of fourBits with flip: " << fourBits1.flip() << std::endl;

  // 4 bits initialized to 0010
  std::vector<bool> fourBits2 {false, false, true, false};
  std::cout << "Initial contents of fourBits: " << std::endl;
  DisplayContainer(fourBits2);
  std::cout << "-----------------------------------" << std::endl;
  fourBits2.flip();
  std::cout << "Initial contents of fourBits with flip: " << std::endl;
  DisplayContainer(fourBits2);

  return 0;
}

// $ g++ -o main 25.6_test_flip.cpp 
// $ ./main.exe 

// Initial contents of fourBits: 1010
// -----------------------------------
// Initial contents of fourBits with flip: 0101
// Initial contents of fourBits: 
// 0 0 1 0 
// -----------------------------------
// Initial contents of fourBits with flip:
// 1 1 0 1