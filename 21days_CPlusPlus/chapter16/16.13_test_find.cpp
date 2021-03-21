#include <iostream>
#include <string>

int main(int argc, char** argv)
{
  std::string str ("Good day String! Today is beautiful!");
  std::cout << "sample string is: " << str << std::endl;
  std::cout << "Locating all instance of character 'a' " << std::endl;

  auto charPosition = str.find('a', 0);
  while (charPosition != std::string::npos)
  {
    std::cout << "'" << 'a' <<  "' found at position: " << charPosition << std::endl;
    // make the find function searach forward from the next character onwards
    size_t charSearchPosition = charPosition + 1; 
    charPosition = str.find('a', charSearchPosition);
  }
  
  return 0;
}

// $ g++ -o main 16.13_test_find.cpp 
// $ ./main.exe 
// sample string is: Good day String! Today is beautiful!
// Locating all instance of character 'a'
// 'a' found at position: 6
// 'a' found at position: 20
// 'a' found at position: 28