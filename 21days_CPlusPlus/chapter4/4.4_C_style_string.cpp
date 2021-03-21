#include <iostream>
// #include <string.h>
#include <cstring>

int main(int argc, char** argv)
{
  // C style string
  char sayHello[] = {'H', 'e', 'l', 'l', 'o', ' ', 'W', 'o', 'r', 'l', 'd', '\0'};
  std::cout << sayHello << std::endl;
  std::cout << "size of array: " << sizeof(sayHello) << std::endl;

  std::cout << "replacing space with null" << std::endl;
  sayHello[5] = '\0';
  std::cout << sayHello << std::endl;
  std::cout << "size of array: " << sizeof(sayHello) << std::endl;

  // 分析 C 风格字符串的终止字符
  std::cout << "Enter a word NOT longer than 20 charachters: " << std::endl;
  char userInput[21] = {'\0'};
  std::cin >> userInput;
  std::cout << "Length of your input was: " << strlen(userInput) << std::endl;

  return 0;
}

// $ g++ -o main 4.4_C_style_string.cpp 
// $ ./main.exe 
// Hello World
// size of array: 12
// replacing space with null
// Hello
// size of array: 12
// Enter a word NOT longer than 20 charachters: 
// wei li
// Length of your input was: 3

// $ ./main.exe 
// Hello World
// size of array: 12
// replacing space with null
// Hello
// size of array: 12
// Enter a word NOT longer than 20 charachters: 
// WeiLi
// Length of your input was: 5