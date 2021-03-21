#include <iostream>
#include <string>

// find the number of character 'chToFind' in string "strInput"
int GetNumCharacters(std::string strInput, char chToFind)
{
  int numCharactersFound = 0;
  size_t charOffset = strInput.find(chToFind);
  while (charOffset != std::string::npos)
  {
    ++numCharactersFound;
    charOffset = strInput.find(chToFind, charOffset + 1);
  }
  return numCharactersFound;
}

int main(int argc, char** argv)
{
  std::cout << "Please enter a string:" << std::endl << ">>";
  std::string strInput;
  getline(std::cin, strInput);

  int numVowels = GetNumCharacters(strInput, 'a');
  numVowels += GetNumCharacters(strInput, 'e');
  numVowels += GetNumCharacters(strInput, 'i');
  numVowels += GetNumCharacters(strInput, 'o');
  numVowels += GetNumCharacters(strInput, 'u');

  std::cout << "The number of vowels in sentence is: " << numVowels << std::endl;

  return 0;
}

// $ g++ -o main 16.10_test_vowel.cpp 
// $ ./main.exe 
// Please enter a string:
// >>weili jxufe hello
// The number of vowels in sentence is: 7