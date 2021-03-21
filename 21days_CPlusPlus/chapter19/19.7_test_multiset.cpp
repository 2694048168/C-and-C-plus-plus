#include <iostream>
#include <set>
#include <string>

struct PAIR_WORD_MEANING
{
  std::string word;
  std::string meaning;

  // default constructor using list initialization. 
  PAIR_WORD_MEANING(const std::string& sWord, const std::string& sMeaning)
                   : word(sWord), meaning(sMeaning)  { }

  // overloading operator < .
  bool operator < (const PAIR_WORD_MEANING& pairAnotherWord) const
  {
    return (word < pairAnotherWord.word);
  }

  // overloading operator == .
  bool operator == (const std::string& key)
  {
    return ((key == this->word));
  }
};

int main(int argc, char** argv)
{
  std::multiset<PAIR_WORD_MEANING> msetDictionary;
  PAIR_WORD_MEANING word1 ("C++", "A programming language.");
  PAIR_WORD_MEANING word2 ("Programmer", "A geek.");

  msetDictionary.insert(word1);
  msetDictionary.insert(word2);

  std::cout << "Please enter a word you wish to find the meaning off >> " << std::endl;
  std::string input;
  getline(std::cin, input);
  
  auto element = msetDictionary.find(PAIR_WORD_MEANING(input, ""));
  if (element != msetDictionary.end())
  {
    std::cout << "Meaning is: " << (*element).meaning << std::endl;
  }
  else
  {
    // 非法访问内存
    // std::cout << (*element).word << " not found." << std::endl;
    std::cout << "The word not found." << std::endl;
  }
  
  return 0;
}

// $ g++ -o main 19.7_test_multiset.cpp 
// $ ./main.exe
// Please enter a word you wish to find the meaning off >>        
// C++
// Meaning is: A programming language.

// $ ./main.exe
// Please enter a word you wish to find the meaning off >>        
// Programmer
// Meaning is: A geek.

// $ ./main.exe 
// Please enter a word you wish to find the meaning off >> 
// liwei
// The word not found.
