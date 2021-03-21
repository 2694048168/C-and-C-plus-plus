#include <iostream>
#include <vector>
#include <string>
#include <sstream>
#include <algorithm>

char DisplayOptions()
{
  std::cout << "==========================================" << std::endl;
  std::cout << "What would you like to do?" << std::endl;
  std::cout << "Select 1: To enter length & breadth." << std::endl;
  std::cout << "Select 2: Query a value given an index." << std::endl;
  std::cout << "Select 3: To display dimensions of all packages." << std::endl;
  std::cout << "Select 0: To quit!" << std::endl;
  std::cout << "==========================================" << std::endl;

  char ch;
  std::cin >> ch;
  return ch;
}

class Dimensions
{
public:
  Dimensions(int inputL, int inputB) : length(inputL), breadth(inputB) {}

  // operator const char* () 运算符重载
  operator const char* ()
  {
    std::stringstream os;
    // os << "Length "s << length << ", Breadth: "s << breadth << std::ednl;
    os << "Length " << length << ", Breadth: " << breadth << std::endl;
    strOut = os.str();
    return strOut.c_str();
  }

private:
  int length, breadth;
  std::string strOut;
};

int main(int argc, char** argv)
{
  std::vector<Dimensions> vecData;
  char userSelect = '\0';
  while ((userSelect = DisplayOptions() ) != '0')
  {
    if (userSelect == '1')
    {
      std::cout << "Please enter length & breadth: ";
      int length = 0, breadth = 0;
      std::cin >> length;
      std::cin >> breadth;
      vecData.push_back(Dimensions(length, breadth));
    }
    else if (userSelect == '2')
    {
      std::cout << "Please enter an index betweet 0 and " << (vecData.size() -1) << ": ";
      size_t index = 0;
      std::cin >> index;

      if (index < (vecData.size()))
      {
        std::cout << "Element [" << index << "] = " << vecData[index] << std::endl;
      }
    }
    else if (userSelect == '3')
    {
      std::cout << "The contents of the vector are: ";
      for (size_t i = 0; i < vecData.size(); ++i)
      {
        std::cout << vecData[i] << ' ';
      }
      std::cout << std::endl;
    }
  }
  
  return 0;
}

// $ g++ -std=c++14 -o main 17.8_test_vector_using.cpp 
// $ ./main.exe 

// ==========================================
// What would you like to do?
// Select 1: To enter length & breadth.      
// Select 2: Query a value given an index.   
// Select 3: To display dimensions of all packages.
// Select 0: To quit!
// ==========================================
// 1
// Please enter length & breadth: 42 24
// ==========================================
// What would you like to do?
// Select 1: To enter length & breadth.
// Select 2: Query a value given an index.
// Select 3: To display dimensions of all packages.
// Select 0: To quit!
// ==========================================
// 2
// Please enter an index betweet 0 and 0: 0
// Element [0] = Length 42, Breadth: 24

// ==========================================
// What would you like to do?
// Select 1: To enter length & breadth.
// Select 2: Query a value given an index.
// Select 3: To display dimensions of all packages.
// Select 0: To quit!
// ==========================================
// 3
// The contents of the vector are: Length 42, Breadth: 24

// ==========================================
// What would you like to do?
// Select 1: To enter length & breadth.
// Select 2: Query a value given an index.
// Select 3: To display dimensions of all packages.
// Select 0: To quit!
// ==========================================
// 0
