#include <iostream>
#include <deque>
#include <string>

template <typename T>
void DisplayDeque(std::deque<T> inputDeque)
{
  for (auto element = inputDeque.begin(); element != inputDeque.end(); ++element)
  {
    std::cout << *element << " ";
  }
  std::cout << std::endl;
}

int main(int argc, char** argv)
{
  // std::deque<std::string> strDeque ("Hello"s, "Containers are cool"s, "C++ is evolving!"s);
  std::deque<std::string> strDeque {"Hello", "Containers are cool", "C++ is evolving!"};

  DisplayDeque(strDeque);
  
  return 0;
}

// $ g++ -o main -std=c++14 17.9_test_deque.cpp 
// $ ./main.exe 

// Hello Containers are cool C++ is evolving! 