#include <iostream>
#include <stack>

int main(int argc, char** argv)
{
  std::cout << "Please enter 5 words what you want: ";
  std::stack<std::string> strInStack;
  for (size_t i = 0; i < 5; ++i)
  {
    std::string str;
    std::cin >> str;
    strInStack.push(str);
  }

  std::cout << "--------------------------------" << std::endl;
  std::cout << "The reverse string: " << std::endl;
  while (!strInStack.empty())
  {
    std::cout << strInStack.top() << ' ';
    strInStack.pop();
  }
  
  return 0;
}

// $ g++ -o main 24.5_test_stack.cpp 
// $ ./main.exe 

// Please enter 5 words what you want: li wei jxufe software hardware
// --------------------------------
// The reverse string:
// hardware software jxufe wei li