#include <iostream>

// 接受数组作为函数的参数
// ReturnType Function(ElementType[], int);
// ReturnType Function(ElementType array[], int length);

void DisplayArray(int array[], int length)
{
  for (size_t i = 0; i < length; ++i)
  {
    std::cout << array[i] << " ";
  }
  std::cout << std::endl;
}

void DisplayArray(char array[], int length)
{
  for (size_t i = 0; i < length; ++i)
  {
    std::cout << array[i] << " ";
  }
  std::cout << std::endl;
}


int main(int argc, char** argv)
{
  int myNums[4] = {24, 58, -1, 245};
  DisplayArray(myNums, 4);

  char myStatement[7] = {'H', 'e', 'l', 'l', 'o', '!', '\0'};
  DisplayArray(myStatement, 7);

  return 0;
}