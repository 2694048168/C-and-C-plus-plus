#include <iostream>
#include <algorithm>
#include <vector>

void DisplayNums(std::vector<int>& dynArray)
{
  for_each(dynArray.begin(), dynArray.end(), [](int Element) {std::cout << Element << " ";});

  std::cout << std::endl;
}

int main(int argc, char** argv)
{
  std::vector<int> myNums;
  myNums.push_back(501);
  myNums.push_back(-1);
  myNums.push_back(25);
  myNums.push_back(-35);

  DisplayNums(myNums);

  std::cout << "Sorting them in descending order." << std::endl;

  // lambda function
  // [optional parameters] (parameter list) {statements;}
  // lambda 函数在配合 STL 使用，提高编程效率
  sort(myNums.begin(), myNums.end(), [](int num1, int num2) {return (num2 < num1);});

  DisplayNums(myNums);

  return 0;
}

// admin@weili /d/VSCode/workspace/21days_CPlusPlus/chapter7
// $ g++ -o main 7.11_lambda_function.cpp 
// $ ./main.exe 
// 501 -1 25 -35
// Sorting them in descending order.
// 501 25 -1 -35