#include <iostream>

int main(int argc, char* argv[])
{
  // 使用 枚举变量 使得代码的可读性更高
  enum DaysOfWeek
  {
    Sunday = 0,
    Monday,
    Tuesday,
    Wednesday,
    Thursday,
    Friday,
    Saturday
  };

  std::cout << "Find what days of the week are named after!" << std::endl;
  std::cout << "Enter a number for a day (Sunday = 0): ";

  int dayInput = Sunday;
  std::cin >> dayInput;

  switch (dayInput)
  {
  case Sunday:
    std::cout << "Sunday was named after the Sun." << std::endl;
    break;

  case Monday:
    std::cout << "Monday was named after the Moon." << std::endl;
  break;

  case Tuesday:
    std::cout << "Tuesday was named after the Mars." << std::endl;
  break;

  case Wednesday:
    std::cout << "Wednesday was named after the Mercury." << std::endl;
  break;

  case Thursday:
    std::cout << "Thursday was named after the Jupiter." << std::endl;
  break;

  case Friday:
    std::cout << "Friday was named after the Venus." << std::endl;
  break;

  case Saturday:
    std::cout << "Saturday was named after the Saturn." << std::endl;
  break;
  
  default:
    std::cout << "Wrong input, execute again." << std::endl;
    break;
  }

  return 0 ;
}

// $ g++ 6.5_switch_case.cpp -o main.exe 

// $ ./main.exe 
// Find what days of the week are named after!
// Enter a number for a day (Sunday = 0): 4   
// Thursday was named after the Jupiter.

// $ ./main.exe 
// Find what days of the week are named after!
// Enter a number for a day (Sunday = 0): 9   
// Wrong input, execute again.