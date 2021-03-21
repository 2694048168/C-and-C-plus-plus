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

  if (dayInput == Sunday)
  {
    std::cout << "Sunday was named after the Sun." << std::endl;
  }
  else if (dayInput == Monday)
  {
    std::cout << "Monday was named after the Moon." << std::endl;
  }
  else if (dayInput == Tuesday)
  {
    std::cout << "Tuesday was named after the Mars." << std::endl;
  }
  else if (dayInput == Wednesday)
  {
    std::cout << "Wednesday was named after the Mercury." << std::endl;
  }
  else if (dayInput == Thursday)
  {
    std::cout << "Thursday was named after the Jupiter." << std::endl;
  }
  else if (dayInput == Friday)
  {
    std::cout << "Friday was named after the Venus." << std::endl;
  }
  else if (dayInput == Saturday)
  {
    std::cout << "Saturday was named after the Saturn." << std::endl;
  }
  else
  {
    std::cout << "Wrong input, execute again." << std::endl;
  }
  
  return 0 ;
}

// $ g++ 6.4_if_else.cpp  -o main
// $ ./main.exe 
// Find what days of the week are named after!
// Enter a number for a day (Sunday = 0): 4   
// Thursday was named after the Jupiter.

// $ ./main.exe 
// Find what days of the week are named after!
// Enter a number for a day (Sunday = 0): 7   
// Wrong input, execute again.