#include <iostream>

// Function Declaration
double ComputeArea(double radius, double PI = 3.14);

int main(int argc, char** argv)
{
  std::cout << "Enter radius: ";
  double radius = 0;
  std::cin >> radius;

  std::cout << "PI is 3.14, do you wish to change this value (y/n)? ";
  char userSeclection = 'n';
  std::cin >> userSeclection;

  if (userSeclection == 'y')
  {
    std::cout << "Enter the new PI value: ";
    double newPI = 3.14;
    std::cin >> newPI;
    // Call function to compute area.
    std::cout << "Area is: " << ComputeArea(radius, newPI) << std::endl;
  }
  else
  {
    // Call function to compute area.
    // Ignore the second param, using the default value.
    std::cout << "Area is: " << ComputeArea(radius) << std::endl;
  }

  return 0;
}

// Function Definitions
double ComputeArea(double radius, double PI)
{
  return PI * radius * radius;
}

// $ g++ -o main 7.3_default_parameters.cpp 

// $ ./main.exe 
// Enter radius: 45
// PI is 3.14, do you wish to change this value (y/n)? n
// Area is: 6358.5

// $ ./main.exe 
// Enter radius: 45
// PI is 3.14, do you wish to change this value (y/n)? y
// Enter the new PI value: 3.14159265
// Area is: 6361.73