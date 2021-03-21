#include <iostream>

const double PI = 3.14159265;
// Function Declaration
double ComputeArea(double radius);
double ComputeCircumference(double radius);



// Function Declaration
void QueryAndCalculate()
{
  std::cout << "Enter radius: ";
  double radius = 0;
  std::cin >> radius;

  // Call function to compute area.
  std::cout << "Area is: " << ComputeArea(radius) << std::endl;

  std::cout << "Do you wish to calculate circumference (y/n)? ";
  char calcCircum = 'n';
  std::cin >> calcCircum;

  if (calcCircum == 'n')
  {
    return;
  }

  // Call function to compute circumference.
  std::cout << "Circumference is: " << ComputeCircumference(radius) << std::endl;
  return;
}

int main(int argc, char** argv)
{
  QueryAndCalculate();
  
  return 0;
}

// Function Definitions
double ComputeArea(double radius)
{
  return PI * radius * radius;
}

double ComputeCircumference(double radius)
{
  return 2 * PI * radius;
}

// $ g++ -o main 7.5_return.cpp 
// $ ./main.exe 
// Enter radius: 4
// Area is: 50.2655
// Do you wish to calculate circumference (y/n)? y
// Circumference is: 25.1327

// $ ./main.exe 
// Enter radius: 4
// Area is: 50.2655
// Do you wish to calculate circumference (y/n)? n