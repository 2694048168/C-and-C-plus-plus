#include <iostream>

const double PI = 3.14159265;

// Function Declaration
double ComputeArea(double radius);
double ComputeCircumference(double radius);

int main(int argc, char** argv)
{
  std::cout << "Enter radius: ";
  double radius = 0;
  std::cin >> radius;

  // Call function to compute area.
  std::cout << "Area is: " << ComputeArea(radius) << std::endl;
  // Call function to compute circumference.
  std::cout << "Circumference is: " << ComputeCircumference(radius) << std::endl;

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

// $ g++ -o main 7.1_compute_circle.cpp 
// $ ./main
// Enter radius: 4.5
// Area is: 63.6173
// Circumference is: 28.2743