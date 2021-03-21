#include <iostream>

const double PI = 3.14159265;

// Function Declaration
double SurfaceArea(double radius, double height);

int main(int argc, char** argv)
{
  std::cout << "Enter radius of the cylinder: ";
  double radius = 0;
  std::cin >> radius;

  std::cout << "Enter height of the cylinder: ";
  double height = 0;
  std::cin >> height;


  // Call function to compute surface area.
  std::cout << "Surface area is: " << SurfaceArea(radius, height) << std::endl;

  return 0;
}

// Function Definitions
double SurfaceArea(double radius, double height)
{
  double area = 2 * PI * radius * radius + 2 * PI * radius * height;
  return area;
}

// $ g++ -o main 7.2_compute_surface.cpp  
// $ ./main.exe 
// Enter radius of the cylinder: 4
// Enter height of the cylinder: 5
// Surface area is: 226.195