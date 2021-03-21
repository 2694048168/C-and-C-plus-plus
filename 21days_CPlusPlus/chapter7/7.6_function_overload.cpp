#include <iostream>

const double PI = 3.14159265;

// Function Overloading
double ComputeArea(double radius);
double ComputeArea(double radius, double height);

int main(int argc, char** argv)
{
  std::cout << "Enter z for Cylinder, c for Circle: ";
  char userSelection = 'z';
  std::cin >> userSelection;

  std::cout << "Enter radius: ";
  double radius = 0;
  std::cin >> radius;

  if (userSelection == 'z')
  {
    std::cout << "Enter height: ";
    double height = 0;
    std::cin >> height;

   // Invoke overloaded variant of Area for Cyclinder 
   std::cout << "Area of cylinder is: " << ComputeArea (radius, height) << std::endl;
 }
else
  std::cout << "Area of cylinder is: " << ComputeArea (radius) << std::endl;

  return 0;
}

// Function Definitions
double ComputeArea(double radius)
{
  return PI * radius * radius;
}

double ComputeArea(double radius, double height)
{
  double area = 2 * PI * radius * radius + 2 * PI * radius * height;
  return area;
}

// $ g++ -o main 7.6_function_overload.cpp 
// $ ./main.exe 
// Enter z for Cylinder, c for Circle: z
// Enter radius: 4
// Enter height: 5
// Area of cylinder is: 226.195

// $ ./main.exe 
// Enter z for Cylinder, c for Circle: c
// Enter radius: 4
// Area of cylinder is: 50.2655