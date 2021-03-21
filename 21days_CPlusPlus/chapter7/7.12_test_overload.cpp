#include <iostream>

const double PI = 3.14159265;

// Volume of sphere = (4 * Pi * radius * radius * radius) / 3
// Volume of a cylinder = Pi * radius * radius * height

// Function Overloading
double ComputeVolume(double radius)
{
  return (4 * PI * radius * radius * radius) / 3;
}

double ComputeVolume(double radius, double height)
{
  return PI * radius * radius * height;
}


int main(int argc, char** argv)
{
  std::cout << "Enter s for sphere, c for Cylinder: ";
  char userSelection = 's';
  std::cin >> userSelection;

  std::cout << "Enter radius: ";
  double radius = 0;
  std::cin >> radius;

  if (userSelection == 'c')
  {
    std::cout << "Enter height: ";
    double height = 0;
    std::cin >> height;

   // Invoke overloaded variant of Area for Cyclinder 
   std::cout << "Area of cylinder is: " << ComputeVolume (radius, height) << std::endl;
  }
  else
    std::cout << "Area of sphere is: " << ComputeVolume (radius) << std::endl;

  return 0;
}
