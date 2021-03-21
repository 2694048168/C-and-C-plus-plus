#include <iostream>

const double PI = 3.14159265;

// Function Declaration
void ComputeArea(const double radius, double &result_area);
void ComputeCircumference(const double radius, double &result_circumference);

int main(int argc, char** argv)
{
  std::cout << "Enter radius: ";
  double radius = 0;
  double result_area = 0;
  double result_circumference = 0;
  std::cin >> radius;

  // Call function to compute.
  ComputeArea(radius, result_area);
  ComputeCircumference(radius, result_circumference);

  std::cout << "Area is: " << result_area << std::endl;
  std::cout << "Circumference is: " << result_circumference << std::endl;

  return 0;
}

// Function Definitions
void ComputeArea(const double radius, double &result_area)
{
  result_area =  PI * radius * radius;
}

void ComputeCircumference(const double radius, double &result_circumference)
{
  result_circumference = 2 * PI * radius;
}


// $ g++ -o main 7.13_test_default_para.cpp 
// admin@weili /d/VSCode/workspace/21days_CPlusPlus/chapter7
// $ ./main.exe 
// Enter radius: 4
// Area is: 50.2655
// Circumference is: 25.1327