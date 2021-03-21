#include <iostream>
#include <iomanip>

// 以定点和科学计数法
int main(int argc, char** argv)
{
  const double PI = (double) 22.0 / 7;
  std::cout << "PI = " << PI << std::endl;

  std::cout << std::endl << "Setting precision to 7: " << std::endl;
  std::cout << std::setprecision(7) << "PI = " << PI << std::endl;
  std::cout << std::fixed << "Fixed PI = " << PI << std::endl;
  std::cout << std::scientific << "Scientific PI = " << PI << std::endl;

  std::cout << std::endl << "Setting precision to 10: " << std::endl;
  std::cout << std::setprecision(10) << "PI = " << PI << std::endl;
  std::cout << std::fixed << "Fixed PI = " << PI << std::endl;
  std::cout << std::scientific << "Scientific PI = " << PI << std::endl;

  std::cout << std::endl << "Please enter a radius: ";
  double radius = 0.0;
  std::cin >> radius;
  std::cout << "Area of circle: " << 2*PI*radius*radius << std::endl;
  
  return 0;
}

// $ g++ -o main 27.2_precision_scientific.cpp 
// $ ./main.exe 
// PI = 3.14286

// Setting precision to 7:
// PI = 3.142857
// Fixed PI = 3.1428571
// Scientific PI = 3.1428571e+00   

// Setting precision to 10:        
// PI = 3.1428571429e+00
// Fixed PI = 3.1428571429
// Scientific PI = 3.1428571429e+00

// Please enter a radius: 2
// Area of circle: 2.5142857143e+01
