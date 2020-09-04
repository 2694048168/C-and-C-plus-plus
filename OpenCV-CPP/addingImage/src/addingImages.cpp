/**
** Simple linear blender 
**    dst = alpha*src1 + beta*src2 
*/

// Image file reading and writing
#include "opencv2/imgcodecs.hpp"

#include "opencv2/highgui.hpp"
#include <iostream>

// we're NOT "using namespace std;"
// here, to avoid collisions between the beta variable and std::beta in c++17

int main(int argc, char **argv)
{
  double alpha = 0.5;
  double beta;
  double input;

  cv::Mat src1, src2, dst;

  // Ask the user enter alpha
  std::cout << " Simple Linear Blender " << std::endl;
  std::cout << "-----------------------" << std::endl;
  std::cout << "* Enter alpha [0.0-1.0]: ";
  std::cin >> input;

  // We use the alpha provided by the user if it is between 0 and 1
  if (input >= 0 && input <= 1)
  {
    alpha = input;
  }

  // loading
  std::string src1_image = "./../../image/LinuxLogo.jpg";   // by default
  std::string src2_image = "./../../image/WindowsLogo.jpg"; // by default
  if (argc > 2)
  {
    src1_image = argv[1];
    src2_image = argv[2];
  }
  // Read images ( both have to be of the same size and type )
  src1 = cv::imread(src1_image);
  src2 = cv::imread(src2_image);

  if (src1.empty())
  {
    std::cout << "Error loading src1" << std::endl;
    return EXIT_FAILURE;
  }
  if (src2.empty())
  {
    std::cout << "Error loading src2" << std::endl;
    return EXIT_FAILURE;
  }

  // blend_images
  // Simple linear blender: dst = alpha * src1 + beta * src2 
  beta = (1.0 - alpha);
  cv::addWeighted(src1, alpha, src2, beta, 0.0, dst);

  // display
  cv::imshow("Linear Blend", dst);
  cv::waitKey(0);

  return 0;
}
