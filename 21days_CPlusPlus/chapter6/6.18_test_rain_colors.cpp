#include <iostream>

int main(int argc, char **argv)
{
  enum AvailableColors
  {
    Violet = 0,
    Indigo,
    Blue,
    Green,
    Yellow,
    Orange,
    Red,
    Crimson,
    Beige,
    Brown,
    Peach,
    Pink,
    White,
  };

  std::cout << "Here are the available colors: " << std::endl; 
  std::cout << "============================== " << std::endl; 
  std::cout << "Violet: " << Violet << std::endl; 
  std::cout << "Indigo: " << Indigo << std::endl; 
  std::cout << "Blue: " << Blue << std::endl; 
  std::cout << "Green: " << Green << std::endl; 
  std::cout << "Yellow: " << Yellow << std::endl; 
  std::cout << "Orange: " << Orange << std::endl; 
  std::cout << "Red: " << Red << std::endl; 
  std::cout << "Crimson: " << Crimson << std::endl; 
  std::cout << "Beige: " << Beige << std::endl; 
  std::cout << "Brown: " << Brown << std::endl; 
  std::cout << "Peach: " << Peach << std::endl; 
  std::cout << "Pink: " << Pink << std::endl; 
  std::cout << "White: " << White << std::endl;
  std::cout << "============================== " << std::endl; 

  std::cout << "Please choose one color as Rainbow colors by entering code: "; 
  int userChoose = Blue;
  std::cin >> userChoose;

  // switch 这种执行效果，值得注意！！！
  switch (userChoose)
  {
  case Violet:
  case Indigo:
  case Blue:
  case Green:
  case Yellow:
  case Orange:
  case Red:
    std::cout << "Bingo, your choice is a Rainbow color." << std::endl;
    break;
  
  default:
    std::cout << "The color you chose is not in the rainbow." << std::endl;
    break;
  }

  return 0;
}