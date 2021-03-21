#include <iostream>

enum CardinalDirections
{
  North = 25,
  South,
  East,
  West
};

enum RainbowColors
{
  Violet = 0,
  Indigo,
  Blue,
  Green,
  Yellow,
  Orange,
  Red
};

enum WeeklyDayName
{
  Monday = 0,
  Tuesday,
  Wednesday,
  Thursday,
  Friday,
  Saturday,
  Sunday
};

int main(int argc, char**argv)
{
  std::cout << "Displaying directions and their symbolic values" << std::endl;
  std::cout << "North: " << North << std::endl;
  std::cout << "South: " << South << std::endl;
  std::cout << "East: " << East << std::endl;
  std::cout << "West: " << West << std::endl;

  std::cout << "Displaying RainbowColors and their symbolic values" << std::endl;
  std::cout << "Violet: " << Violet << std::endl;
  std::cout << "Blue: " << Blue << std::endl;
  std::cout << "Indigo: " << Indigo << std::endl;
  std::cout << "Green: " << Green << std::endl;
  std::cout << "Yellow: " << Yellow << std::endl;
  std::cout << "Orange: " << Orange << std::endl;
  std::cout << "Red: " << Red << std::endl;

  std::cout << "Displaying RainbowColors and their symbolic values" << std::endl;
  std::cout << "Monday: " << Monday << std::endl;
  std::cout << "Tuesday: " << Tuesday << std::endl;
  std::cout << "Wednesday: " << Wednesday << std::endl;
  std::cout << "Thursday: " << Thursday << std::endl;
  std::cout << "Friday: " << Friday << std::endl;
  std::cout << "Saturday: " << Saturday << std::endl;
  std::cout << "Sunday: " << Sunday << std::endl;

  return 0;
}


// $ g++ -o main 3.8_emumerator.cpp 
// $ ./main.exe 
// Displaying directions and their symbolic values   
// North: 25
// South: 26
// East: 27
// West: 28
// Displaying RainbowColors and their symbolic values
// Violet: 0
// Blue: 2
// Indigo: 1
// Green: 3
// Yellow: 4
// Orange: 5
// Red: 6
// Displaying RainbowColors and their symbolic values
// Monday: 0
// Tuesday: 1
// Wednesday: 2
// Thursday: 3
// Friday: 4
// Saturday: 5
// Sunday: 6