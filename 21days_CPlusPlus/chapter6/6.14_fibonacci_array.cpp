#include <iostream>

int main(int argc, char** argv)
{
  const int numsToCalculate = 5;
  std::cout << "This program will calculate " << numsToCalculate 
            << " Fibonacci Numbers at a tiem." << std::endl;

  int fibonacci_one = 0, fibonacci_two = 1;
  char wantMore = '\0';
  std::cout << fibonacci_one << " " << fibonacci_two << " ";

  // 计算斐波纳契数列
  do
  {
    for (size_t i = 0; i < numsToCalculate; ++i)
    {
      std::cout << fibonacci_two + fibonacci_one << " ";
      int fibonacci_two_temp = fibonacci_two;
      fibonacci_two = fibonacci_two + fibonacci_one;
      fibonacci_one = fibonacci_two_temp;
    }
    std::cout << std::endl << "Do you want more numbers (y/n)? ";
    std::cin >> wantMore;
  } while (wantMore == 'y');
    
  std::cout << "Goodbye!" << std::endl;

  return 0;
}

// $ g++ -o main 6.14_fibonacci_array.cpp
// $ ./main.exe 

// This program will calculate 5 Fibonacci Numbers at a tiem.
// 0 1 1 2 3 5 8
// Do you want more numbers (y/n)? y
// 13 21 34 55 89 
// Do you want more numbers (y/n)? y
// 144 233 377 610 987 
// Do you want more numbers (y/n)? y
// 1597 2584 4181 6765 10946 
// Do you want more numbers (y/n)? y
// 17711 28657 46368 75025 121393 
// Do you want more numbers (y/n)? y
// 196418 317811 514229 832040 1346269 
// Do you want more numbers (y/n)? y
// 2178309 3524578 5702887 9227465 14930352 
// Do you want more numbers (y/n)? n
// Goodbye!