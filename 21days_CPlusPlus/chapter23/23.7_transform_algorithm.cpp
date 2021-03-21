#include <iostream>
#include <algorithm>
#include <string>
#include <vector>
#include <deque>
#include <functional>

// 使用 std::transform() 对范围进行变换
// std::transform() 和 std::for_each() 类似，都是对范围内的元素调用指定的函数对象
// std::transform() 有两个重载版本，可用接受一元函数和二元函数
// 一元函数：toupper() tolower() 将字符串转换为大写或者小写
// 二元函数：plus() 对两个参数进行相加
int main(int argc, char** argv)
{
  std::string str {"THIS is a TEst string."};
  std::cout << "The sample string is: " << str << std::endl;

  std::string strLowerCaseCopy;
  strLowerCaseCopy.resize(str.size());
  std::transform(str.cbegin(), str.cend(), strLowerCaseCopy.begin(), ::tolower);
  std::cout << "Result of 'transform' on the string with 'tolowe': " << std::endl;
  std::cout << "\"" << strLowerCaseCopy << "\"" << std::endl << std::endl;

  // two sample vectors of integers.
  std::vector<int> numInVec1 {2021, 0, -1, 2020, 25, 42};
  std::vector<int> numInVec2 {-2, -2, -2, -2, -2, -2};

  // a destination range for holding the result of addition.
  std::deque<int> sumInDeque(numInVec1.size());
  std::transform(numInVec1.cbegin(), numInVec1.cend(), numInVec2.cbegin(), 
                 sumInDeque.begin(), std::plus<int>());
  std::cout << "Result of 'transform' using binary function 'plus': " << std::endl;
  std::cout << "Index Vector1 + Vector2 = Result (in deque)" << std::endl;
  for (size_t i = 0; i < numInVec1.size(); ++i)
  {
    std::cout << i << " \t " << numInVec1[i] << " \t " << numInVec2[i] 
              << " \t = " << sumInDeque[i] << std::endl;    
  }
  
  return 0;
}

// $ g++ -o main 23.7_transform_algorithm.cpp 
// $ ./main.exe

// The sample string is: THIS is a TEst string.
// Result of 'transform' on the string with 'tolowe':
// "this is a test string."

// Result of 'transform' using binary function 'plus':
// Index Vector1 + Vector2 = Result (in deque)
// 0        2021    -2      = 2019
// 1        0       -2      = -2
// 2        -1      -2      = -3
// 3        2020    -2      = 2018
// 4        25      -2      = 23
// 5        42      -2      = 40