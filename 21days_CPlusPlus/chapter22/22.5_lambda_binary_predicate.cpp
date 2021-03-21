#include <iostream>
#include <algorithm>
#include <string>
#include <vector>

// 参考文件 21.4_binary_function.cpp
// 全部没有变化，只是将二元函数的函数对象，替换为 lambda 函数
// 代码简洁了很多，但是函数对象的优点在于只写一次，随时调用
// 故此 lambda expressio 一定要简短，高效，才采用

template <typename T>
void DisplayContainer(const T &container)
{
  for (auto element = container.begin(); element != container.end(); ++element)
  {
    std::cout << *element << std::endl;
  }
  std::cout << std::endl;
}

int main(int argc, char **argv)
{
  // define a vector of string to hold names.
  std::vector<std::string> names;

  // insert some sample names in to the vector.
  names.push_back("jim");
  names.push_back("Jack");
  names.push_back("Sam");
  names.push_back("Anna");

  std::cout << "The names in vector in order of insertion: " << std::endl;
  DisplayContainer(names);
  std::cout << "==============================================" << std::endl;

  std::cout << "Names after sorting using default std::less<> : " << std::endl;
  std::sort(names.begin(), names.end());
  DisplayContainer(names);
  std::cout << "==============================================" << std::endl;

  std::cout << "Names after sorting using binary predicate that ignore case: " << std::endl;
  std::sort(names.begin(), names.end(),
            [](const std::string &str1, const std::string str2) -> bool {
              std::string str1LowerCase;

              // assign space.
              str1LowerCase.resize(str1.size());

              // convert every character to the lower case.
              std::transform(str1.begin(), str1.end(), str1LowerCase.begin(), ::tolower);

              std::string str2LowerCase;
              // assign space.
              str2LowerCase.resize(str2.size());
              // convert every character to the lower case.
              std::transform(str2.begin(), str2.end(), str2LowerCase.begin(), ::tolower);

              return (str1LowerCase < str2LowerCase);
            });
  DisplayContainer(names);
  std::cout << "==============================================" << std::endl;

  return 0;
}

// $ g++ -o main 22.5_lambda_binary_predicate.cpp 
// $ ./main.exe 

// The names in vector in order of insertion:
// jim
// Jack
// Sam
// Anna

// ==============================================
// Names after sorting using default std::less<> :
// Anna
// Jack
// Sam
// jim

// ==============================================
// Names after sorting using binary predicate that ignore case: 
// Anna
// Jack
// jim
// Sam

// ==============================================