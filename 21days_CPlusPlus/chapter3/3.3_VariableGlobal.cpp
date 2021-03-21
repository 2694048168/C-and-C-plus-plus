#include <iostream>

// three global intergers variable
int firstNumber = 0;
int secondNumber = 0;
int multiplicationResult = 0;

void MultiplyNumbers()
{
  std::cout << "Enter the first number: ";
  std::cin >> firstNumber;

  std::cout << "Enter the second number: ";
  std::cin >> secondNumber;

  // Multiply two numbers,store result in a variable
  int multiplicationResult = firstNumber * secondNumber;

  // Display result
  std::cout << firstNumber << " * " << secondNumber << " = " 
            << multiplicationResult << std::endl;
}

int main(int argc, char**argv)
{
  std::cout << "This program will help you multiply two numbers" << std::endl;

  // Multiply two numbers,store result in a variable
  MultiplyNumbers();

  // 全局变量，整个文件可见
  // This line will now compile and work
  std::cout << "=======================================" << std::endl;
  std::cout << firstNumber << " * " << secondNumber << " = " 
            << multiplicationResult << std::endl;

  return 0;
}


/**命令约定
 * Pascal 拼写法，每个单词的首字母都大写，常用于函数名，类名,MultiplyNumbers
 * 骆驼拼写法，第一个单词首字母小写，后面每个单词首字母大写，常用于变量名,FirstNumber
 * 匈牙利表示法，变量开头包含指出变量类型的字符，iFirstNumber
 * 文件名全部小写，使用下划线进行链接
 * 见名知意
 */