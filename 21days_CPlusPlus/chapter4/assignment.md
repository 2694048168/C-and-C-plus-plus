## Assignment 作业
- 作业包括测验和练习，前者帮助读者加深对所学知识的理解，后者提供了使用新学知识的机会。
- 请尽量先完成测验和练习题，继续学习下一章前，请务必弄懂这些答案。

### 4.8.1 测验
1. 对于程序4.1中数组arrayNumbers，第一个元素和最后一个元素的索引分别是多少？
- 对于 C/C++ 而言，数组索引从 **零** 开始
- 0；size-1（4）

2. 如果需要让用户输入字符串，该使用 C 风格的字符串吗？
- 不建议，C 风格字符串需要知道字符串的大小，一般不适合
- 建议使用 C++ 的安全类型的动态数组 std::vector

3. 在编译器看来，'\0'表示多少个字符？
- 编译器认定为一个字符，终止字符

4. 如果忘了在 C 风格字符串末尾添加终止空白符，使用它的结果将如何？
- 取决于实际使用情况，std::cout 会一直读取字符，知道遇到终止空白符
- 很有可能会导致程序的崩溃，是一种不安全的做法 

5. 根据程序4.4中矢量的声明，尝试声明一个包含 char 元素的动态数组？
```C++
#include <iostream>
#include <vector>

int main(int argc, char* argv[])
{
  // dynamic array of integer
  std::vector<char> dynamicArray (10);

  dynamicArray[0] = 'h';
  dynamicArray[1] = 'e';
  dynamicArray[2] = 'l';
  dynamicArray[2] = 'l';
  dynamicArray[2] = 'o';

  std::cout << "Number of integers in array: " << dynamicArray.size() << std::endl;
  for (int i = 0; i < 10; ++i)
  {
  std::cout << dynamicArray[i] << " "
  }
  std::cout << std::endl;

  return 0;
}
```

### 4.8.2 练习
1. 声明一个表示国际象棋棋盘的数组；该数组的类型应为枚举，该枚举定义了可能出现在棋盘方格中共的棋子。提示：这个枚举包含枚举量 Rook、Bishhop 等，从而限制了数组元素的取值范围。另外，别忘了棋盘方格也可能为空。
```C++
#include <iostream>

int main(int argc, char** argv)
{
  // 這是一種解決方案
  enum Square
  {
    Empty = 0,
    Pawn,
    Rook,
    Knight,
    Bishop,
    King,
    Queen
  };

  Square chessBoard[8][8];
  // Initialize the squares containing rooks
  chessBoard[0][0] = chessBoard[0][7] = Rook;
  chessBoard[7][0] = chessBoard[7][7] = Rook;

  return 0;
}
```

2. 查错：下面的代码段有什么错误？

```C++
int myNumbers[5] = {0};
myNumbers[5] = 450; // Setting the 5th element to value 450
```

- 数组访问下标越界，数组索引从 **零** 开始

3. 查错：下面的代码段有什么错误？

```C++
int myNumbers[5];
cout << myNumbers[3];
```

- 数组未初始化，