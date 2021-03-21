## Assignment 作业
- 作业包括测验和练习，前者帮助读者加深对所学知识的理解，后者提供了使用新学知识的机会。
- 请尽量先完成测验和练习题，继续学习下一章前，请务必弄懂这些答案。

### 16.6.1 测验
1. std::string 具体化了哪一个 STL 模板类？
- std::basic_string<T>

2. 如果要对两个字符串进行区分大小写的比较，该如何做?
- 将两个字符串复制到两个副本对象中
- 将每一个副本字符串都转换为小写或者大写
- 对转换后的字符串进行比较，并返回结果

3. STL string 与 C 风格字符串是否类似?
- 否
- STL string 是一个类，实现了各种运算符和成员函数，使得字符串操作和处理更加简单 
- C 风格字符串实际上就是字符数组的原始指针


### 16.6.2 练习
1. 编写一个程序检查用户输入的单词是否为回文。例如，ATOYOTA 是回文，因为该单词反转后与原来相同。
- 参考文件 16.9_test_palindrome.cpp

```C++
#include <iostream>
#include <string>
#include <algorithm>

int main(int argc, char** argv)
{
  std::cout << "Please enter a word for palindrome-check:" << std::endl;
  std::string strInput;
  std::cin >> strInput;

  // 复制一份副本，反转之后与原始字符串进行对比
  std::string strCopy(strInput);
  std::reverse(strCopy.begin(), strCopy.end());
  if (strCopy == strInput)
  {
    std::cout << strInput << " is a palindrome." << std::endl;
  }
  else
  {
    std::cout << strInput << " is not a palindrome." << std::endl;
  }
  
  return 0;
}

// $ g++ -o main 16.9_test_palindrome.cpp 
// $ ./main.exe 
// Please enter a word for palindrome-check:
// liwei
// liwei is not a palindrome.

// $ ./main.exe 
// Please enter a word for palindrome-check:
// atoota
// atoota is a palindrome.
```

2. 编写一个程序，告诉用户输入的句子包含多少个元音字母。
- 参考文件 16.10_test_vowel.cpp

```C++
#include <iostream>
#include <string>

// find the number of character 'chToFind' in string "strInput"
int GetNumCharacters(std::string strInput, char chToFind)
{
  int numCharactersFound = 0;
  size_t charOffset = strInput.find(chToFind);
  while (charOffset != std::string::npos)
  {
    ++numCharactersFound;
    charOffset = strInput.find(chToFind, charOffset + 1);
  }
  return numCharactersFound;
}

int main(int argc, char** argv)
{
  std::cout << "Please enter a string:" << std::endl << ">>";
  std::string strInput;
  getline(std::cin, strInput);

  int numVowels = GetNumCharacters(strInput, 'a');
  numVowels += GetNumCharacters(strInput, 'e');
  numVowels += GetNumCharacters(strInput, 'i');
  numVowels += GetNumCharacters(strInput, 'o');
  numVowels += GetNumCharacters(strInput, 'u');

  std::cout << "The number of vowels in sentence is: " << numVowels << std::endl;

  return 0;
}

// $ g++ -o main 16.10_test_vowel.cpp 
// $ ./main.exe 
// Please enter a string:
// >>weili jxufe hello
// The number of vowels in sentence is: 7
```

3. 将字符串的字符交替地转换为大写。
- 参考文件 16.11_test_upper.cpp

```C++
#include <iostream>
#include <string>
#include <algorithm>

int main(int argc, char** argv)
{
  std::cout << "Please enter a string for case-conversion:" << std::endl << ">>";
  std::string strInput;
  getline(std::cin, strInput);
  std::cout << std::endl;

  for (size_t i = 0; i < strInput.length(); i += 2)
  {
    strInput[i] = toupper(strInput[i]);
  }

  std::cout << "The string converted to upper cass is:" << std::endl;
  std::cout << strInput << std::endl << std::endl;
  
  return 0;
}

// $ g++ -o main 16.11_test_upper.cpp 
// $ ./main.exe 
// Please enter a string for case-conversion:
// >>weili

// The string converted to upper cass is:
// WeIlI
```

4. 编写一个程序，将 4 个 string 对象分别初始化为 I、 Love、 STL 和 String，然后在这些字符串之间添加空格，再显示整个句子。
- 参考文件 16.12_test_operators.cpp

```C++
#include <iostream>
#include <string>

int main(int argc, char** argv)
{
  std::string str1 ("I");
  std::string str2 ("Love");
  std::string str3 ("STL");
  std::string str4 ("string");
  std::string strConcat (str1 + " " + str2 + " " + str3 + " " + str4);

  std::cout << strConcat << std::endl;
  
  return 0;
}

// $ g++ -o main -std=c++14 16.12_test_operators.cpp 
// $ ./main.exe 
// I Love STL string
```

5. 编写一个程序，显示字符串 Good day String! Today is beautiful!中每个 a 所在的位置。
- 参考文件 16.13_test_find.cpp

```C++
#include <iostream>
#include <string>

int main(int argc, char** argv)
{
  std::string str ("Good day String! Today is beautiful!");
  std::cout << "sample string is: " << str << std::endl;
  std::cout << "Locating all instance of character 'a' " << std::endl;

  auto charPosition = str.find('a', 0);
  while (charPosition != std::string::npos)
  {
    std::cout << "'" << 'a' <<  "' found at position: " << charPosition << std::endl;
    // make the find function searach forward from the next character onwards
    size_t charSearchPosition = charPosition + 1; 
    charPosition = str.find('a', charSearchPosition);
  }
  
  return 0;
}

// $ g++ -o main 16.13_test_find.cpp 
// $ ./main.exe 
// sample string is: Good day String! Today is beautiful!
// Locating all instance of character 'a'
// 'a' found at position: 6
// 'a' found at position: 20
// 'a' found at position: 28
```