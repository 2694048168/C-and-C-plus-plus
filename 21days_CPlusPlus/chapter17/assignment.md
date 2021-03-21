## Assignment 作业
- 作业包括测验和练习，前者帮助读者加深对所学知识的理解，后者提供了使用新学知识的机会。
- 请尽量先完成测验和练习题，继续学习下一章前，请务必弄懂这些答案。

### 17.7.1 测验
1. 在 vector 的开头或者中间插入元素时，所需时间是否固定的？
- 否
- 因为需要向后移动元素，因此与插入位置的后面的元素数量相关

2. 有一个 vector，对其调用函数 size() 和 capacity() 分别返回 10 和 20. 还可以插入多少给元素而不会导致 vector 重新分配其缓冲区 ?
- capacity - size = 10
- 插入第 11 个元素时候，将导致重新分配缓冲区
- 建议使用 reservse(num) 一次性增加分配的缓冲区

3. pop_back 函数有何功能？
- 对 std::vector std::deque 删除最后一个元素

4. 如果 vector<int> 是一个整型动态数组，那 vector<Mammal> 是什么类型的动态数组？
- 使用的是模板类，故此类型为 Mammal

5. 能否随机访问 vector 中的元素？如果是，如何访问？
- 能
- 通过下标运算符 [] 或者 使用 at() 函数

6. 那种迭代器可用于随机访问 vector 中的元素？
- 随机访问迭代器

### 17.7.2 练习
1. 编写一个交互式程序，接受用户输入的整数并将其储存到 vector 中共。用户应该能够随时使用索引查询 vector 中储存的值。
- 参考文件 17.7_test_vector.cpp

```C++
#include <iostream>
#include <vector>
#include <algorithm>

char DisplayOptions()
{
  std::cout << "==========================================" << std::endl;
  std::cout << "What would you like to do?" << std::endl;
  std::cout << "Select 1: To enter an integer." << std::endl;
  std::cout << "Select 2: Query a value given an index." << std::endl;
  std::cout << "Select 3: Query a value." << std::endl;
  std::cout << "Select 4: To display the vector." << std::endl;
  std::cout << "Select 0: To quit!" << std::endl;
  std::cout << "==========================================" << std::endl;

  char ch;
  std::cin >> ch;
  return ch;
}

int main(int argc, char** argv)
{
  std::vector<int> vecData;
  char userSelect = '\0';
  while ((userSelect = DisplayOptions() ) != '0')
  {
    if (userSelect == '1')
    {
      std::cout << "Please enter an integer to be inserted: ";
      int dataInput;
      std::cin >> dataInput;
      vecData.push_back(dataInput);
    }
    else if (userSelect == '2')
    {
      std::cout << "Please enter an index betweet 0 and " << (vecData.size() -1) << ": ";
      size_t index = 0;
      std::cin >> index;

      if (index < (vecData.size()))
      {
        std::cout << "Element [" << index << "] = " << vecData[index] << std::endl;
      }
    }
    else if (userSelect == '3')
    {
      // test std::find()
      std::cout << "Please enter the value that you want to find: ";
      int value;
      std::cin >> value;
      std::vector<int>::const_iterator elementFound = std::find(vecData.begin(), vecData.end(), value);
      if (elementFound != vecData.end())
      {
        std::cout << "Element found in the vector." << std::endl;
      }
      else
      {
        std::cout << "Element not found int the vector." << std::endl;
      }
    }
    else if (userSelect == '4')
    {
      std::cout << "The contents of the vector are: ";
      for (size_t i = 0; i < vecData.size(); ++i)
      {
        std::cout << vecData[i] << ' ';
      }
      std::cout << std::endl;
    }
  }
  
  return 0;
}

// $ g++ -o main 17.7_test_vector.cpp                             
// $ ./main.exe 

// ==========================================
// What would you like to do?
// Select 1: To enter an integer.
// Select 2: Query a value given an index.
// Select 3: Query a value.
// Select 4: To display the vector.
// Select 0: To quit!
// ==========================================
// 1
// Please enter an integer to be inserted: 42
// ==========================================
// What would you like to do?
// Select 1: To enter an integer.
// Select 2: Query a value given an index.
// Select 3: Query a value.
// Select 4: To display the vector.
// Select 0: To quit!
// ==========================================
// 1
// Please enter an integer to be inserted: 24
// ==========================================
// What would you like to do?
// Select 1: To enter an integer.
// Select 2: Query a value given an index.
// Select 3: Query a value.
// Select 4: To display the vector.
// Select 0: To quit!
// ==========================================
// 1
// Please enter an integer to be inserted: 2021
// ==========================================
// What would you like to do?
// Select 1: To enter an integer.
// Select 2: Query a value given an index.
// Select 3: Query a value.
// Select 4: To display the vector.
// Select 0: To quit!
// ==========================================
// 2
// Please enter an index betweet 0 and 2: 2
// Element [2] = 2021
// ==========================================
// What would you like to do?
// Select 1: To enter an integer.
// Select 2: Query a value given an index.
// Select 3: Query a value.
// Select 4: To display the vector.
// Select 0: To quit!
// ==========================================
// 3
// Please enter the value that you want to find: 42
// Element found in the vector.
// ==========================================
// What would you like to do?
// Select 1: To enter an integer.
// Select 2: Query a value given an index.
// Select 3: Query a value.
// Select 4: To display the vector.
// Select 0: To quit!
// ==========================================
// 3
// Please enter the value that you want to find: 2020
// Element not found int the vector.
// ==========================================
// What would you like to do?
// Select 1: To enter an integer.
// Select 2: Query a value given an index.
// Select 3: Query a value.
// Select 4: To display the vector.
// Select 0: To quit!
// ==========================================
// 4
// The contents of the vector are: 42 24 2021 
// ==========================================
// What would you like to do?
// Select 1: To enter an integer.
// Select 2: Query a value given an index.
// Select 3: Query a value.
// Select 4: To display the vector.
// Select 0: To quit!
// ==========================================
// 0
```

2. 对练习 1 中的程序进行扩展，使其能够告诉用户查询的值是否在 vector 中。
- 参考文件 17.7_test_vector.cpp

3. Jack 在 eBay 销售广口瓶。为了帮助他打包和发货，请编写一个程序，让他能够输入每件商品的尺寸，将其存储在 vector 中再显示到屏幕上。
- 参考文件 17.8_test_vector_using.cpp

```C++
#include <iostream>
#include <vector>
#include <string>
#include <sstream>
#include <algorithm>

char DisplayOptions()
{
  std::cout << "==========================================" << std::endl;
  std::cout << "What would you like to do?" << std::endl;
  std::cout << "Select 1: To enter length & breadth." << std::endl;
  std::cout << "Select 2: Query a value given an index." << std::endl;
  std::cout << "Select 3: To display dimensions of all packages." << std::endl;
  std::cout << "Select 0: To quit!" << std::endl;
  std::cout << "==========================================" << std::endl;

  char ch;
  std::cin >> ch;
  return ch;
}

class Dimensions
{
public:
  Dimensions(int inputL, int inputB) : length(inputL), breadth(inputB) {}

  // operator const char* () 运算符重载
  operator const char* ()
  {
    std::stringstream os;
    // os << "Length "s << length << ", Breadth: "s << breadth << std::ednl;
    os << "Length " << length << ", Breadth: " << breadth << std::endl;
    strOut = os.str();
    return strOut.c_str();
  }

private:
  int length, breadth;
  std::string strOut;
};

int main(int argc, char** argv)
{
  std::vector<Dimensions> vecData;
  char userSelect = '\0';
  while ((userSelect = DisplayOptions() ) != '0')
  {
    if (userSelect == '1')
    {
      std::cout << "Please enter length & breadth: ";
      int length = 0, breadth = 0;
      std::cin >> length;
      std::cin >> breadth;
      vecData.push_back(Dimensions(length, breadth));
    }
    else if (userSelect == '2')
    {
      std::cout << "Please enter an index betweet 0 and " << (vecData.size() -1) << ": ";
      size_t index = 0;
      std::cin >> index;

      if (index < (vecData.size()))
      {
        std::cout << "Element [" << index << "] = " << vecData[index] << std::endl;
      }
    }
    else if (userSelect == '3')
    {
      std::cout << "The contents of the vector are: ";
      for (size_t i = 0; i < vecData.size(); ++i)
      {
        std::cout << vecData[i] << ' ';
      }
      std::cout << std::endl;
    }
  }
  
  return 0;
}

// $ g++ -std=c++14 -o main 17.8_test_vector_using.cpp 
// $ ./main.exe 

// ==========================================
// What would you like to do?
// Select 1: To enter length & breadth.      
// Select 2: Query a value given an index.   
// Select 3: To display dimensions of all packages.
// Select 0: To quit!
// ==========================================
// 1
// Please enter length & breadth: 42 24
// ==========================================
// What would you like to do?
// Select 1: To enter length & breadth.
// Select 2: Query a value given an index.
// Select 3: To display dimensions of all packages.
// Select 0: To quit!
// ==========================================
// 2
// Please enter an index betweet 0 and 0: 0
// Element [0] = Length 42, Breadth: 24

// ==========================================
// What would you like to do?
// Select 1: To enter length & breadth.
// Select 2: Query a value given an index.
// Select 3: To display dimensions of all packages.
// Select 0: To quit!
// ==========================================
// 3
// The contents of the vector are: Length 42, Breadth: 24

// ==========================================
// What would you like to do?
// Select 1: To enter length & breadth.
// Select 2: Query a value given an index.
// Select 3: To display dimensions of all packages.
// Select 0: To quit!
// ==========================================
// 0
```

4. 编写一个应用程序，将一个队列初始化为包含以下 3 个字符串： Hello、 Containers are cool! 和 C++ is evolving!，并使用适用于各种队列的泛型函数来显示这些元素。另外，在这个应用程序中，使用 C++11 引入的列表初始化和 C++14 引入的 operator “”s 。
- 参考文件 17.9_test_deque.cpp

```C++
#include <iostream>
#include <deque>
#include <string>

template <typename T>
void DisplayDeque(std::deque<T> inputDeque)
{
  for (auto element = inputDeque.begin(); element != inputDeque.end(); ++element)
  {
    std::cout << *element << " ";
  }
  std::cout << std::endl;
}

int main(int argc, char** argv)
{
  // std::deque<std::string> strDeque ("Hello"s, "Containers are cool"s, "C++ is evolving!"s);
  std::deque<std::string> strDeque {"Hello", "Containers are cool", "C++ is evolving!"};

  DisplayDeque(strDeque);
  
  return 0;
}

// $ g++ -o main -std=c++14 17.9_test_deque.cpp 
// $ ./main.exe 

// Hello Containers are cool C++ is evolving! 
```