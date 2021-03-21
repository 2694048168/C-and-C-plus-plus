## Assignment 作业
- 作业包括测验和练习，前者帮助读者加深对所学知识的理解，后者提供了使用新学知识的机会。
- 请尽量先完成测验和练习题，继续学习下一章前，请务必弄懂这些答案。

### 18.6.1 测验
1. 与在开头或者末尾插入元素相比，在 STL list 中间插入元素是否会降低性能？
- 不会
- 将元素插入 list 中间或者两端，插入位置不会影响性能

2. 假设由两个迭代器分别指向 STL list 对象中的两个元素，然后在这两个元素之间插入一个元素。请问这种插入是否会导致这两个迭代器无效?
- 不会
- 这也是 list 的独特之处

3. 任何情况 std::list 的内容？
- list.clear();
- list.erase(list.begin(), list.end());

4. 能否在 list 中插入多个元素？
- 可以
- insert() 有一个重载版本可以用于插入集合中特定范围的元素


### 18.6.2 练习
1. 编写一个交互式程序，接受用户输入的数字并将他们插入到 list 开头。
- 参考文件 18.8_test_list.cpp

```C++
#include <iostream>
#include <list>

template <typename T>
void DisplayAsContents(const T& container)
{
  for (auto element = container.cbegin(); element != container.cend(); ++element)
  {
    std::cout << *element << " ";
  }
  std::cout << std::endl;
}

char DisplayOptions()
{
  std::cout << "==========================================" << std::endl;
  std::cout << "What would you like to do?" << std::endl;
  std::cout << "Select 1: To enter an integer." << std::endl;
  std::cout << "Select 2: To display the list." << std::endl;
  std::cout << "Select 0: To quit!" << std::endl;
  std::cout << "==========================================" << std::endl;

  char ch;
  std::cin >> ch;
  return ch;
}

int main(int argc, char** argv)
{
  std::list<int> listData;
  char userSelect = '\0';
  while ((userSelect = DisplayOptions() ) != '0')
  {
    if (userSelect == '1')
    {
      std::cout << "Please enter an integer to be inserted: ";
      int dataInput;
      std::cin >> dataInput;
      listData.push_front(dataInput);
    }
    else if (userSelect == '2')
    {
      std::cout << "Contents of list: " << std::endl;
      DisplayAsContents(listData);
    }
  }
  
  return 0;
}

// $ touch 18.8_test_list.cpp
// $ g++ -o main 18.8_test_list.cpp 
// $ ./main.exe 

// ==========================================
// What would you like to do?
// Select 1: To enter an integer.
// Select 2: To display the list.
// Select 0: To quit!
// ==========================================
// 1
// Please enter an integer to be inserted: 23
// ==========================================
// What would you like to do?
// Select 1: To enter an integer.
// Select 2: To display the list.
// Select 0: To quit!
// ==========================================
// 1
// Please enter an integer to be inserted: 42
// ==========================================
// What would you like to do?
// Select 1: To enter an integer.
// Select 2: To display the list.
// Select 0: To quit!
// ==========================================
// 1 
// Please enter an integer to be inserted: 2021
// ==========================================
// What would you like to do?
// Select 1: To enter an integer.
// Select 2: To display the list.
// Select 0: To quit!
// ==========================================
// 2
// Contents of list: 
// 2021 42 23
// ==========================================
// What would you like to do?
// Select 1: To enter an integer.
// Select 2: To display the list.
// Select 0: To quit!
// ==========================================
// 0
```

2. 使用一个简短的程序演示这样一点：在 list 中插入一个新元素，导致迭代器指向的元素的相对位置发生变化后，该迭代器有效。
- 参考文件 18.9_test_iterator_list.cpp
```C++
#include <iostream>
#include <list>

template <typename T>
void DisplayAsContents(const T& container)
{
  for (auto element = container.cbegin(); element != container.cend(); ++element)
  {
    std::cout << *element << " ";
  }
  std::cout << std::endl;
}

int main(int argc, char** argv)
{
  std::list<int> listData {42, 24, 2020, 2021, 66, 99};
  DisplayAsContents(listData);

  std::list<int>::const_iterator elementV11 = listData.begin();
  std::cout << "The value of first iterator elementV11: " << *elementV11 << std::endl;

  listData.insert(listData.begin(), 2);
  DisplayAsContents(listData);

  std::cout << "The value of first iterator elementV11: " << *elementV11 << std::endl;
  
  return 0;
}

// $ g++ -o main 18.9_test_iterator_list.cpp
// $ ./main.exe

// 42 24 2020 2021 66 99 
// The value of first iterator elementV11: 42
// 2 42 24 2020 2021 66 99
// The value of first iterator elementV11: 42
```

3. 编写一个程序，使用 list 的 insert() 函数将一个 vector 的内容插入到一个 STL list 中。
- 参考文件 18.10_test_container.cpp

```C++
#include <iostream>
#include <list>
#include <vector>

template <typename T>
void DisplayAsContents(const T& container)
{
  for (auto element = container.cbegin(); element != container.cend(); ++element)
  {
    std::cout << *element << " ";
  }
  std::cout << std::endl;
}

int main(int argc, char** argv)
{
  std::list<int> listData {1, 5, 3, 7, 9};
  std::vector<int> vectorData {2, 4, 0, 8, 6};

  std::cout << "The list: ";
  DisplayAsContents(listData);

  std::cout << "The vector: ";
  DisplayAsContents(vectorData);

  listData.insert(listData.end(), vectorData.begin(), vectorData.end());
  std::cout << "After insert: ";
  DisplayAsContents(listData);
  
  return 0;
}

// $ g++ -o main 18.10_test_container.cpp 
// $ ./main.exe 

// The list: 1 5 3 7 9 
// The vector: 2 4 0 8 6
// After insert: 1 5 3 7 9 2 4 0 8 6
```

4. 编写一个程序，对字符串 list 进行排序以及反转排序。
- 参考文件 18.11_test_reverse_sort.cpp

```C++
#include <iostream>
#include <list>
#include <string>

template <typename T>
void DisplayAsContents(const T& container)
{
  for (auto element = container.cbegin(); element != container.cend(); ++element)
  {
    std::cout << *element << " ";
  }
  std::cout << std::endl;
}

int main(int argc, char** argv)
{
  std::list<std::string> listDataStr;
  listDataStr.push_back("Jack");
  listDataStr.push_back("Skat");
  listDataStr.push_back("Anna");
  listDataStr.push_back("John");

  std::cout << "The list: ";
  DisplayAsContents(listDataStr);

  std::cout << "The list after reverse: ";
  listDataStr.reverse();
  DisplayAsContents(listDataStr);

  std::cout << "The vector after sort: ";
  DisplayAsContents(listDataStr);

  return 0;
}

// $ g++ -o main 18.11_test_reverse_sort.cpp 
// $ ./main.exe

// The list: Jack Skat Anna John 
// The list after reverse: John Anna Skat Jack
// The vector after sort: John Anna Skat Jack
```