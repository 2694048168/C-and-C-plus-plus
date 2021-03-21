## Assignment 作业
- 作业包括测验和练习，前者帮助读者加深对所学知识的理解，后者提供了使用新学知识的机会。
- 请尽量先完成测验和练习题，继续学习下一章前，请务必弄懂这些答案。

### 19.6.1 测验
1. 使用 set<int> 声明整型 set 时，排序标准将由那个函数提供？
- 默认使用 std::less<> 提供, 使用 运算符< 来比较两个整数，返回 bool 值
- 也可以自定义 排序谓词函数 来覆盖默认的

2. 在 multiset 中，重复的值以什么方式出现？
- multiset 在插入值时会进行排序，重复的值一定在一起，彼此相邻
- 使用成员函数 multiset.count(value) 返回重复的值的个数

3. set 和 multiset 的那个成员函数指出容器包含多少个元素？
- set.size() ; multiset.size() 该成员函数返回容器的包含元素个数

### 19.6.2 练习
1. 在不修改 ContactItem 的情况下，扩展本章的电话簿应用程序，使其能够根据电话号码查询人名（提示：调整运算符 < 和 ==，确保根据电话号码对元素进行比较和排序)。
- 参考文件 19.6_test_contactitem.cpp

```C++
#include <iostream>
#include <set>
#include <string>

template <typename T>
void DisplayContents(const T& container)
{
  for (auto element = container.cbegin(); element != container.cend(); ++element)
  {
    std::cout << *element << std::endl;
  }
  std::cout << std::endl;
}

struct ContactItem
{
  std::string name;
  std::string phoneNum;
  std::string diaplayAs;

  ContactItem(const std::string& nameInput, const std::string& phone)
  {
    name = nameInput;
    phoneNum = phone;
    diaplayAs = {name + ": " + phoneNum};
  }

  // usedd by set::find() given contact list item.
  bool operator == (const ContactItem& itemToCompare) const
  {
    return (itemToCompare.phoneNum == this->phoneNum);
  }

  // used to sort.
  bool operator < (const ContactItem& itemToCompare) const
  {
    return (this->phoneNum < itemToCompare.phoneNum);
  }

  // used in DisplayContents via cout.
  operator const char* () const
  {
    return diaplayAs.c_str();
  }
};

int main(int argc, char** argv)
{
  std::set<ContactItem> setContacts;
  setContacts.insert(ContactItem("Jack Welsch", "+1 7889 879 879"));
  setContacts.insert(ContactItem("Bill Gates", "+1 97 7897 8799 8"));
  setContacts.insert(ContactItem("Angi Merkel", "+49 23456 5466"));
  setContacts.insert(ContactItem("Vlad Putin", "+7 6645 4564 797"));
  setContacts.insert(ContactItem("John Travolta", "91 234 4564 789"));
  setContacts.insert(ContactItem("Ben Affleck", "+1 745 641 314"));
  DisplayContents(setContacts);

  // std::cout << "Please enter a phone number you wish to search: ";
  std::string inputPhoneNum {"+49 23456 5466"};
  // std::cin >> inputPhoneNum;
  // getline(std::cin, inputPhoneNum);

  auto contactFound = setContacts.find(ContactItem("", inputPhoneNum));
  if (contactFound != setContacts.end())
  {
    std::cout << "The number belongs to " << (*contactFound).name << std::endl;
    std::cout << "=======================" << std::endl;
    DisplayContents(setContacts);
  }
  else
  {
    std::cout << "Contact not found." << std::endl;
  }

  return 0;
}

// $ g++ -o main 19.6_test_contactitem.cpp 
// $ ./main.exe

// Ben Affleck: +1 745 641 314
// Jack Welsch: +1 7889 879 879
// Bill Gates: +1 97 7897 8799 8
// Angi Merkel: +49 23456 5466
// Vlad Putin: +7 6645 4564 797
// John Travolta: 91 234 4564 789

// The number belongs to Angi Merkel
// =======================
// Ben Affleck: +1 745 641 314
// Jack Welsch: +1 7889 879 879
// Bill Gates: +1 97 7897 8799 8
// Angi Merkel: +49 23456 5466
// Vlad Putin: +7 6645 4564 797
// John Travolta: 91 234 4564 789
```

2. 定义一个 multiset 来储存单词以及其含义，即将 multiset 用作词典（提示：multiset 储存的对象应该是一个包含两个字符串的结构，其中一个字符串为单词，另一个字符串是单词的含义)。
- 参考文件 19.7_test_multiset.cpp

```C++
#include <iostream>
#include <set>
#include <string>

struct PAIR_WORD_MEANING
{
  std::string word;
  std::string meaning;

  // default constructor using list initialization. 
  PAIR_WORD_MEANING(const std::string& sWord, const std::string& sMeaning)
                   : word(sWord), meaning(sMeaning)  { }

  // overloading operator < .
  bool operator < (const PAIR_WORD_MEANING& pairAnotherWord) const
  {
    return (word < pairAnotherWord.word);
  }

  // overloading operator == .
  bool operator == (const std::string& key)
  {
    return ((key == this->word));
  }
};

int main(int argc, char** argv)
{
  std::multiset<PAIR_WORD_MEANING> msetDictionary;
  PAIR_WORD_MEANING word1 ("C++", "A programming language.");
  PAIR_WORD_MEANING word2 ("Programmer", "A geek.");

  msetDictionary.insert(word1);
  msetDictionary.insert(word2);

  std::cout << "Please enter a word you wish to find the meaning off >> " << std::endl;
  std::string input;
  getline(std::cin, input);
  
  auto element = msetDictionary.find(PAIR_WORD_MEANING(input, ""));
  if (element != msetDictionary.end())
  {
    std::cout << "Meaning is: " << (*element).meaning << std::endl;
  }
  else
  {
    // 非法访问内存
    // std::cout << (*element).word << " not found." << std::endl;
    std::cout << "The word not found." << std::endl;
  }
  
  return 0;
}

// $ g++ -o main 19.7_test_multiset.cpp 
// $ ./main.exe
// Please enter a word you wish to find the meaning off >>        
// C++
// Meaning is: A programming language.

// $ ./main.exe
// Please enter a word you wish to find the meaning off >>        
// Programmer
// Meaning is: A geek.

// $ ./main.exe 
// Please enter a word you wish to find the meaning off >> 
// liwei
// The word not found.
```

3. 通过一个简单程序演示 set 不接受重复的元素，而 multiset 接受。
- 参考文件 19.8_test_repeat_element.cpp

```C++
#include <iostream>
#include <set>

template <typename T>
void DisplayContent (const T& container)
{
  for (auto element = container.begin(); element != container.end(); ++element)
  {
    std::cout << *element << " ";
  }
  std::cout << std::endl;
}

int main(int argc, char** argv)
{
  std::multiset<int> msetIntegers;
  msetIntegers.insert(5);
  msetIntegers.insert(5);
  msetIntegers.insert(5);

  std::set<int> setIntegers;
  setIntegers.insert(5);
  setIntegers.insert(5);
  setIntegers.insert(5);

  std::cout << "Displaying the contents of the multiset: ";
  DisplayContent(msetIntegers);
  std::cout << "The size of the multiset is: " << msetIntegers.size() << std::endl;
  std::cout << "=========================================" << std::endl;

  std::cout << "Displaying the contents of the set: ";
  DisplayContent(setIntegers);
  std::cout << "The size of the set is: " << setIntegers.size() << std::endl;
  std::cout << "=========================================" << std::endl;
  
  return 0;
}

// $ g++ -o main 19.8_test_repeat_element.cpp 
// $ ./main.exe 

// Displaying the contents of the multiset: 5 5 5 
// The size of the multiset is: 3
// =========================================
// Displaying the contents of the set: 5
// The size of the set is: 1
// =========================================
```