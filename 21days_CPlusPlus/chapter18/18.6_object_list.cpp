#include <iostream>
#include <list>
#include <string>

// STL list 是一个模板类，可用于创建任何对象类型的列表，
// 说明运算符与谓词的重要性

// std::list<class/struct> 进行排序和删除操作方法
// 1. 在 std::list 包含的对象所属的类中，实现运算符 <
// 2. 提供排序二元谓词函数

struct ContactItem
{
  std::string name;
  std::string phone;
  std::string displayAs;

  ContactItem(const std::string& conName, const std::string& conNum)
  {
    name = conName;
    phone = conNum;
    displayAs = (name + ": " + phone);
  }

  // used by list::remove() given contact list item.
  bool operator == (const ContactItem& itemToCompare) const
  {
    return (itemToCompare.name == this->name);
  }

  // used by list::sort() without parameters.
  bool operator < (const ContactItem& itemToCompare) const
  {
    return (this->name < itemToCompare.name);
  }

  // used in displayAsContents via cout
  operator const char*() const
  {
    return displayAs.c_str();
  }
};

bool SortOnphoneNumber(const ContactItem& item1, const ContactItem& item2)
{
  // define criteria for list::sort: return true for desired order.
  return (item1.phone < item2.phone);
}

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
  std::list<ContactItem> contacts;
  contacts.push_back(ContactItem("Jack Welsch", "+1 7889879879"));
  contacts.push_back(ContactItem("Bill Gates", "+1 97789787998"));
  contacts.push_back(ContactItem("Angi Merkel", "+49 234565466"));
  contacts.push_back(ContactItem("Vlad Putin", "+7 66454564797"));
  contacts.push_back(ContactItem("Ben Affleck", "+1 745641314"));
  contacts.push_back(ContactItem("Dan Craig", "+44 123641976"));

  std::cout << "List in initial order: " << std::endl;
  DisplayAsContents(contacts);

  contacts.sort();
  std::cout << "Sorting in alphabetical order via operator< :" << std::endl;
  // 默认升序排序
  DisplayAsContents(contacts);

  // 二元谓词函数
  contacts.sort(SortOnphoneNumber);
  std::cout << "Sorting in order of phone numbers via predicate:" << std::endl; 
  DisplayAsContents(contacts);

  std::cout << "Erasing Putin from the list: " << std::endl;
  contacts.remove(ContactItem("Vlad Putin", ""));
  DisplayAsContents(contacts);

  return 0;
}

// $ touch 18.6_object_list.cpp
// $ g++ -o main 18.6_object_list.cpp 
// $ ./main.exe 

// List in initial order: 
// Jack Welsch: +1 7889879879 Bill Gates: +1 97789787998 Angi Merkel: +49 234565466 Vlad Putin: +7 66454564797 Ben Affleck: +1 745641314 Dan Craig: +44 123641976
// Sorting in alphabetical order via operator< :
// Angi Merkel: +49 234565466 Ben Affleck: +1 745641314 Bill Gates: +1 97789787998 Dan Craig: +44 123641976 Jack Welsch: +1 7889879879 Vlad Putin: +7 66454564797
// Sorting in order of phone numbers via predicate:
// Ben Affleck: +1 745641314 Jack Welsch: +1 7889879879 Bill Gates: +1 97789787998 Dan Craig: +44 123641976 Angi Merkel: +49 234565466 Vlad Putin: +7 66454564797
// Erasing Putin from the list:
// Ben Affleck: +1 745641314 Jack Welsch: +1 7889879879 Bill Gates: +1 97789787998 Dan Craig: +44 123641976 Angi Merkel: +49 234565466