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