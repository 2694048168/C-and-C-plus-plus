#include <iostream>
#include <list>
#include <vector>

// std::list::insert() 由三个版本
// 1. iterator insert(iterator pos, const T& x)
// 2. void insert(iterator pos, size_type n, const T& x)
// 3. template <class InputIterator> void insert(iterator pos, InputIterator f, InputIterator l)

template <typename T>
void DisplayContents(const T& container)
{
  for (auto element = container.cbegin(); element != container.cend(); ++element)
  {
    std::cout << *element << " ";
  }
  std::cout << std::endl;
}

int main(int argc, char** argv)
{
  std::list<int> linkInt1;

  // inserting elements at the beginning. 
  linkInt1.insert(linkInt1.begin(), 2);
  linkInt1.insert(linkInt1.begin(), 1);

  // inserting an element at the end.
  linkInt1.insert(linkInt1.end(), 3);

  std::cout << "The contents of list 1 after inserting elements: " << std::endl;
  DisplayContents(linkInt1);

  std::list<int> linkInt2;
  // inserting 4 elements of the same value 0.
  linkInt2.insert(linkInt2.begin(), 4, 0);

  std::cout << "The contents of list 2 after inserting " 
            << linkInt2.size() << " elements of a value: " << std::endl;

  DisplayContents(linkInt2);

  std::list<int> linkInt3;
  // inserting elements from another list at the beginning.
  linkInt3.insert(linkInt3.begin(), linkInt1.begin(), linkInt1.end());

  std::cout << "The contents of list 3 after inserting the container of list 1 at the beginning:" << std::endl;
  DisplayContents(linkInt3);

  // inserting elements from another list at the end.
  linkInt3.insert(linkInt3.end(), linkInt2.begin(), linkInt2.end());

  std::cout << "The contents of list 3 after inserting the contents of list 2 at the end:" << std::endl;
  DisplayContents(linkInt3);
  
  return 0;
}

// $ touch 18.2_insert_list.cpp
// $ g++ -o main 18.2_insert_list.cpp 
// $ ./main.exe 

// The contents of list 1 after inserting elements:
// 1 2 3
// The contents of list 2 after inserting 4 elements of a value:  
// 0 0 0 0
// The contents of list 3 after inserting the container of list 1 at the beginning:
// 1 2 3
// The contents of list 3 after inserting the contents of list 2 at the end:
// 1 2 3 0 0 0 0