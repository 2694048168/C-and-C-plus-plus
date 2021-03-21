/**自适应容器：栈和队列
 * 内部使用一种容器但是呈现另一中容器的行为特征的容器称之为自适应容器 adaptive container
 * stack, 栈是后进先出 LIFO, <stack>
 * queue, 栈是先进先出 FIFO, <queue>
 * 
 * template <class elementType, class Container=deque<Type> > class stack;
 * template <class elementType, class Container=deque<Type> > class queue;
 * 
 * stack 的成员函数
 * std::stack.push(); std::stack.pop(); std::stack.top()
 * std::stack.empty(); std::stack.size();
 */

#include <iostream>
#include <stack>
#include <vector>

int main(int argc, char** argv)
{
  // a stack of integers.
  std::stack<int> numInStack;

  // a stack of doubles.
  std::stack<double> dblsInStack;

  // a stack of doubles contained in a vector. 
  std::stack<double, std::vector<double>> boublesStackedInVec; 

  // initializing one stack to be a copy of another.
  std::stack<int> numInStackCopy(numInStack);

  // push : insert values at top of the stack.
  std::cout << "Pushing {25, 10, -1, 5} on stack in that order: " << std::endl;
  numInStack.push(25);
  numInStack.push(10);
  numInStack.push(-1);
  numInStack.push(5);

  std::cout << "Stack contains " << numInStack.size() << " elements" << std::endl;
  while (numInStack.size() != 0)
  {
    std::cout << "Popping topmost element: " << numInStack.top() << std::endl;
    numInStack.pop();
  }

  if (numInStack.empty())
  {
    std::cout << "Popping all elements empties stack." << std::endl;
  }
  
  return 0;
}

// $ g++ -o main 24.1_stack.cpp 
// $ ./main.exe 

// Pushing {25, 10, -1, 5} on stack in that order: 
// Stack contains 4 elements
// Popping topmost element: 5
// Popping topmost element: -1
// Popping topmost element: 10
// Popping topmost element: 25
// Popping all elements empties stack.