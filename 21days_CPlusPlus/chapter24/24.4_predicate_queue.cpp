/**自适应容器：栈和队列
 * template <class elementType, class Container=deque<Type> > class queue;
 * template <class elementType, class Container=deque<Type> > class queue;
 * template <class elementType, class Container=vector<Type>, 
 *           class Compare=less<typename Container::value_type> > class priority_queue;
 * 
 * priority_queue 的成员函数
 * std::queue.push(); std::queue.pop();
 * std::queue.top()
 * std::queue.empty(); std::queue.size();
 */

#include <iostream>
#include <queue>
#include <functional>  // std::greater<>

int main(int argc, char** argv)
{
  // a priority queue of integers sorted using std::greater<>
  std::priority_queue<int, std::deque<int>, std::greater<int>> numInDescendingQueue; 

  std::cout << "Inserting {10, 5, -1, 20} into the priority_queue" << std::endl;
  numInDescendingQueue.push(10);
  numInDescendingQueue.push(5);
  numInDescendingQueue.push(-1);
  numInDescendingQueue.push(20);

  std::cout << "Deleting the " << numInDescendingQueue.size() << " elements" << std::endl;
  while (!numInDescendingQueue.empty())
  {
    std::cout << "Deleting topmost element: " << numInDescendingQueue.top() << std::endl;
    numInDescendingQueue.pop();
  }
  
  return 0;
}

// $ g++ -o main 24.4_predicate_queue.cpp 
// $ ./main.exe 

// Inserting {10, 5, -1, 20} into the priority_queue
// Deleting the 4 elements
// Deleting topmost element: -1
// Deleting topmost element: 5
// Deleting topmost element: 10
// Deleting topmost element: 20