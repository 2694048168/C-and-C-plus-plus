/**自适应容器：栈和队列
 * template <class elementType, class Container=deque<Type> > class queue;
 * template <class elementType, class Container=deque<Type> > class queue;
 * 
 * queue 的成员函数
 * std::queue.push(); std::queue.pop();
 * std::queue.front(); std::queue.back();
 * std::queue.empty(); std::queue.size();
 */

#include <iostream>
#include <list>
#include <queue>

int main(int argc, char** argv)
{
  // a queue of integers.
  std::queue<int> numInQueue;

  // a queue of doubles.
  std::queue<double> dblsInQueue;

  // a queue of doubles contained in a list. 
  std::queue<double, std::list<double>> boublesQueueedInVec; 

  // initializing one queue to be a copy of another.
  std::queue<int> numInQueueCopy(numInQueue);

  // push : insert values at top of the queue.
  std::cout << "Pushing {10, 5, -1, 20} on queue in that order: " << std::endl;
  numInQueue.push(10);
  numInQueue.push(5);
  numInQueue.push(-1);
  numInQueue.push(20);

  std::cout << "queue contains " << numInQueue.size() << " elements" << std::endl;
  std::cout << "Element at front: " << numInQueue.front() << std::endl;
  std::cout << "Element at back: " << numInQueue.back() << std::endl;

  while (numInQueue.size() != 0)
  {
    std::cout << "Deleting element: " << numInQueue.front() << std::endl;
    numInQueue.pop();
  }

  if (numInQueue.empty())
  {
    std::cout << "The queue in now empty." << std::endl;
  }
  
  return 0;
}

// $ g++ -o mian 24.2_queue.cpp 
// $ ./mian.exe 

// Pushing {10, 5, -1, 20} on queue in that order: 
// queue contains 4 elements
// Element at front: 10
// Element at back: 20
// Deleting element: 10
// Deleting element: 5
// Deleting element: -1
// Deleting element: 20
// The queue in now empty.