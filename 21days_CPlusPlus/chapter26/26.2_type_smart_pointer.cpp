/**智能指针的类型
 * 内存资源管理(实现的内存所有权模型)是智能指针的优势，
 * 重点在复制和赋值时如何处理资源，提高性能
 * 
 * 智能指针的分类本质就是内存资源管理策略：
 * 1. 深拷贝(深复制)
 * 2. 写时复制(Copy on Write, COW)
 * 3. 引用计数
 * 4. 引用链接
 * 5. 破坏性复制 C++11 放弃了，新增了 std::unique_ptr <memory>
 */

#include <iostream>

// 使用基于深拷贝的智能指针将多态对象作为基类对象进行传递
// 避免 切除问题 slicing
template <typename T>
class destructivecopy_ptr
{
public:
  //... other functions

  // copy constructor of the deepcopy pointer
  deepcopy_smart_ptr(const deepcopy_smart_ptr &source)
  {
    // Clone() is virtual: ensures deep copy of Derived class object
    object = source->Clone();
  }

  // copy assignment operator
  deepcopy_smart_ptr &operator=(const deepcopy_smart_ptr &source)
  {
    if (object)
      delete object;

    object = source->Clone();
  }

private:
  T *object;
};

template <typename T>
class destructivecopy_ptr
{
private:
  T *object;

public:
  destructivecopy_ptr(T *input) : object(input) {}
  ~destructivecopy_ptr() { delete object; }

  // copy constructor
  destructivecopy_ptr(destructivecopy_ptr &source)
  {
    // Take ownership on copy
    object = source.object;

    // destroy source
    source.object = 0;
  }

  // copy assignment operator
  destructivecopy_ptr& operator = (destructivecopy_ptr& source)
  {
    if (object != source.object)
    {
      delete object;
      object = source.object;
      source.object = 0;
    }
  }
};

int main(int argc, char **argv)
{
  std::cout << "hello smart pointer." << std::endl;
  std::cout << "hello smart pointer with deep copy." << std::endl;
  std::cout << "hello smart pointer with Copy on Write COW." << std::endl;
  std::cout << "hello smart pointer with reference count." << std::endl;
  std::cout << "hello smart pointer with reference link." << std::endl;
  std::cout << "hello smart pointer with destructive copy." << std::endl;

  // destructivecopy_ptr<int> num (new int);
  // destructivecopy_ptr<int> copy = num;
  // num is now invalid

  return 0;
}
