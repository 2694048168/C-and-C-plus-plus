/**理解智能指针
 * 管理堆 heap (自由存储区) 中的内存时
 * 1. 使用常规指针(原始指针)
 * 2. 使用智能指针
 * 
 * C++ 的智能指针是包含重载运算符的类，智能的内存管理，减少错误
 */

#include <iostream>

// 智能指针如何实现 泛型的
// 首先重载了 解引用运算符 * 和成员选择运算符 ->
template <typename T>
class smart_pointer
{
public:
  smart_pointer(T* pData) : rawPtr(pData) {}
  ~smart_pointer() {delete rawPtr;}

  // copy constructor
  smart_pointer(const smart_pointer& anotherSP);

  // copy assignment operator
  smart_pointer& operator = (const smart_pointer& anotherSP);

  T& operator * () const
  {
    return *(rawPtr);
  }

  T* operator -> () const
  {
    return rawPtr;
  }

private:
  T* rawPtr;
};

// 智能指针的智能之处在于： 
// 1. 复制构造函数
// 2. 赋值运算符重载
// 3. 析构函数重载

int main(int argc, char** argv)
{
  std::cout << "hello smart pointer." << std::endl;
  
  return 0;
}
