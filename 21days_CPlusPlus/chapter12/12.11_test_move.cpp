#include <iostream>

// 用于高性能编程的移动构造函数和移动赋值运算符
class DynIntegers
{
public:
  // default constructor.
  // 参考文件 12.8_move_constructor_assignment.cpp

  // move constructor.
  DynIntegers(DynIntegers&& moveSrc)
  {
    std::cout << "Move constructor moves: " << moveSrc.arrayNums << std::endl;
    if (moveSrc.arrayNums != NULL)
    {
      arrayNums = moveSrc.arrayNums;  // take ownershiop i.e. 'move'
      moveSrc.arrayNums = NULL; // free move source
    }
  }

  // move assignment operator
  DynIntegers& operator = (DynIntegers&& moveSrc)
  {
    std::cout << "Move assignment operator moves: " << moveSrc.arrayNums << std::endl;
    if ((moveSrc.arrayNums != NULL) && (this != &moveSrc))
    {
      delete [] arrayNums; // release own arrayNums

      arrayNums = moveSrc.arrayNums;  // take ownershiop i.e. 'move'
      moveSrc.arrayNums = NULL;  // free move source
    }

    return *this;
  }

  // copy constructor
  // 参考文件 12.8_move_constructor_assignment.cpp

  // copy assignment operator.
  // 参考文件 12.8_move_constructor_assignment.cpp

  // destructor
  ~DynIntegers()
  {
    // if (arrayNums != NULL)
    if (!arrayNums)
    {
      delete [] arrayNums;
    }
  }

private:
  int* arrayNums;
};

// 移动 避免不必要的复制和内存分配，节省处理时间，提高性能
int main(int argc, char** argv)
{
  // 参考文件 12.8_move_constructor_assignment.cpp
  return 0;
}