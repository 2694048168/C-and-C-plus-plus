#include <iostream>

// 对于编译器而言，只有模板使用的时候，其代码才存在
// 静态成员 表明由类的所有实例共享的
// 模板的静态成员于此类型，由特定的具体化的所有实例共享

template <typename T>
class TestStatic
{
public:
  static int staticVal;
};

// static member initialization
template<typename T> int TestStatic<T>::staticVal;

int main(int argc, char** argv)
{
  TestStatic<int> intInstance;
  std::cout << "Setting staticVal for initialization to 2021" << std::endl;
  intInstance.staticVal = 2021;

  TestStatic<double> doubleInstance;
  std::cout << "Setting staticVal for initialization to 202" << std::endl;
  doubleInstance.staticVal = 202;

  std::cout << "intInstance.staticVal = " << intInstance.staticVal << std::endl;
  std::cout << "doubleInstance.staticVal = " << doubleInstance.staticVal << std::endl;

  return 0;
}

// $ touch 14.6_template_static.cpp
// $ g++ -o main 14.6_template_static.cpp 
// $ ./main.exe 
// Setting staticVal for initialization to 2021
// Setting staticVal for initialization to 202 
// intInstance.staticVal = 2021
// doubleInstance.staticVal = 202