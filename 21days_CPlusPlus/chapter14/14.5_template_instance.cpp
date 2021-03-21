#include <iostream>

// 模板的实例化 和 具体化
// 模板定义而未使用，编译器则忽略

// 模板的实例化：
// HoldPair<int, double> pairIntDboule;

// 模板的具体化
// template<> class HoldsPair<int, int>
// {
//  // implementation code here.
// }

template <typename T1, typename T2>
class HoldsPair
{
public:
  HoldsPair(const T1& value1, const T2& value2) : value1(value1), value2(value2)
  {
    
  }

  // Accessor functions
  const T1 GetFirstValue() const;
  const T2 GetSecondValue() const;

private:
  T1 value1;
  T2 value2;
};

// 具体化
// specialization of HoldsPair for types int & int here
template<> class HoldsPair<int, int>
{
public:
  HoldsPair(const int& value1, const int& value2) : value1(value1), value2(value2) {}

  // Accessor functions
  const int GetFirstValue() const
  {
    std::cout << "Teturning integer " << value1 << std::endl;
    return value1;
  }
  const int GetSecondValue() const
  {
    std::cout << "Teturning integer " << value2 << std::endl;
    return value2;
  }

private:
  int value1;
  int value2;
  std::string strFunc;
};

int main(int argc, char** argv)
{
  HoldsPair<int, int> pairIntInt(234,456);
  pairIntInt.GetFirstValue();
  pairIntInt.GetSecondValue();
  
  return 0;
}

// $ g++ -o main 14.5_template_instance.cpp 
 // $ ./main.exe 
// Teturning integer 234
// Teturning integer 456