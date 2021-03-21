#include <iostream>

// 字面量 常量 literal constant
// 通过自定义字面量，使得程序便于理解和维护
/*
returnType operator "" yourLiteral (valueType value)
{
  // conversion code here.
}
*/

// 参数 ValueType 只能是下面几个之一，具体使用哪个取决于用户定义字面量的性质:
// unsigned long long int：用于定义整型字面量。
// long double：用于定义浮点字面量。
// char、 wchar_t、 char16_t 和 char32_t：用于定义字符字面量。
// const char*：用于定义原始字符串字面量。
// const char*和 size_t：用于定义字符串字面量。
// const wchar_t*和 size_t：用于定义字符串字面量。
// const char16_t*和 size_t：用于定义字符串字面量。
// const char32_t*和 size_t：用于定义字符串字面量。

struct Temperature
{
  double Kelvin;
  Temperature(long double kelvin) : Kelvin(kelvin) {}
};

Temperature operator "" _C(long double celcius)
{
  return Temperature(celcius + 273);
}

Temperature operator "" _F(long double fahrenheit)
{
  return Temperature((fahrenheit + 459.67) * 5 / 9);
}


// 华氏温度和摄氏温度转换为开尔文温度
int main(int argc, char** argv)
{
  Temperature k1 = 31.73_F;
  Temperature k2 = 0.0_C;

  std::cout << "k1 is " << k1.Kelvin << " Kelvin." << std::endl;
  std::cout << "k2 is " << k2.Kelvin << " Kelvin." << std::endl;
  
  return 0;
}

// $ g++ -o main 12.9_literal_constant.cpp 
// $ ./main.exe 
// k1 is 273 Kelvin.
// k2 is 273 Kelvin.
