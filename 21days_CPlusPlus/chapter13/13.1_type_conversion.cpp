/**类型转换运算符
 * 类型转换是一种机制，让程序员能够暂时或者永久性改变编译器对 对象的解释
 * 这并不意味着改变了对象本身，而只是改变了编译器对 对象的解释
 * 这种可以改变对象解释方式的运算符称之为类型转换运算符
 * 
 * 1. 为何需要类型转换
 * C++ 都是类型安全而且强类型的
 * 需要与 C 语言以及一些 C 语言写的库兼容，需要让编译器按照需要的方式解释数据
 * C 风格类型转换支持者；C++ 类型转换支持者
 * 
 * 2. C++ 类型转换运算符
 * static_cast
 * dynamic_cast
 * reinterpret_cast
 * const_cast
 * 
 * 使用语法相同
 * destination_type result = cast_operator<destination_type> (object_to_cast);
 * 
 * 3. static_cast
 * 用于相关类型的指针之间进行转换，编译阶段检查
 * 
 * 4. dynamic_cast
 * 用于动态类型转换，在运行阶段判断类型转换是否成功
 * 这种运行阶段识别对象类型的机制称之为运行阶段类型识别 
 * runtime type identification, RTTI
 * 
 * 5. reinterpret_cast
 * C 风格类型转换最接近的类型转换运算符
 * 
 * 6. const_cast
 * const_cast 让程序员关闭对象的访问修饰符 const
 * 特别在对第三方库使用而无法修改情况下
 * 
 * 7. C++ 类型转换存在问题
 * 语法冗余，不如 C 风格的类型转换
 * 在现代 C++ 中，除了 dynamic_cast 之外，其他三个都可以避免
 */

#include <iostream>

// 使用动态转换判断 Fish 指针指向的是否是 Tuna 对象或者 Carp 对象
class Fish
{
public:
  virtual void Swim()
  {
    std::cout << "Fish swims ini water." << std::endl;
  }

  // base class should always have virtual destructor
  virtual ~Fish() {}
};

class Tuna : public Fish
{
public:
  void Swim()
  {
    std::cout << "Tuna swims real fast in the sea." << std::endl;
  }

  void BecomeDinner()
  {
    std::cout << "Tuna become dinner in Sushi." << std::endl;
  }
};

class Carp : public Fish
{
public:
  void Swim()
  {
    std::cout << "Carp swims real slow in the lake." << std::endl;
  }

  void Talk()
  {
    std::cout << "Carp talked Carp!" << std::endl;
  }
};

void DetectFishType(Fish* objFish)
{
  Tuna* objTuna = dynamic_cast<Tuna*>(objFish);
  // if (objTuna) to check success of cast
  if (objTuna)
  {
    std::cout << "Detected Tuna. Making Tuna dinner: " << std::endl;
    objTuna->BecomeDinner();
  }

  Carp* objCarp = dynamic_cast<Carp*>(objFish);
  // if (objTuna) to check success of cast
  if (objCarp)
  {
    std::cout << "Detected Carp. Making Tuna dinner: " << std::endl;
    objCarp->Talk();
  }

  std::cout << "Verifying type using virtual Fish::Swim: " << std::endl;
  objFish->Swim(); // calling virtual function
}


int main(int argc, char** argv)
{
  Carp myLunch;
  Tuna myDinner;

  DetectFishType(&myDinner);
  std::cout << "================================" << std::endl;
  DetectFishType(&myLunch);
  
  return 0;
}



// $ g++ -o main 13.1_type_conversion.cpp 
// $ ./main.exe

// Detected Tuna. Making Tuna dinner:       
// Tuna become dinner in Sushi.
// Verifying type using virtual Fish::Swim: 
// Tuna swims real fast in the sea.
// ================================
// Detected Carp. Making Tuna dinner:       
// Carp talked Carp!
// Verifying type using virtual Fish::Swim: 
// Carp swims real slow in the lake. 