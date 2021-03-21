#include <iostream>
#include <string>

/**aggregate initialization 聚合初始化
 * 对类和结构使用聚合初始化
 * Type objectName = {argument1, argument2, ..., argumentN}
 * 等价于下面初始化
 * Type objectName {argument1, argument2, ..., argumentN}
 * 
 * 聚合初始化可用于聚合类型，那些数据类型属于聚合类型呢？
 */

class AggregateOne
{
public:
  int num;
  double PI;
};

struct AggregateTwo
{
  char hello[6];
  int impYears[3];
  std::string world;
};

int main(int argc, char**)
{
  int myNum[] = {9, 5, -1};
  AggregateOne al {2021, 3.14};
  std::cout << "PI is approximately: " << al.PI << std::endl;

  AggregateTwo a2 { {'h', 'e', 'l', 'l', 'o'}, {2011, 2014, 2020}, "world"};

  // Alternatively
  AggregateTwo a2_2 {'h', 'e', 'l', 'l', 'o', '\0', 2011, 2014, 2020, "world"};

  std::cout << a2.hello << ' ' << a2.world << std::endl;
  std::cout << "C++ standard update scheduled in: " << a2.impYears[2] << std::endl;

  return 0;
}

// $ g++ -o main 9.16_aggregate_class.cpp 
// $ ./main.exe
// PI is approximately: 3.14
// hello world
// C++ standard update scheduled in: 2020