/* exercise 3-18、3-19、3-20
** 练习3.18: 下面的程序合法吗?如果不合法，你准备如何修改?
** vector<int> ivec; .
** ivec[0] = 42;
** solution : 不合法，一个空的 vector 容器，里面都没有任何元素，怎么赋值呢？
** 只能使用下标进行访问，而不能进行添加
**
** 练习3.19: 如果想定义一个含有10个元素的vector对象，所有元素的值都是42，
** 请列举出三种不同的实现方法。哪种方法更好呢?为什么?
** solution：
** std::vector<int> sequence_vector {42, 42, 42, 42, 42, 42, 42, 42, 42, 42}
** std::vector<int> sequence_vector (10, 42)
** std::vector<int> sequence_vector;
** for (int i = 0; i != 10; ++i) sequence_vector.push_back(42);
** summary : 第二种方式最好
**
** 练习3.20: 读入一组整数并把它们存入一个vector对象,将每对相邻整数的和输出出来。
** 改写你的程序，这次要求先输出第1个和最后1个元素的和，
** 接着输出第2个和倒数第2个元素的和，以此类推
** solution: 高斯求和步骤！！！
*/

#include <iostream>
#include <vector>

int main()
{
    // solutin 3-20
    std::vector<int> sequence_ivector;
    int sequence_integer;
    // 读取 cin 一组整数，保存到 vector 中
    while (std::cin >> sequence_integer)
    {
        sequence_ivector.push_back(sequence_integer);
    }
    // 输出每对相邻整数的和
    std::cout << "The sum of two adjacent integers: " << std::endl;
    for (auto i = 0; i != sequence_ivector.size() - 1; ++i)
    {
        std::cout << sequence_ivector[i] + sequence_ivector[i+1] << " ";
    }
    std::cout << std::endl;

    // 输出高斯求和步骤
    // 计算输入的序列数是整数还是奇数
    auto input_size = sequence_ivector.size();
    if (input_size % 2 != 0)
    {
        // 奇数处理，+1 变成偶数个，使得中间那个数自己与自己相加
        input_size = input_size / 2 + 1;
    }
    else
    {
        input_size /= 2;
    } 
    // 高斯求和方法
    for (auto i = 0; i != input_size; ++i)
    {
        std::cout << sequence_ivector[i] + sequence_ivector[sequence_ivector.size() - 1 - i] << " ";
    }
    std::cout << std::endl;
    
    return 0;
}

/* 编译命令操作流程
** 0、打开终端 terminal，VSCode使用 Ctrl+shift+`; Ubuntu使用 Ctrl+Alt+T
** 1、进入当前源文件路径，cd src; cd chapter03
** 2、编译源代码文件，g++ --version; g++ -o exercise exercise3-18-19-20.cpp
** 3、运行生成的可执行程序，exercise; Ubuntu使用 ./exercise
** 4、随意输入一组数字，例如：1 2 3 4 5 6 7 8 9 ，键入 EOF 即可结束输入
** 5、删除生成的可执行程序，rm -rf exercise.exe; Ubuntu使用 rm -rf exercise
*/
