/* exercise 5-15、5-16、5-17
** 练习5.15: 说明下列循环的含义并改正其中的错误。
** (a) for (int ix = 0; ix != sz; ++ix){ }
**     if (ix != sz)
**          // ...
** (b) int ix;
**     for (ix != sz; ++ix) {  }
** (c) for (int ix = 0; ix != sz; ++ix，++sz){ }
**
** solution: 
** (a) 局部变量 ix 声明周期只在 for 循环内部
** (b) for 循环允许省略，但是需要加分号 ；
** (c) 
**
** 练习5.16: while循环特别适用于那种条件保持不变、反复执行操作的情况，
** 例如，当未达到文件末尾时不断读取下一个值。
** for循环则更像是在按步骤迭代，它的索引值在某个范围内依次变化。
** 根据每种循环的习惯用法各自编写段程序，然后分别用另一种循环改写。
** 如果只能使用一种循环，你倾向于使用哪种呢? 为什么?
** solution：
** 
** 练习5.17: 假设有两个包含整数的vector对象,编写一段程序， 检验其中一个vector对象是否是另一个的前缀。
** 为了实现这一目标，对于两个不等长的vector对象，只需挑出长度较短的那个，
** 把它的所有元素和另一个vector对象比较即可。
** 例如，如果两个vector对象的元素分别是0、1、1、2和0、1、1、2、3、5、8，则程序的返回结果应该为真。
**
*/

#include <iostream>
#include <vector>

// solutin 5-17
bool is_prefix(std::vector<int> const &ivec_one, std::vector<int> const &ivec_two)
{
    if (ivec_one.size() > ivec_two.size())
    {
        return is_prefix(ivec_two, ivec_one);
    }
    
    // 第二个参数长度比第一个参数长度要长
    for (unsigned i = 0; i != ivec_one.size(); ++i)
        if (ivec_one[i] != ivec_two[i])
            return false;
    return true;
}

int main()
{
    // solution 5-17
    std::vector<int> ivec_one{0, 1, 1, 2};
    std::vector<int> ivec_two{0, 1, 1, 2, 3, 5, 8};

    std::cout << (is_prefix(ivec_two, ivec_two) ? "Yes" : "No") << std::endl;

    return 0;
}

/* 编译命令操作流程
** 0、打开终端 terminal，VSCode使用 Ctrl+shift+`; Ubuntu使用 Ctrl+Alt+T
** 1、进入当前源文件路径，cd src; cd chapter05
** 2、编译源代码文件，g++ --version; g++ -o exercise exercise5-15-16-17.cpp
** 3、运行生成的可执行程序，exercise; Ubuntu使用 ./exercise
** 4、删除生成的可执行程序，rm -rf exercise.exe; Ubuntu使用 rm -rf exercise
*/
