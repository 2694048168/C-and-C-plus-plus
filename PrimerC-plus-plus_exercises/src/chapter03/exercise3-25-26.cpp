/* exercise 3-25
** 练习3.25: 3.3.3节中划分分数段的程序是使用下标运算符实现的，
** 请利用迭代器改写该程序并实现完全相同的功能
**
** 在具体实现时使用一个含有11 个元素的vector对象，每个元素分别用于统计各分数段上出现的成绩个数。
** 对于某个成绩来说，将其除以10就能得到对应的分数段索引注意:两个整数相除，结果还是整数，余数部分被自动忽略掉
** 例如，42/10=4、65/10=6、100/10=10等。
** 一旦计算得到了分数段索引，就能用它作为vector对象的下标，进而取该分数段的计数值并加 1:
** //以10分为一个分数段统计成绩的数量: 0~9, 10~19.... 90~99, 100
** vector<unsigned> scores(11， 0); // 11个分数段，全都初始化为0
** unsigned grade ;
** while (cin >> grade)  //读取成绩 
** {
**     if (grade <= 100)  //只处理有效的成绩
**         ++scores [grade/10];  //将对应分数段的计数值加1
** }
**
** 练习3.26: 在100页的二分搜索程序中，为什么用的是 mid = beg + (end - beg) / 2,
** 而非 mid = (beg + end) / 2; ?
** 
** 下面的程序使用迭代器完成了二分搜索:
** // text必须是有序的
** //beg和end表示我们搜索的范围
** auto beg = text.begin()，end = text.end();
** auto mid = text.begin() + (end - beg)/2; // 初始状态下的中间点
** //当还有元素尚未检查并且我们还没有找到sought时执行循环
** while (mid != end && *mid != sought) 
** {
**     if (sought < *mid)              // 我们要找的元素在前半部分吗?
**         end = mid;                 // 如果是，调整搜索范围使得忽略掉后半部分
**     else                          // 我们要找的元素在后半部分
**         beg = mid + 1;           // 在mid之后寻找
**     mid = beg + (end-beg) / 2;  // 新的中间点
** } 
** 
** solution: 第一种写法与第二种写法在数学上得到的结果是完全一样的，很容易证明
** 但是beg＋end这一步操作很可能会出现整数溢出的风险，而 beg + (end-beg)/2 写法不会出现比end要大的中间数据，
** 所以比较安全，就不用担心这种写法会整数溢出了。
** 同时第二种写法通用性很高，而且可以使用迭代器
**
*/

#include <iostream>
#include <vector>

int main()
{
    // solutin 3-25
    std::vector<unsigned> scores(11, 0);
    unsigned grade;
    while (std::cin >> grade)
    {
        // 有效分数
        if (grade <= 100)
        {
            // iteration
            // 第一个分数段 + 分数段偏移量 = 分数所在的分数段
            ++*(scores.begin() + grade/10);
        }
    }

    return 0;
}

/* 编译命令操作流程
** 0、打开终端 terminal，VSCode使用 Ctrl+shift+`; Ubuntu使用 Ctrl+Alt+T
** 1、进入当前源文件路径，cd src; cd chapter03
** 2、编译源代码文件，g++ --version; g++ -o exercise exercise3-25-26.cpp
** 3、运行生成的可执行程序，exercise; Ubuntu使用 ./exercise
** 4、随意输入一组数字，例如：1 2 3 4 5 6 7 8 9 ，键入 EOF 即可结束输入
** 5、删除生成的可执行程序，rm -rf exercise.exe; Ubuntu使用 rm -rf exercise
*/
