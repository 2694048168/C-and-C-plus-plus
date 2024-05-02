/**
 * @file 06_mainCommandLine.cpp
 * @author Wei Li (weili_yzzcq@163.com)
 * @brief 
 * @version 0.1
 * @date 2024-05-02
 * 
 * @copyright Copyright (c) 2024
 * 
 */

#include <cstdio>

/**
 * @brief The main Function and the Command Line
 * main 函数和命令行 / program’s entry point
 * 所有 C++ 程序都必须包含一个名为 main 的全局函数, 这个函数被定义为程序的入口点,
 * 即在程序启动时调用的函数, 程序在启动时可以接受任意数量由环境提供的参数,
 * 这些参数称为命令行参数, 用户通过将命令行参数传递给程序的方式自定义其行为.
 * $ copy file_a.txt file_b.txt
 * C++ 程序, 可以通过声明 main 的方式来选择程序是否处理命令行参数.
 * 
 * ?main 的三个重载变体(The Three main Overloads):
 * 1. int main(); 
 * 2. int main(int argc, char* argv[]); 
 * 3. int main(int argc, char* argv[], impl-parameters);
 * 
 * 操作系统将程序可执行文件的全路径作为第一个命令行参数传递, 此行为取决于操作环境.
 * 在 macOS、Linux 和 Windows 上, 可执行文件的路径是第一个参数;
 * *此路径的格式取决于操作系统.
 * 
 * 第一个重载变体, 不带任何参数, 如果想忽略提供给程序的任何参数, 请使用此种形式;
 * 第二个重载变体, 接受两个参数, 即 argc 和 argv;
 *    第一个参数 argc 是一个非负数, 对应于 argv 中元素的个数, 环境会自动计算此数据;
 *    第二个参数 argv 是一个指向以空字符结尾的字符串的指针数组, 该字符串对应于从执行环境传入的参数;
 * 第三个重载变体, 第二个重载变体的扩展版, 接受任意数量的附加实现参数,
 *    这样目标平台可以为程序提供一些额外的参数, 实现参数在现代桌面环境中并不常见.
 * 
 * ==== 退出状态 Exit Status
 * !main 函数可以返回一个与程序退出状态对应的 int 值,
 * 这些值代表的含义是环境决定的, 在现代桌面系统上, 返回值 0 对应于程序执行成功;
 * 如果没有显式给出 return 语句, 编译器会添加一个隐式 return 0;
 * 
 */

// 直方图是显示分布相对频率的方法, 构建一个程序来计算命令行参数的字母分布的直方图.
// 1. 两个确定给定字符是大写字母还是小写字母的辅助函数
constexpr char pos_A{65}, pos_Z{90}, pos_a{97}, pos_z{122};

constexpr bool within_AZ(char x)
{
    return pos_A <= x && pos_Z >= x;
}

constexpr bool within_az(char x)
{
    return pos_a <= x && pos_z >= x;
}

// 使它获取命令行元素并存储字符频率
struct AlphaHistogram
{
    // ingest 方法将接受以空字符结尾的字符串并适当地更新counts
    void ingest(const char *x);

    // print 方法将显示存储在 counts 中的直方图信息
    void print() const;

private:
    // 存储每个字母的频率, 每当构造时, 此数组都会初始化为零
    size_t counts[26]{};
};

// -----------------------------------
int main(int argc, const char **argv)
{
    printf("Arguments: %d\n", argc);
    for (int i{}; i < argc; i++)
    {
        // printf("%zd: %s\n", i, argv[i]);
        printf("%d: %s\n", i, argv[i]);
    }

    printf("============ histogram ============\n");
    AlphaHistogram hist;
    for (int i{1}; i < argc; i++)
    {
        hist.ingest(argv[i]);
    }
    hist.print();
    // test_str == "The quick brown fox jumps over the lazy dog"

    return 0;
}

void AlphaHistogram::ingest(const char *x)
{
    size_t index{};
    while (const auto &c = x[index])
    {
        if (within_AZ(c))
            counts[c - pos_A]++;
        else if (within_az(c))
            counts[c - pos_a]++;
        index++;
    }
}

void AlphaHistogram::print() const
{
    for (auto index{pos_A}; index <= pos_Z; index++)
    {
        printf("%c: ", index);
        auto n_asterisks = counts[index - pos_A];
        while (n_asterisks--) printf("*");
        printf("\n");
    }
}
