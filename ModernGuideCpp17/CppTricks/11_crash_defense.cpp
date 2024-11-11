/**
 * @file 11_crash_defense.cpp
 * @author Wei Li (weili_yzzcq@163.com)
 * @brief 
 * @version 0.1
 * @date 2024-11-11
 * 
 * @copyright Copyright (c) 2024
 * 
 */

#include <array>
#include <chrono>
#include <iostream>
#include <numeric>
#include <span>
#include <string>

/**
  * @brief C++ 中崩溃(crash) 很多情况下是发生了未定义行为(undefined behavior, UB)
  * C++ 中最强大的两大法宝: 编译期检查和运行时检查, 让代码固若金汤, 更加稳健
  * 
  * *====编译期魔法
  * 1. constexpr 编译期计算小助手: 提前计算各种数值; 优化程序性能; 在编译期捕获错误;
  * 2. static_assert 编译期的守护神: 编译时就解决那些调皮的bug;
  * 3. 模板推导 auto: 编译期的千里眼;
  * 
  * *====运行时检查
  * 1. 原始指针 VS 标准库容器: 自动内存管理; 边界检查; 异常安全; 代码简洁;
  * 2. std::span: 现代 C++ 中给数组加了一道隐形防护罩， 再也不用担心越界问题;
  *    ----零开销抽象,性能杠杠的; 自带范围检查, 安全感满满;支持动态和静态大小,灵活性max;和各种容器完美配合,最佳搭档;
  * 3. 打造终极安全字符串
  * 
  */

const double PI_ = 3.14159265359;

const int factorial_(int n)
{
    // 运行时才能计算
    if (n <= 1)
        return 1;
    return n * factorial_(n - 1);
}

constexpr double PI = 3.14159265359;

constexpr int factorial(int n)
{
    // 编译期就能算好！
    if (n <= 1)
        return 1;
    return n * factorial(n - 1);
}

constexpr int square(int x)
{
    return x * x;
}

template<typename T>
class Vector_
{
    T     *data;
    size_t size;

public:
    Vector_()
    {
        if (sizeof(T) > 256)
        {
            throw std::runtime_error("类型太大了！"); // !运行时才发现问题，太晚了
        }
    }
};

template<typename T>
class Vector
{
    // ?编译期就能发现问题，就像提前剧透一样
    static_assert(sizeof(T) <= 256, "Vector只支持256字节以下的类型哦!");

    // 防止有人塞个奇怪的类型进来
    static_assert(std::is_default_constructible_v<T>, "这个类型连默认构造都没有，您是认真的吗?");
    // ... 其余代码 ...
};

struct Student
{
    int         id   = 0;
    std::string name = "Joh";
};

Student findStudent_(Student *students, size_t size, int id)
{
    // 危险指数, 看到这个 <= 了吗? 这就是一个定时炸弹
    for (size_t i = 0; i <= size; i++)
    {
        if (students[i].id == id)
        {
            return students[i]; // 越界访问随时可能引爆
        }
    }
}

Student findStudent(const std::vector<Student> &students, int id)
{
    // 安全指数
    auto it = std::find_if(students.begin(), students.end(), [id](const Student &s) { return s.id == id; });

    if (it == students.end())
    {
        throw std::runtime_error("这位同学可能逃课了！"); // 优雅地处理异常
    }
    return *it; // vector自动管理内存，再也不怕越界
}

double calculateAverage(std::span<const double> grades)
{
    // span自带保镖，空数组? 不可能溜过去!
    if (grades.empty())
    {
        throw std::invalid_argument("咦？同学们都翘课了吗？成绩单怎么是空的！");
    }

    // 自动范围检查，再也不怕越界啦
    double sum = std::accumulate(grades.begin(), grades.end(), 0.0);
    return sum / grades.size(); // 安全又优雅，就是这么简单
}

/* 
1. 永远不会让你访问到不存在的内存;
2. 对空指针说"不！", 优雅地处理;
3. 所有操作都有边界检查, 就像过安检一样严格;
4. 背后有std::string的强力支持, 安全性双保险;
*/
class SafeString
{
public:
    // 构造函数像个严格的门卫，空指针？不存在的
    SafeString(const char *str)
        : data_(str ? str : "")
    {
    }

    // at函数就像个细心的保安，每次访问都要检查证件
    char at(size_t pos) const
    {
        if (pos >= data_.length())
        {
            throw std::out_of_range("想偷偷越界？被我抓到啦！"); // 边界检查
        }
        return data_[pos];
    }

    // substring就像个魔法分身术，但绝不会失控
    SafeString substring(size_t start, size_t length) const
    {
        if (start >= data_.length())
        {
            throw std::out_of_range("这个起点太远了，我可不敢跳！");
        }
        return SafeString(data_.substr(start, length).c_str());
    }

private:
    std::string data_; // std::string来当我们的保镖
};

int main(int argc, const char **argv)
{
    constexpr unsigned int num_repeat = 10000;

    auto start_time_ = std::chrono::high_resolution_clock::now();
    for (size_t idx{0}; idx < num_repeat; ++idx)
    {
        factorial_(idx);
    }
    auto end_time_ = std::chrono::high_resolution_clock::now();
    std::cout << "耗时为: " << std::chrono::duration_cast<std::chrono::seconds>(end_time_ - start_time_).count()
              << " s.\n";
    std::cout << "耗时为: " << std::chrono::duration_cast<std::chrono::milliseconds>(end_time_ - start_time_).count()
              << " ms.\n";
    std::cout << "耗时为: " << std::chrono::duration_cast<std::chrono::microseconds>(end_time_ - start_time_).count()
              << " us.\n";
    std::cout << "耗时为: " << std::chrono::duration_cast<std::chrono::nanoseconds>(end_time_ - start_time_).count()
              << " ns.\n";

    auto start_time = std::chrono::high_resolution_clock::now();
    for (size_t idx{0}; idx < num_repeat; ++idx)
    {
        factorial(idx);
    }
    auto end_time = std::chrono::high_resolution_clock::now();
    std::cout << "耗时为: " << std::chrono::duration_cast<std::chrono::seconds>(end_time - start_time).count()
              << " s.\n";
    std::cout << "耗时为: " << std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count()
              << " ms.\n";
    std::cout << "耗时为: " << std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time).count()
              << " us.\n";
    std::cout << "耗时为: " << std::chrono::duration_cast<std::chrono::nanoseconds>(end_time - start_time).count()
              << " ns.\n";

    constexpr int result = square(10); // 编译器直接算出100

    std::array<int, square(3)> arr; // 编译期就知道数组大小是9

    // ==========================
    std::pair<std::string, std::vector<int>> p1_ = std::make_pair(std::string("hello"), std::vector<int>{1, 2, 3});

    // *编译器帮我们多做事！毕竟编译器不会累，也不会犯错
    auto p1 = std::make_pair("hello", std::vector{1, 2, 3}); // C++17 让生活更轻松

    std::vector grades_vec = {3.1, 12.2, 2.3, 123.1, 12.12, 123.9};

    std::span<const double> grades{grades_vec};

    auto avg = calculateAverage(grades);

    std::cout << "The average is: " << avg << std::endl;

    return 0;
}
