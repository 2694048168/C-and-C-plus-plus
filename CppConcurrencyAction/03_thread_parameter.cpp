/**
 * @file 03_thread_parameter.cpp
 * @author Wei Li (weili_yzzcq@163.com)
 * @brief 
 * @version 0.1
 * @date 2023-06-06
 * 
 * @copyright Copyright (c) 2023
 * 
 */

#include <functional>
#include <iostream>
#include <memory>
#include <thread>

void foo(int i, const std::string &str)
{
    std::cout << "integer: " << i << " is: " << str << "\n";
}

void func(int &i, const std::string &str)
{
    std::cout << "integer: " << i << " is: " << str << "\n";
    ++i;
}

class X
{
public:
    void do_lengthy_work(const std::string &msg)
    {
        std::cout << msg << std::endl;
    }
};

class BigObject
{
public:
    void prepare_data(const int &big_number)
    {
        this->m_number = big_number;
        std::cout << "prepare data: " << big_number << std::endl;
    }

    int get_data() const
    {
        return m_number;
    }

private:
    int m_number{0};
};

void process_big_object(std::unique_ptr<BigObject>){};

int main(int argc, const char **argv)
{
    // 向可调用对象或函数传递参数，只需要将参数作为 std::thread 构造函数的附加参数即可。
    // 注意，这些参数会拷贝至新线程的内存空间中(同临时变量一样),
    // 即使函数中的参数是引用的形式，拷贝操作(value copy)也会执行

    // 这里使用的是字符串的字面值，也就是 char const * 类型，
    // 线程的上下文完成字面值向 std::string 的转化
    int magic_number = 42;

    std::thread t{foo, magic_number, "magic number"};
    t.join();
    std::cout << "magic number: " << magic_number << std::endl;

    const char *str = "magic number pass by reference";
    // 无法保证隐式转换的操作和 std::thread 构造函数的拷贝操作的顺序
    // std::thread t1{func, magic_number, std::string(str)}; /* error? */
    std::thread t2{func, std::ref(magic_number), std::string(str)};
    t2.join();
    std::cout << "magic number: " << magic_number << std::endl;

    // std::thread 构造函数和 std::bind 的操作在标准库中以相同的机制进行定义。
    // 比如，也可以传递一个成员函数指针作为线程函数，并提供一个合适的对象指针作为第一个参数：
    // std::thread 构造函数的第三个传递参数才是可调用函数的第一个参数
    X my_x;

    std::thread t3{&X::do_lengthy_work, &my_x, std::string(str)};
    t3.join();

    // std::move “移动”是指原始对象中的数据所有权转移给另一对象，
    // 从而这些数据就不再在原始对象中保存(译者：比较像在文本编辑的剪切操作)。
    std::unique_ptr<BigObject> ptr{new BigObject};
    ptr->prepare_data(42);
    std::cout << "the data: " << ptr->get_data() << "\n";

    std::thread t5{process_big_object, std::move(ptr)};

    // 通过在 std::thread 构造函数中执行 std::move(ptr),
    // BigObject 对象的所有权首先被转移到新创建线程的的内部存储中,
    // 之后再传递给 process_big_object 函数.
    // std::cout << "the data: " << ptr->get_data() << "\n"; /* runtime error */
    t5.join();

    std::cout << "------ main thread ending ------\n";

    return 0;
}