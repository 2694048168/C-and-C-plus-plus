/* exercise 1-17、1-18
** 练习1.17 : 如果输入的值都相等，那么程序输出什么？
** 如果没有重复值，那么程序输出又是什么？
** 源程序，统计输入中每个值连续出现的次数
** solution: step0 输入的值都相等，则输出该值的统计次数
**           step1 没有重复值，则输出每个值统计次数为 1
** 练习1.18 : 运行程序，输入的值都相等，输入没有重复值，查看程序输出是什么
*/

#include <iostream>

int main()
{
    // current_value 是正在统计的数，将读入的新值存入 value
    int current_value = 0, value = 0;
    // 读取第一个数，并确保有数据可以处理
    // EOF 判断，Windows=Ctrl+Z，Enter；Unix & Linux & Mac = Ctrl+D
    if (std::cin >> current_value)
    {
        // 保存正在处理的当前值的个数
        int curr_number = 1;
        // 读取剩余的数
        while (std::cin >> value)
        {
            // 判断是否相等，相等则加1
            if (value == current_value)
            {
                ++curr_number;
            }
            else
            {
                std::cout << current_value << " occures "
                          << curr_number << " times " << std::endl;
                // 记住新值
                current_value = value;
                // 重置计数器
                curr_number = 1;
            }  
        }
        // while 循环结束
            // 打印文件最后一个值的个数
        std::cout << current_value << " occures "
                  << curr_number << " times " << std::endl;        
    }

    return 0;
}

/* 编译命令操作流程
** 0、打开终端 terminal，VSCode使用 Ctrl+shift+`; Ubuntu使用 Ctrl+Alt+T
** 1、进入当前源文件路径，cd src; cd chapter01
** 2、编译源代码文件，g++ --version; g++ -o exercise exercise1-17-18.cpp
** 3、运行生成的可执行程序，exercise; Ubuntu使用 ./exercise
** 4、输入例子：42 42 42 42 42 55 55 62 100 100 100 
**             42 42 42 42 42 42 42 42 42 42 42
**             1 2 3 4 5 6 7 8 9 
** 5、删除生成的可执行程序，rm -rf exercise.exe; Ubuntu使用 rm -rf exercise
*/
