/**
 * @file 06_stack.cpp
 * @author Wei Li (weili_yzzcq@163.com)
 * @brief 
 * @version 0.1
 * @date 2024-05-06
 * 
 * @copyright Copyright (c) 2024
 * 
 */

#include <cassert>
#include <cstdio>
#include <stack>
#include <vector>

/**
 * @brief STL 提供了三个容器适配器, 它们封装了其他 STL 容器, 并针对定制的情况公开了特殊的接口;
 * *这些适配器有栈(stack) || 队列(queue) || 优先级队列(priority queue)
 * 
 * 栈是一种具有两个基本操作(push 和 pop)的数据结构;
 * *当将一个元素压入(push)栈时, 其实是将元素插入栈的末端;
 * *当从栈中弹出(pop)一个元素时, 其实是从栈的末端移除了元素,
 * 这种方式称为后进先出: 最后被压入栈的元素是第一个被弹出的元素.
 * 
 * STL 在＜stack＞头文件中提供了 std::stack, 类模板 stack 接受两个模板参数,
 * 第一个是被包装容器的底层类型, 如 int;
 * 第二个是被包装容器的类型, 如 deque 或vector, 是可选的, 默认为 deque;
 * 
 * 要构造栈,可以传递要封装的双端队列、向量或列表的引用,
 * 这样栈将其操作(例如push 和 pop)转换为底层容器可以理解的方法, 例如 push_back 和 pop_back;
 * ?如果不提供构造函数参数, 则栈默认使用 deque, 第二个模板参数必须匹配这个容器的类型.
 * 
 */

// -----------------------------------
int main(int argc, const char **argv)
{
    printf("std::stack supports push/pop/top operations\n");

    std::vector<int> vec{1, 3};

    std::stack<int, decltype(vec)> easy_as(vec);
    assert(easy_as.top() == 3);
    easy_as.pop();
    easy_as.push(2);
    assert(easy_as.top() == 2);

    easy_as.pop();
    assert(easy_as.top() == 1);

    easy_as.pop();
    assert(easy_as.empty());

    return 0;
}
