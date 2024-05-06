/**
 * @file 07_queue.cpp
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
#include <queue>

/**
 * @brief 队列(queue)是一种数据结构, 它与栈一样, 将 push 和 pop 作为其基本操作;
 * *队列是先进先出的, 当将一个元素推入队列时, 其实是将元素插入队列的末端;
 * *当从队列中弹出一个元素时, 其实是将元素从队列的开头移除;这样队列中停留时间最长的元素就是将被弹出的元素;
 *
 * STL 在＜queue＞头文件中提供了 std::queue, 队列也有两个模板参数;
 * 第一个参数是被包装容器的底层类型, 可选的第二个参数是被包装容器的类型, 默认为 deque;
 * 在 STL 容器中, 只能使用 deque 或 list 作为队列的底层容器, 
 * ?因为在 vector 的前面压入元素和从前面弹出元素是低效的. 
 * 
 * 使用 front 和 back 方法可以访问队列前面或后面的元素
 * 
 */

// ------------------------------------
int main(int argc, const char **argv)
{
    printf("std::queue supports push/pop/front/back\n");

    std::deque<int> deq{1, 2};
    std::queue<int> easy_as(deq);

    assert(easy_as.front() == 1);
    assert(easy_as.back() == 2);
    printf("The front value: %d\n", easy_as.front());
    printf("The back value: %d\n", easy_as.back());

    easy_as.pop();
    easy_as.push(3);
    assert(easy_as.front() == 2);
    assert(easy_as.back() == 3);

    easy_as.pop();
    assert(easy_as.front() == 3);
    easy_as.pop();
    assert(easy_as.empty());

    return 0;
}
