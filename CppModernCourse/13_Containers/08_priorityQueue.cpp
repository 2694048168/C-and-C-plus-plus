/**
 * @file 08_priorityQueue.cpp
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
 * @brief 优先级队列(priority queue)也称为堆(heap),
 * 是一种支持 push 和 pop 操作的数据结构, 
 * *可以根据用户指定的比较器对象保持元素有序.
 * 比较器对象是一个可接受两个参数的函数对象, 如果第一个参数小于第二个参数, 则返回 true;
 * ?当从优先级队列中弹出一个元素时, 会根据比较器对象删除最大的元素;
 * 
 * =====STL 在＜queue＞头文件中提供了 std::priority_queue,有三个模板参数:
 * 1. 被包装容器的底层类型;
 * 2. 被包装容器的类型;
 * 3. 比较器对象的类型;
 * !只有底层类型是强制性的, 被包装容器的类型默认为 vector; 比较器对象的类型默认为 std::less.
 * 
 * NOTE: std::less 类模板可从＜functional＞头文件中获得, 如果第一个参数小于第二个参数, 则返回 true.
 * 
 * priority_queue 具有与 stack 相同的接口,
 * 唯一的区别是, stack 根据后进先出的方式弹出元素,
 * ?而 priority_queue 根据比较器比较结果决定弹出元素;
 * 
 */

// -----------------------------------
int main(int argc, const char **argv)
{
    printf("std::priority_queue supports push/pop\n");

    std::priority_queue<double> pri_que;
    pri_que.push(1.0); // 1.0
    pri_que.push(2.0); // 2.0 1.0
    pri_que.push(1.5); // 2.0 1.5 1.0

    printf("the top value of priority queue: %f\n", pri_que.top());
    pri_que.pop();     // 1.5 1.0
    pri_que.push(1.0); // 1.5 1.0 1.0

    printf("the top value of priority queue: %f\n", pri_que.top());
    pri_que.pop(); // 1.0 1.0

    printf("the top value of priority queue: %f\n", pri_que.top());
    pri_que.pop(); // 1.0
    pri_que.pop(); //
    assert(pri_que.empty());

    /**
     * @brief 优先级队列将元素保存在树结构中, 因此如果查看其底层容器,
     * *内存顺序将与代码中所暗示的顺序并不匹配.
     */

    return 0;
}
