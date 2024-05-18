/**
 * @file 09_moveAlgorithm.cpp
 * @author Wei Li (weili_yzzcq@163.com)
 * @brief 
 * @version 0.1
 * @date 2024-05-18
 * 
 * @copyright Copyright (c) 2024
 * 
 */

#include <algorithm>
#include <ios>
#include <iostream>
#include <vector>

/**
 * @brief move
 * move 算法将一个序列移动到另一个序列中,
 * 该算法移动目标序列并返回接收序列的结束迭代器.
 * !有责任确保目标序列表示的序列至少与源序列具有一样多的元素.
 * ?OutputIterator move([ep], ipt_begin, ipt_end, result);
 * 1. 可选的 std::execution 执行策略 ep(默认值为 std::execution::seq);
 * 2. 一对 InputIterator 对象, 即 ipt_begin/ipt_end，代表目标序列;
 * 3. 一个 InputIterator, 即 result, 代表要移入的序列的开头;
 * *复杂度
 * 线性复杂度, 该算法从目标序列移动元素 distance(ipt_begin, ipt_end) 次;
 * *其他要求
 * 1. 除非向左移动, 否则序列不得重叠;
 * 2. 类型必须是可移动的, 但不一定是可复制的;
 * 
 */
class MoveDetector
{
public:
    MoveDetector()
        : m_owner(true) {};
    MoveDetector(const MoveDetector &)            = delete;
    MoveDetector &operator=(const MoveDetector &) = delete;
    MoveDetector(MoveDetector &&o)                = delete;

    MoveDetector &operator=(MoveDetector &&o)
    {
        o.m_owner = false;
        m_owner   = true;
        return *this;
    }

    bool m_owner;
};

// -----------------------------------
int main(int argc, const char **argv)
{
    std::cout << "[====]std::move algorithm\n";
    std::vector<MoveDetector> detectors1(2);
    std::vector<MoveDetector> detectors2(2);

    std::move(detectors1.begin(), detectors1.end(), detectors2.begin());

    std::cout << "detectors1[0].owner: " << std::boolalpha << detectors1[0].m_owner << '\n';
    std::cout << "detectors1[1].owner: " << std::boolalpha << detectors1[1].m_owner << '\n';
    std::cout << "detectors2[0].owner: " << std::boolalpha << detectors2[0].m_owner << '\n';
    std::cout << "detectors2[1].owner: " << std::boolalpha << detectors2[1].m_owner << '\n';

    /**
     * @brief move_backward 算法将一个序列反向移动到另一个序列中.
     * 该算法将序列 1 移动到序列 2 中并返回指向最后一个移动的元素的迭代器.
     * 元素向后移动, 但会以原来的顺序出现在目标序列中.
     * !有责任确保目标序列表示的序列至少有与源序列一样多的元素.
     * ?OutputIterator move_backward([ep], ipt_begin, ipt_end, result);
     * 1. 可选的 std::execution 执行策略 ep(默认值为 std::execution::seq);
     * 2. 一对 InputIterator 对象,即 ipt_begin/ipt_end, 代表目标序列;
     * 3. 一个 InputIterator, 即 result, 代表要移入的序列;
     * *复杂度
     * 线性复杂度, 该算法从目标序列移动元素 distance(ipt_begin, ipt_end) 次;
     * *其他要求
     * 1. 序列不得重叠;
     * 2. 类型必须是可移动的,但不一定是可复制的;
     * 
     */
    std::cout << "[====]std::move_backward algorithm\n";
    std::vector<MoveDetector> detectors3(2);
    std::vector<MoveDetector> detectors4(2);

    std::move_backward(detectors3.begin(), detectors3.end(), detectors4.end());

    std::cout << "detectors3[0].owner: " << std::boolalpha << detectors3[0].m_owner << '\n';
    std::cout << "detectors3[1].owner: " << std::boolalpha << detectors3[1].m_owner << '\n';
    std::cout << "detectors4[0].owner: " << std::boolalpha << detectors4[0].m_owner << '\n';
    std::cout << "detectors4[1].owner: " << std::boolalpha << detectors4[1].m_owner << '\n';

    return 0;
}
