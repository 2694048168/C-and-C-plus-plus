/**
 * @file 09_for_range.cpp
 * @author Wei Li (weili_yzzcq@163.com)
 * @brief 
 * @version 0.1
 * @date 2024-08-29
 * 
 * @copyright Copyright (c) 2024
 * 
 */

/** 在C++98/03中,不同的容器和数组遍历的方式不尽相同,写法不统一,也不够简洁,
 *  而C++11基于范围的for循环可以以简洁、统一的方式来遍历容器和数组,用起来也更方便了.
 * 
 * 
 * 
 */

#include <iostream>
#include <map>
#include <set>
#include <vector>

// -----------------------------------
int main(int argc, const char **argv)
{
    // 在基于范围的for循环中,不需要再传递容器的两端,循环会自动以容器为范围展开,
    // 并且循环中也屏蔽掉了迭代器的遍历细节,直接抽取容器中的元素进行运算,
    // 使用这种方式进行循环遍历会让编码和维护变得更加简便
    std::vector<int> t{1, 2, 3, 4, 5, 6};
    for (auto it = t.begin(); it != t.end(); ++it)
    {
        std::cout << *it << " ";
    }
    std::cout << std::endl;

    for (const auto &value : t)
    {
        std::cout << value << " ";
    }
    std::cout << std::endl;
    // 是将容器中遍历的当前元素拷贝到了声明的变量value中,因此无法对容器中的元素进行写操作
    // 如果需要在遍历过程中修改元素的值，需要使用引用.

    std::cout << "遍历修改之前的容器: ";
    for (auto &value : t)
    {
        std::cout << value++ << " ";
    }
    std::cout << std::endl << "遍历修改之后的容器: ";

    for (auto &value : t)
    {
        std::cout << value << " ";
    }
    std::cout << std::endl;

    std::cout << "=============================\n";
    std::map<int, std::string> m{
        {1, "lucy"},
        {2, "lily"},
        {3,  "tom"}
    };

    // 基于范围的for循环方式
    // 使用普通的for循环方式（基于迭代器）遍历关联性容器,
    // auto自动推导出的是一个迭代器类型,需要使用迭代器的方式取出元素中的键值对（和指针的操作方法相同）
    for (auto &it : m)
    {
        std::cout << "id: " << it.first << ", name: " << it.second << std::endl;
    }

    // 普通的for循环方式
    // 使用基于范围的for循环遍历关联性容器,auto自动推导出的类型是容器中的value_type,
    // 相当于一个对组（std::pair）对象，提取键值对的方式如下
    for (auto it = m.begin(); it != m.end(); ++it)
    {
        std::cout << "id: " << it->first << ", name: " << it->second << std::endl;
    }

    // !元素只读, 对应set容器来说内部元素都是只读的,
    // *这是由容器的特性决定的，因此在for循环中auto&会被视为const auto & 
    std::set<int> st{1, 2, 3, 4, 5, 6};
    for (auto &item : st)
    {
        // std::cout << item++ << std::endl; // error, 不能给常量赋值
        std::cout << item << " ";
    }
    std::cout << "\n";

    // 在遍历关联型容器时也会出现同样的问题，基于范围的for循环中,
    // 虽然可以得到一个std::pair引用，但是我们是不能修改里边的first值的，也就是key值
    std::map<int, std::string> m_const{
        {1, "lucy"},
        {2, "lily"},
        {3,  "tom"}
    };

    for (auto &item : m_const)
    {
        // item.first 是一个常量
        // std::cout << "id: " << item.first++ << ", name: " << item.second << std::endl; // error
        std::cout << "id: " << item.first << ", name: " << item.second << std::endl; // error
    }

    // 访问次数
    // 基于范围的for循环遍历的对象可以是一个表达式或者容器/数组等.
    // 假设对一个容器进行遍历，在遍历过程中for循环对这个容器的访问频率是一次还是多次呢？
    std::vector<int> vec{1, 2, 3, 4, 5, 6};

    auto getRange = [&vec]() -> std::vector<int> &
    {
        std::cout << "get vector range..." << std::endl;
        return vec;
    };

    for (auto val : getRange())
    {
        std::cout << val << " ";
    }
    std::cout << std::endl;
    /* 对应基于范围的for循环来说,冒号后边的表达式只会被执行一次,
      在得到遍历对象之后会先确定好迭代的范围, 基于这个范围直接进行遍历.
      如果是普通的for循环, 在每次迭代的时候都需要判断是否已经到了结束边界.
     */

    return 0;
}
