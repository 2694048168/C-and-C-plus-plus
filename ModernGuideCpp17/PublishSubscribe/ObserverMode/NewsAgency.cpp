#include "NewsAgency.h"

// 在头文件中只是对 Observer 类进行了声明定义了这种类型的指针变量,
// 在源文件中需要调用 Observer 类提供的 API, 所以必须要包含这个类的头文件.
// 这么处理的目的是为了避免两个相互关联的类他们的头文件相互包含.
#include "Observer.h" // 在源文件中包含该头文件

#include <iostream>

void NewsAgency::attach(Observer *ob)
{
    m_list.push_back(ob);
}

void NewsAgency::deatch(Observer *ob)
{
    m_list.remove(ob);
}

void Morgans::notify(std::string msg)
{
    std::cout << "摩根斯新闻社报纸的订阅者一共有<" << m_list.size() << ">人\n";
    for (const auto &item : m_list)
    {
        item->update(msg); // 订阅者类的定义在下面
    }
}

void Gossip::notify(std::string msg)
{
    std::cout << "八卦新闻社报纸的订阅者一共有<" << m_list.size() << ">人\n";
    for (const auto &item : m_list)
    {
        item->update(msg);
    }
}
