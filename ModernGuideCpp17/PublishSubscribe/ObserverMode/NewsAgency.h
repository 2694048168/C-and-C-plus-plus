/**
 * @file NewsAgency.h
 * @author Wei Li (weili_yzzcq@163.com)
 * @brief 
 * @version 0.1
 * @date 2024-09-16
 * 
 * @copyright Copyright (c) 2024
 * 
 */

/* 
 * =====发布者
 * 1. 添加订阅者，将所有的订阅者存储起来
 * 2. 删除订阅者，将其从订阅者列表中删除
 * 3. 将消息发送给订阅者（发通知）
 * =====发布者的抽象类
 */

#pragma once

#include <list>
#include <string>

// 声明订阅者类, 只是做了声明, 并没有包含这个类的头文件
class Observer;

// 新闻社
class NewsAgency
{
public:
    void attach(Observer *ob);
    void deatch(Observer *ob);

    // 将通知信息发送给list 容器中的所有订阅者
    virtual void notify(std::string msg) = 0;
    virtual ~NewsAgency() {};

protected:
    // 订阅者列表
    std::list<Observer *> m_list;
};

// 摩根斯的新闻社
class Morgans : public NewsAgency
{
public:
    void notify(std::string msg) override;
};

// 八卦新闻
class Gossip : public NewsAgency
{
public:
    void notify(std::string msg) override;
};
