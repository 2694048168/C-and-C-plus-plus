/**
 * @file Observer.h
 * @author Wei Li (weili_yzzcq@163.com)
 * @brief 
 * @version 0.1
 * @date 2024-09-16
 * 
 * @copyright Copyright (c) 2024
 * 
 */

#pragma once

#include "NewsAgency.h"

#include <iostream>
#include <string>

// 抽象的订阅者类
class Observer
{
public:
    // ?通过构造函数给观察者类提供一个信息的发布者
    Observer(std::string name, NewsAgency *news)
        : m_name(name)
        , m_news(news)
    {
        // 通过发布者对象将观察者对象存储了起来,这样就可以收到发布者推送的消息了
        m_news->attach(this);
    }

    // 观察者取消订阅,取消之后将不再接收订阅消息
    void unsubscribe()
    {
        m_news->deatch(this);
    }

    // 观察者得到最新消息之后,用于更新自己当前的状态
    virtual void update(std::string msg) = 0;

    virtual ~Observer() {}

protected:
    std::string m_name; // topic
    NewsAgency *m_news;
};

class Dragon : public Observer
{
public:
    // modern C++ 直接使用父类的构造函数等, 类似Python中super操作
    using Observer::Observer;

    void update(std::string msg) override
    {
        std::cout << "@@@路飞的老爸革命军龙收到消息: " << msg << std::endl;
    }
};

class Shanks : public Observer
{
public:
    using Observer::Observer;

    void update(std::string msg) override
    {
        std::cout << "@@@路飞的引路人红发香克斯收到消息: " << msg << std::endl;
    }
};

class Bartolomeo : public Observer
{
public:
    using Observer::Observer;

    void update(std::string msg) override
    {
        std::cout << "@@@路飞的头号粉丝巴托洛米奥收到消息: " << msg << std::endl;
    }
};
