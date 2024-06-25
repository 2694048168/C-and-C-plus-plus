#include "AppSubscriber.hpp"

#include "TopicMessage.hpp"

#include <chrono>
#include <ctime>
#include <iostream>

AppSubscriber::AppSubscriber()
{
    HandlerMap.clear();
    HandlerMap["Person"] = HandleEvent_Person;
    HandlerMap["Other"]  = HandleEvent_Other;
}

void AppSubscriber::Subscribe(const std::string &Topic)
{
    MessageCenter::getInstance()->RegisterSubscribe(Topic, this);
}

void AppSubscriber::UnSubscribe(const std::string &Topic)
{
    MessageCenter::getInstance()->CancelSubscribe(Topic, this);
}

void AppSubscriber::HandleEvent(const std::string &Topic, void *message)
{
    if (HandlerMap.find(Topic) != HandlerMap.end())
        HandlerMap[Topic](message);
}

void AppSubscriber::HandleEvent_Person(void *message)
{
    struct Person *dt = (struct Person *)message;

    // 获取当前系统时间
    auto        now = std::chrono::system_clock::now();
    // 转换为 time_t
    std::time_t now_time = std::chrono::system_clock::to_time_t(now);
    std::cout << dt->name << dt->age << std::ctime(&now_time) << std::endl;
}

void AppSubscriber::HandleEvent_Other(void *message)
{
    struct Other *dt = (struct Other *)message;
    //do something
}
