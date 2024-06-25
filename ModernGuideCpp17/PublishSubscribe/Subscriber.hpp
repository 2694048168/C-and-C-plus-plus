/**
 * @file Subscriber.hpp
 * @author Wei Li (weili_yzzcq@163.com)
 * @brief 
 * @version 0.1
 * @date 2024-06-25
 * 
 * @copyright Copyright (c) 2024
 * 
 */

#pragma once

#include <string>

class Subscriber
{
public:
    virtual void Subscribe(const std::string &Topic)                  = 0;
    virtual void UnSubscribe(const std::string &Topic)                = 0;
    virtual void HandleEvent(const std::string &Topic, void *message) = 0;
};
