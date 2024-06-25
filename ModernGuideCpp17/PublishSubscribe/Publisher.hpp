#pragma once

/**
 * @file Publisher.hpp
 * @author Wei Li (weili_yzzcq@163.com)
 * @brief 
 * @version 0.1
 * @date 2024-06-25
 * 
 * @copyright Copyright (c) 2024
 * 
 */

#include <string>

class Publisher
{
public:
    // virtual void Publish(const std::string &Topic, void *message, unsigned int datasize) = 0;
    virtual void Publish(const std::string &Topic, void *message, unsigned int datasize);
};
