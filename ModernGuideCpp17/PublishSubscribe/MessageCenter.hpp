/**
 * @file MessageCenter.hpp
 * @author Wei Li (weili_yzzcq@163.com)
 * @brief 
 * @version 0.1
 * @date 2024-06-25
 * 
 * @copyright Copyright (c) 2024
 * 
 */

#pragma once

#include <list>
#include <map>
#include <mutex>
#include <thread>

class MessageCenter
{
public:
    static MessageCenter *getInstance();
    MessageCenter(const MessageCenter &t)            = delete;
    MessageCenter &operator=(const MessageCenter &t) = delete;

    void Run();
    void RegisterPublish(const std::string &tpcKey, void *message, unsigned int datasize);
    void RegisterSubscribe(const std::string &tpcKey, class Subscriber *subscriber);
    void CancelSubscribe(const std::string &tpcKey, class Subscriber *subscriber);

private:
    MessageCenter();
    virtual ~MessageCenter();
    void CoreProcess();

private:
    //topic-key:Publish Data,只关心数据，不关心是谁发布的
    std::map<std::string, std::list<void *>> mPublisher;

    //topic-key:Subscribers  只关心订阅者(其后续会处理订阅的消息)
    std::map<std::string, std::list<class Subscriber *>> mSubscriber;

    //核心线程，维护发布数据队列 + 订阅触发处理
    std::unique_ptr<std::thread> mCoreProcess;
    //发布数据队列修改时的保护锁
    std::mutex                   mPublishMutex;
    //订阅者注册/取消订阅时的保护锁
    std::mutex                   mSubscribeMutex;

    static MessageCenter *mSgMC;    //消息中心单例对象
    static std::mutex     mMCMutex; //线程安全单例保护锁
};
