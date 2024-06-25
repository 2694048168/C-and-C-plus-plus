#include "MessageCenter.hpp"
#include "Subscriber.hpp"

#include <map>
#include <string>

class AppSubscriber : public Subscriber
{
    typedef void (*HandlerFun)(void *);

public:
    AppSubscriber();

public:
    void Subscribe(const std::string &Topic) override;
    void UnSubscribe(const std::string &Topic) override;
    void HandleEvent(const std::string &Topic, void *message) override;

private:
    static void HandleEvent_Person(void *message);
    static void HandleEvent_Other(void *message);

private:
    //TopicKey:HandlerFun
    std::map<std::string, HandlerFun> HandlerMap;
};
