/**
 * @file Mediator.cpp
 * @author Wei Li (weili_yzzcq@163.com)
 * @brief 
 * @version 0.1
 * @date 2025-01-02
 * 
 * @copyright Copyright (c) 2024
 * 
 * g++ Mediator.cpp -std=c++20
 * clang++ Mediator.cpp -std=c++20
 * 
 */

#include <iostream>
#include <string>
#include <vector>

class Colleague;

class Mediator
{
public:
    virtual ~Mediator() = default;

    virtual void sendMessage(const std::string &msg, Colleague *colleague) = 0;
    virtual void addColleague(Colleague *colleague)                        = 0;
};

class Colleague
{
public:
    Colleague(Mediator *mediator)
        : mediator_(mediator)
    {
    }

    virtual ~Colleague() = default;

    virtual void sendMessage(const std::string &message)    = 0;
    virtual void receiveMessage(const std::string &message) = 0;

protected:
    Mediator *mediator_;
};

class ConcreteMediator : public Mediator
{
public:
    void sendMessage(const std::string &msg, Colleague *colleague) override
    {
        for (auto col : colleagues_)
        {
            if (col != colleague)
            {
                col->receiveMessage(msg);
            }
        }
    }

    void addColleague(Colleague *colleague) override
    {
        colleagues_.push_back(colleague);
    }

private:
    std::vector<Colleague *> colleagues_;
};

class ConcreteColleague : public Colleague
{
public:
    ConcreteColleague(Mediator *mediator)
        : Colleague(mediator)
    {
    }

    void sendMessage(const std::string &message) override
    {
        mediator_->sendMessage(message, this);
    }

    void receiveMessage(const std::string &message) override
    {
        std::cout << "Received message: " << message << std::endl;
    }
};

// Demo1: 基于中介者模式实现的消息群发功能
class User;

class UserMediator
{
public:
    virtual void sendMessage(const std::string &message, User *user) = 0;
    virtual void addUser(User *user)                                 = 0;

    virtual ~UserMediator() = default;
};

class User
{
public:
    User(const std::string &name, UserMediator *mediator)
    {
        this->name     = name;
        this->mediator = mediator;
    }

    virtual ~User() = default;

    const std::string &getName() const
    {
        return name;
    }

    void sendMessage(const std::string &message)
    {
        mediator->sendMessage(message, this);
    }

    virtual void receiveMsg(const std::string &message) = 0;

private:
    std::string   name;
    UserMediator *mediator;
};

class ChatRoom : public UserMediator
{
public:
    void addUser(User *user) override
    {
        users.push_back(user);
    }

    void sendMessage(const std::string &message, User *sender) override
    {
        for (User *user : users)
        {
            if (user != sender)
            {
                user->receiveMsg(message);
            }
        }
    }

private:
    std::vector<User *> users;
};

class ChatUser : public User
{
public:
    ChatUser(const std::string &name, UserMediator *mediator)
        : User(name, mediator)
    {
    }

    void receiveMsg(const std::string &msg) override
    {
        std::cout << getName() << " received a message: " << msg << std::endl;
    }
};

// Demo2: 模拟的聊天室
struct ChatRoom_
{
    virtual void broadcast(std::string from, std::string msg)               = 0;
    virtual void message(std::string from, std::string to, std::string msg) = 0;
};

struct Person
{
    std::string              m_name;
    ChatRoom_               *m_room{nullptr};
    std::vector<std::string> m_chat_log;

    Person(std::string n)
        : m_name(n)
    {
    }

    void say(std::string msg) const
    {
        m_room->broadcast(m_name, msg);
    }

    void pm(std::string to, std::string msg) const
    {
        m_room->message(m_name, to, msg);
    }

    void receive(std::string from, std::string msg)
    {
        std::string s{from + ": \"" + msg + "\""};
        std::cout << "[" << m_name << "'s chat session]" << s << "\n";
        m_chat_log.emplace_back(s);
    }
};

struct GoogleChat : ChatRoom_
{
    std::vector<Person *> m_people;

    void broadcast(std::string from, std::string msg)
    {
        for (auto p : m_people)
            if (p->m_name != from)
                p->receive(from, msg);
    }

    void join(Person *p)
    {
        std::string join_msg = p->m_name + " joins the chat";
        broadcast("room", join_msg);
        p->m_room = this;
        m_people.push_back(p);
    }

    void message(std::string from, std::string to, std::string msg)
    {
        auto target = find_if(begin(m_people), end(m_people), [&](const Person *p) { return p->m_name == to; });
        if (target != end(m_people))
            (*target)->receive(from, msg);
    }
};

// --------------------------------------------------
int main(int /* argc */, const char ** /* argv */)
{
    std::cout << "--------------------------------------\n";
    Mediator  *mediator   = new ConcreteMediator();
    Colleague *colleague1 = new ConcreteColleague(mediator);
    Colleague *colleague2 = new ConcreteColleague(mediator);

    mediator->addColleague(colleague1);
    mediator->addColleague(colleague2);

    colleague1->sendMessage("Hello from colleague1");
    colleague2->sendMessage("Hello from colleague2");

    delete colleague1;
    delete colleague2;
    delete mediator;

    std::cout << "--------------------------------------\n";
    UserMediator *chatRoom = new ChatRoom();
    User         *user1    = new ChatUser("User1", chatRoom);
    User         *user2    = new ChatUser("User2", chatRoom);
    User         *user3    = new ChatUser("User3", chatRoom);

    chatRoom->addUser(user1);
    chatRoom->addUser(user2);
    chatRoom->addUser(user3);
    user1->sendMessage("Hello, everyone!");

    delete user1;
    delete user2;
    delete user3;
    delete chatRoom;

    std::cout << "--------------------------------------\n";
    GoogleChat room;
    Person     john{"John"};
    Person     jane{"Jane"};
    room.join(&john);
    room.join(&jane);

    john.say("hi room");
    jane.say("oh, hey john");
    Person simon{"Simon"};
    room.join(&simon);
    simon.say("hi everyone!");
    jane.pm("Simon", "glad you found us, simon!");

    return 0;
}
