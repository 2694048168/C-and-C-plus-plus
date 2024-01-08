/**
 * @file main.cpp
 * @author Wei Li (weili_yzzcq@163.com)
 * @brief Customer service system
 * @version 0.1
 * @date 2024-01-08
 * 
 * @copyright Copyright (c) 2024
 * 
 */

#include <array>
#include <chrono>
#include <condition_variable>
#include <iostream>
#include <mutex>
#include <queue>
#include <random>
#include <string>
#include <string_view>
#include <thread>
#include <vector>

/**
 * @brief Customer service system
 * 
 * Write a program that simulates the way customers are served in an office.
 *  The office has three desks where customers can be served at the same time. 
 * Customers can enter the office at any time. They take a ticket with a service number
 *  from a ticketing machine and wait until their number is next for service at one of
 *  the desks. Customers are served in the order they entered the office, 
 * or more precisely, in the order given by their ticket. 
 * Every time a service desk finishes serving a customer, 
 * the next customer in order is served. The simulation should stop after 
 * a particular number of customers have been issued tickets and served.
 * 
 */

/**
 * @brief Solution:
------------------------------------------------------ */
class Logger
{
protected:
    Logger() {}

public:
    static Logger &instance()
    {
        static Logger lg;
        return lg;
    }

    Logger(const Logger &)            = delete;
    Logger &operator=(const Logger &) = delete;

    void log(std::string_view message)
    {
        std::lock_guard<std::mutex> lock(mutex_);
        std::cout << "LOG: " << message << std::endl;
    }

private:
    std::mutex mutex_;
};

class ticketing_machine
{
public:
    ticketing_machine(const int start)
        : last_ticket(start)
        , first_ticket(start)
    {
    }

    int next()
    {
        return last_ticket++;
    }

    int last() const
    {
        return last_ticket - 1;
    }

    void reset()
    {
        last_ticket = first_ticket;
    }

private:
    int first_ticket;
    int last_ticket;
};

class customer
{
public:
    customer(const int no)
        : number(no)
    {
    }

    int ticket_number() const noexcept
    {
        return number;
    }

private:
    int number;

    friend bool operator<(const customer &l, const customer &r);
};

bool operator<(const customer &l, const customer &r)
{
    return l.number > r.number;
}

// ------------------------------
int main(int argc, char **argv)
{
    std::priority_queue<customer> customers;
    bool                          store_open = true;
    std::mutex                    mt;
    std::condition_variable       cv;

    std::vector<std::thread> desks;
    for (int i = 1; i <= 3; ++i)
    {
        desks.emplace_back(
            [i, &store_open, &mt, &cv, &customers]()
            {
                std::random_device rd;
                auto               seed_data = std::array<int, std::mt19937::state_size>{};
                std::generate(std::begin(seed_data), std::end(seed_data), std::ref(rd));
                std::seed_seq                   seq(std::begin(seed_data), std::end(seed_data));
                std::mt19937                    eng(seq);
                std::uniform_int_distribution<> ud(2000, 3000);

                Logger::instance().log("desk " + std::to_string(i) + " open");

                while (store_open || !customers.empty())
                {
                    std::unique_lock<std::mutex> locker(mt);

                    cv.wait_for(locker, std::chrono::seconds(1), [&customers]() { return !customers.empty(); });

                    if (!customers.empty())
                    {
                        auto const c = customers.top();
                        customers.pop();

                        Logger::instance().log("[-] desk " + std::to_string(i) + " handling customer "
                                               + std::to_string(c.ticket_number()));

                        Logger::instance().log("[=] queue size: " + std::to_string(customers.size()));

                        locker.unlock();
                        cv.notify_one();

                        std::this_thread::sleep_for(std::chrono::milliseconds(ud(eng)));

                        Logger::instance().log("[ ] desk " + std::to_string(i) + " done with customer "
                                               + std::to_string(c.ticket_number()));
                    }
                }

                Logger::instance().log("desk " + std::to_string(i) + " closed");
            });
    }

    std::thread store(
        [&store_open, &customers, &mt, &cv]()
        {
            ticketing_machine  tm(100);
            std::random_device rd;
            auto               seed_data = std::array<int, std::mt19937::state_size>{};
            std::generate(std::begin(seed_data), std::end(seed_data), std::ref(rd));
            std::seed_seq                   seq(std::begin(seed_data), std::end(seed_data));
            std::mt19937                    eng(seq);
            std::uniform_int_distribution<> ud(200, 500);

            for (int i = 1; i <= 25; ++i)
            {
                customer c(tm.next());
                customers.push(c);

                Logger::instance().log("[+] new customer with ticket " + std::to_string(c.ticket_number()));
                Logger::instance().log("[=] queue size: " + std::to_string(customers.size()));

                cv.notify_one();

                std::this_thread::sleep_for(std::chrono::milliseconds(ud(eng)));
            }

            store_open = false;
        });

    store.join();

    for (auto &desk : desks)
    {
        desk.join();
    }

    return 0;
}
