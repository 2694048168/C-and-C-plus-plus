/**
 * @file main.cpp
 * @author Wei Li (weili_yzzcq@163.com)
 * @brief 
 * @version 0.1
 * @date 2023-12-04
 * 
 * @copyright Copyright (c) 2023
 * 
 */

#include <chrono>
#include <iomanip>
#include <iostream>
#include <string>
#include <string_view>
#include <vector>

/**
 * @brief  Meeting time for multiple time zones
 * 
 * Write a function that, given a list of meeting participants 
 * and their time zones, displays the local meeting time for each participant.
 * 
 */

/**
 * @brief Solution:
------------------------------------------------------ */
struct user
{
    std::string                   Name;
    const std::chrono::time_zone *Zone;

    user(std::string_view name, std::string_view zone)
        : Name{name.data()}
        , Zone(std::chrono::locate_zone(zone.data()))
    {
    }
};

template<class Duration, class TimeZonePtr>
void print_meeting_times(const std::chrono::zoned_time<Duration, TimeZonePtr> &time, const std::vector<user> &users)
{
    std::cout << std::left << std::setw(15) << std::setfill(' ') << "Local time: " << time << std::endl;

    for (const auto &user : users)
    {
        std::cout << std::left << std::setw(15) << std::setfill(' ') << user.Name
                  << std::chrono::zoned_time<Duration, TimeZonePtr>(user.Zone, time) << std::endl;
    }
}

// ------------------------------
int main(int argc, char **argv)
{
    std::vector<user> users{
        user{"Ildiko",  "Europe/Budapest"},
        user{  "Jens",    "Europe/Berlin"},
        user{  "Jane", "America/New_York"}
    };

    unsigned int h, m;
    std::cout << "Hour:";
    std::cin >> h;
    std::cout << "Minutes:";
    std::cin >> m;

    std::chrono::year_month_day today = std::chrono::floor<std::chrono::days>(std::chrono::system_clock::now());

    auto localtime = std::chrono::zoned_time<std::chrono::minutes>(
        std::chrono::current_zone(),
        static_cast<std::chrono::local_days>(today) + std::chrono::hours{h} + std::chrono::minutes{m});

    print_meeting_times(localtime, users);

    return 0;
}
