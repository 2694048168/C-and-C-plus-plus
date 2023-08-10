#include "11_9_6_stone_weight_overload.hpp"

#include <iostream>

// construct StoneWeight object from double value
StoneWeight::StoneWeight(double lbs)
{
    stone    = int(lbs) / Lbs_per_stn; // integer division
    pds_left = int(lbs) % Lbs_per_stn + lbs - int(lbs);
    pounds   = lbs;
}

// construct StoneWeight object from stone, double values
StoneWeight::StoneWeight(int stn, double lbs)
{
    stone    = stn;
    pds_left = lbs;
    pounds   = stn * Lbs_per_stn + lbs;
}

StoneWeight::StoneWeight() // default constructor, wt = 0
{
    stone = pounds = pds_left = 0;
}

StoneWeight::~StoneWeight() // destructor
{
}

void StoneWeight::set_status(StoneWeight::STATUS status)
{
    this->status = status;
}

double StoneWeight::get_pounds()
{
    return this->pounds;
}

StoneWeight StoneWeight::operator+(const StoneWeight &s)
{
    if (s.status < 0 || s.status >= StoneWeight::STATUS::NUM_STATUS)
    {
        std::cout << "This format for STATUS is NOT implement\n";
    }

    if (s.status == StoneWeight::integer_pounds_form)
    {
        this->pounds += s.pounds;
        this->stone += int(s.pounds) / Lbs_per_stn; // integer division
        this->pds_left += int(s.pounds) % Lbs_per_stn + s.pounds - int(s.pounds);
    }
    else if (s.status == StoneWeight::floating_point_pounds_form)
    {
        this->stone += s.stone;
        this->pds_left += s.pds_left;
        this->pounds += s.stone * Lbs_per_stn + s.pds_left;
    }

    return *this;
}

StoneWeight StoneWeight::operator-(const StoneWeight &s)
{
    if (s.status < 0 || s.status >= StoneWeight::STATUS::NUM_STATUS)
    {
        std::cout << "This format for STATUS is NOT implement\n";
    }

    if (s.status == StoneWeight::integer_pounds_form)
    {
        this->pounds -= s.pounds;
        this->stone -= int(s.pounds) / Lbs_per_stn; // integer division
        this->pds_left -= int(s.pounds) % Lbs_per_stn + s.pounds - int(s.pounds);
    }
    else if (s.status == StoneWeight::floating_point_pounds_form)
    {
        this->stone -= s.stone;
        this->pds_left -= s.pds_left;
        this->pounds -= s.stone * Lbs_per_stn + s.pds_left;
    }

    return *this;
}

StoneWeight StoneWeight::operator*(double val)
{
    if (this->status < 0 || this->status >= StoneWeight::STATUS::NUM_STATUS)
    {
        std::cout << "This format for STATUS is NOT implement\n";
    }

    if (this->status == StoneWeight::integer_pounds_form)
    {
        this->pounds *= val;
        this->stone    = int(this->pounds) / Lbs_per_stn; // integer division
        this->pds_left = int(this->pounds) % Lbs_per_stn + this->pounds - int(this->pounds);
    }
    else if (this->status == StoneWeight::floating_point_pounds_form)
    {
        this->stone *= val;
        this->pds_left *= val;
        this->pounds -= this->stone * Lbs_per_stn + this->pds_left;
    }

    return *this;
}

std::ostream &operator<<(std::ostream &os, const StoneWeight &s)
{
    if (s.status == StoneWeight::integer_pounds_form)
    {
        os << s.pounds << " pounds\n"
           << "----------------------------------------\n";
        return os;
    }
    else if (s.status == StoneWeight::floating_point_pounds_form)
    {
        os << s.stone << " stone, " << s.pds_left << " pounds\n"
           << "----------------------------------------\n";
        return os;
    }
    else
    {
        std::cout << "This format is NOT implement\n";
        return os;
    }
}

// 关系运算符重载
bool StoneWeight::operator<(const StoneWeight &s)
{
    if (this->pounds < s.pounds)
    {
        return true;
    }
    return false;
}

bool StoneWeight::operator<=(const StoneWeight &s)
{
    if (this->pounds < s.pounds || this->pounds == s.pounds)
    {
        return true;
    }
    return false;
}

bool StoneWeight::operator>(const StoneWeight &s)
{
    if (this->pounds > s.pounds)
    {
        return true;
    }
    return false;
}

bool StoneWeight::operator>=(const StoneWeight &s)
{
    if (this->pounds > s.pounds || this->pounds == s.pounds)
    {
        return true;
    }
    return false;
}

bool operator==(const StoneWeight &s1, const StoneWeight &s)
{
    if (s1.pounds == s.pounds)
    {
        return true;
    }
    return false;
}

bool operator!=(const StoneWeight &s1, const StoneWeight &s)
{
    if (s1.pounds == s.pounds)
    {
        return false;
    }
    return true;
}