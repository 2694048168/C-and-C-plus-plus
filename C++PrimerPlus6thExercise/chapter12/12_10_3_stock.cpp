#include "12_10_3_stock.hpp"

#include <cstring>
#include <iostream>


// // constructors
// Stock::Stock() // default constructor
// {
//     shares    = 0;
//     share_val = 0.0;
//     total_val = 0.0;
// }

// Stock::Stock(const std::string &co, long n, double pr)
// {
//     company = co;
//     if (n < 0)
//     {
//         std::cout << "Number of shares can't be negative; " << company << " shares set to 0.\n";
//         shares = 0;
//     }
//     else
//         shares = n;
//     share_val = pr;
//     set_tot();
// }

// // class destructor
// Stock::~Stock() // quiet class destructor
// {
// }

// // other methods
// void Stock::buy(long num, double price)
// {
//     if (num < 0)
//     {
//         std::cout << "Number of shares purchased can't be negative. "
//                   << "Transaction is aborted.\n";
//     }
//     else
//     {
//         shares += num;
//         share_val = price;
//         set_tot();
//     }
// }

// void Stock::sell(long num, double price)
// {
//     using std::cout;
//     if (num < 0)
//     {
//         cout << "Number of shares sold can't be negative. "
//              << "Transaction is aborted.\n";
//     }
//     else if (num > shares)
//     {
//         cout << "You can't sell more than you have! "
//              << "Transaction is aborted.\n";
//     }
//     else
//     {
//         shares -= num;
//         share_val = price;
//         set_tot();
//     }
// }

// void Stock::update(double price)
// {
//     share_val = price;
//     set_tot();
// }

// /*
// void Stock::show() const
// {
//     using std::cout;
//     using std::ios_base;
//     // set format to #.###
//     ios_base::fmtflags orig = cout.setf(ios_base::fixed, ios_base::floatfield);
//     std::streamsize    prec = cout.precision(3);

//     cout << "Company: " << company << "  Shares: " << shares << '\n';
//     cout << "  Share Price: $" << share_val;
//     // set format to #.##
//     cout.precision(2);
//     cout << "  Total Worth: $" << total_val << '\n';

//     // restore original format
//     cout.setf(orig, ios_base::floatfield);
//     cout.precision(prec);
// }
// */
// std::ostream &operator<<(std::ostream &os, const Stock &st)
// {
//     // set format to #.###
//     std::ios_base::fmtflags orig = os.setf(std::ios_base::fixed, std::ios_base::floatfield);
//     std::streamsize         prec = os.precision(3);

//     os << "Company: " << st.company << "  Shares: " << st.shares << '\n';
//     os << "  Share Price: $" << st.share_val;
//     // set format to #.##
//     os.precision(2);
//     os << "  Total Worth: $" << st.total_val << '\n';

//     // restore original format
//     os.setf(orig, std::ios_base::floatfield);
//     os.precision(prec);

//     return os;
// }

// const Stock &Stock::top_val(const Stock &s) const
// {
//     if (s.total_val > total_val)
//         return s;
//     else
//         return *this;
// }

// -------------------------------------------------------

Stock::Stock()
{
    company   = new char[8];
    company   = std::strcpy(company, "no name");
    shares    = 0;
    share_val = 0.0;
    total_val = 0.0;
}

Stock::Stock(const char *co, long n, double pr)
{
    int len = std::strlen(co);
    company = new char[len + 1];
    company = std::strcpy(company, co);
    if (n < 0)
    {
        std::cout << "Number of shares con't be negative; " << company << " shares set to 0.\n";
        shares = 0;
    }
    else
        shares = n;
    share_val = pr;
    set_tot();
}

Stock::Stock(const Stock &st)
{
    int len = std::strlen(st.company);
    company = new char[len + 1];
    company   = std::strcpy(company, st.company);
    shares    = st.shares;
    share_val = st.share_val;
    total_val = st.total_val;
}

Stock::~Stock()
{
    delete[] company;
}

void Stock::buy(long num, double price)
{
    if (num < 0)
    {
        std::cout << "Number of shares purchased can't be negative. "
                  << "Transaction is aborted.\n";
    }
    else
    {
        shares += num;
        share_val = price;
        set_tot();
    }
}

void Stock::sell(long num, double price)
{
    using std::cout;
    if (num < 0)
    {
        std::cout << "Number of shares sold can't be negative. "
                  << "Transaction is aborted.\n";
    }
    else if (num > shares)
    {
        cout << "You can't sell more than you have! "
             << "Transaction is aborted.\n";
    }
    else
    {
        shares -= num;
        share_val = price;
        set_tot();
    }
}

void Stock::update(double price)
{
    share_val = price;
    set_tot();
}

std::ostream &operator<<(std::ostream &os, const Stock &st)
{
    using std::ios_base;
    ios_base::fmtflags orig = os.setf(ios_base::fixed, ios_base::floatfield);
    std::streamsize    prec = os.precision(3);
    os << "Company: " << st.company << " Shares: " << st.shares << '\n';
    os << " Share Price: $" << st.share_val;
    os.precision(2);
    os << " Total Worth: $" << st.total_val << '\n';
    os.setf(orig, ios_base::floatfield);
    os.precision(prec);

    return os;
}

const Stock &Stock::top_val(const Stock &s) const
{
    if (s.total_val > total_val)
        return s;
    else
        return *this;
}

// -------------------------------------------------------