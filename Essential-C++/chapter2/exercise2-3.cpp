#include <iostream>
#include <vector>
#include <string>

extern void really_calc_elems(std::vector<int> &, int);
inline bool calc_elems(std::vector<int> &vec, int pos)
{
    if (pos <=0 || pos > 64)
    {
        std::cerr << "Sorry. Invalid position: " << pos << std::endl;
        return false;
    }

    if (vec.size() < pos)
    {
        really_calc_elems(vec, pos);
    }
    
    return true;
}

void display_elems(std::vector<int> &vec, const std::string &title, std::ostream &os = std::cout);


int main(int argc, char**argv)
{
    std::vector<int> pent;
    const std::string title("Pentagonal Numeric Series");

    if (calc_elems(pent, 0))
    {
        display_elems(pent, title);
    }

    if (calc_elems(pent, 8))
    {
        display_elems(pent, title);
    }

    if (calc_elems(pent, 14))
    {
        display_elems(pent, title);
    }

    if (calc_elems(pent, 138))
    {
        display_elems(pent, title);
    }

    return 0;
}


void really_calc_elems(std::vector<int> &vec, int pos)
{
    for (size_t ix = vec.size() + 1; ix <= pos; ++ix)
    {
        vec.push_back( (ix*(3*ix-1))/2 );
    }
}


void display_elems(std::vector<int> &vec, const std::string &title, std::ostream &os)
{
    os << '\n' << title << "\n\t";
    for (size_t ix = 0; ix < vec.size(); ++ix)
    {
        os << vec[ix] << ' ';
    }
    os << std::endl;
}
