#include <iostream>
#include <vector>
#include <string>

bool calc_elements(std::vector<int> &vec, int pos);
void display_elems(std::vector<int> &vec, const std::string &title, std::ostream &os = std::cout);


int main(int argc, char**argv)
{
    std::vector<int> pent;
    const std::string title("Pentagonal Numeric Series");

    if (calc_elements(pent, 0))
    {
        display_elems(pent, title);
    }

    if (calc_elements(pent, 8))
    {
        display_elems(pent, title);
    }

    if (calc_elements(pent, 14))
    {
        display_elems(pent, title);
    }

    if (calc_elements(pent, 138))
    {
        display_elems(pent, title);
    }

    return 0;
}


bool calc_elements(std::vector<int> &vec, int pos)
{
    if (pos <=0 || pos > 64)
    {
        std::cerr << "Sorry. Invalid position: " << pos << std::endl;
        return false;
    }

    for (int ix = vec.size() + 1; ix <= pos; ++ix)
    {
        vec.push_back( (ix*(3*ix-1))/2 );
    }
    
    return true;
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
