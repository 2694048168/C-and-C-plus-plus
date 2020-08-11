#include <iostream>
#include <vector>

inline bool check_validity(int pos)
{
    return (pos <= 0 || pos > 64) ? false : true;
}

const std::vector<int> *pentagonal_series(int pos)
{
    static std::vector<int> _elems;
    if (check_validity(pos) && (pos > _elems.size()))
    {
        for (size_t ix = _elems.size() + 1; ix <= pos; ++ix)
        {
            _elems.push_back( (ix*(3*ix-1))/2 );
        }
    }

    return &_elems;
}

bool pentagonal_elem(int pos, int &elem)
{
    if (! check_validity(pos))
    {
        std::cout << "Sorry. Invalid position: " << pos << std::endl;
        elem = 0;
        return false;
    }

    const std::vector<int> *pent = pentagonal_series(pos);
    elem = (*pent)[pos-1];
    return true;
}

int main(int argc, char **argv)
{
    int elem;
    if (pentagonal_elem(8, elem))
    {
        std::cout << "element 8 is " << elem << '\n';
    }

    if (pentagonal_elem(88, elem))
    {
        std::cout << "element 88 is " << elem << '\n';
    }

    if (pentagonal_elem(12, elem))
    {
        std::cout << "element 12 is " << elem << '\n';
    }

    if (pentagonal_elem(64, elem))
    {
        std::cout << "element 64 is " << elem << '\n';
    }
    
    return 0;
}