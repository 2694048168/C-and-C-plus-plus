#include <iostream>
#include <vector>
#include <string>
#include <algorithm>

template <typename Type>
inline Type max(Type t1, Type t2)
{
    return t1 > t2 ? t1 : t2;
}

template <typename elemType>
inline elemType max(const std::vector<elemType> &vec)
{
    return *max_element(vec.begin(), vec.end());
}

template <typename arrayType>
inline arrayType max(const arrayType *parray, int size)
{
    return *max_element(parray, parray+size);
}


int main(int argc, char**argv)
{
    std::string sarray[] = {"we", "were", "her", "pride", "of", "ten"};
    std::vector<std::string> svec(sarray, sarray+6);

    int iarray[] = {12, 70, 2, 169, 1, 5, 29};
    std::vector<int> ivec(iarray, iarray+7);

    float farray[] = {2.5, 24.8, 18.7, 4.1, 23.9};
    std::vector<float> fvec(farray, farray+5);

    int imax = max(max(ivec), max(iarray, 7));
    float fmax = max(max(fvec), max(farray, 5));
    std::string smax = max(max(svec), max(sarray, 6));

    std::cout << "imax should be 169 -- found: " << imax << '\n'
              << "fmax should be 24.8 -- found: " << fmax << '\n'
              << "smax should be were -- found: " << smax << '\n';

    return 0;
}