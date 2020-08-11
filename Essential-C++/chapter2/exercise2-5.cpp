#include <iostream>
#include <vector>
#include <string>
#include <algorithm>

inline int max(int t1, int t2)
{
    return t1 > t2 ? t1 : t2;
}

inline float max(float t1, float t2)
{
    return t1 > t2 ? t1 : t2;
}

inline std::string max(const std::string& t1, const std::string&  t2)
{
    return t1 > t2 ? t1 : t2;
}

inline int max(const std::vector<int> &vec)
{
    return *max_element(vec.begin(), vec.end());
}

inline float max(const std::vector<float> &vec)
{
    return *max_element(vec.begin(), vec.end());
}

inline std::string max (const std::vector<std::string> &vec)
{
    return *max_element(vec.begin(), vec.end());
}

inline int max(const int *parray, int size)
{
    return *max_element(parray, parray+size);
}

inline float max(const float *parray, int size)
{
    return *max_element(parray, parray+size);
}

inline std::string max(const std::string *parray, int size)
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