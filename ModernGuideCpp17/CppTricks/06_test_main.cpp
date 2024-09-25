#include "06_BigInteger.h"

#include <windows.h>

#include <cmath>
#include <cstdlib>
#include <fstream>

typedef long long Type;
const int         TESTS      = 80000;
const char        typefile[] = "type.txt";
const char        bigfile[]  = "big.txt";
const Type        ZeroType   = 0;
const BigInteger  ZeroBig    = 0;

Type       type[TESTS + 5];
BigInteger big[TESTS + 5];

void Rand(bool half = false, bool Sqrt = false);
void Output();
void Input();
void Absolute();
void Compare_Calculate();

// ---------------------------------------
int main(int argc, const char **argv)
{
    std::ios::sync_with_stdio(false);
    srand((unsigned)time(NULL));

    Rand();
    Output();
    Input();
    Absolute();
    Compare_Calculate();

    // ================================
    BigInteger encoder = 123456789123456789;
    encoder += 123456789123456789;
    std::cout << "\nThe result of BigInteger: " << encoder << std::endl;
    // ================================

    return 0;
}

void Rand(bool half, bool Sqrt)
{
    Type tmp;
    for (int i = 0; i < TESTS; ++i)
    {
        do
        {
            tmp = ((Type)rand()) << (rand() % (sizeof(Type) * 4));
            if (rand() % 2 == 1)
            {
                tmp *= ((Type)rand()) << (rand() % (sizeof(Type) * 4));
            }
            if (half)
            {
                tmp = abs(tmp / 2) - 1;
            }
            if (Sqrt)
            {
                if (abs(tmp) < 0)
                {
                    --tmp;
                }
                tmp = sqrt(abs(tmp)) - 1;
            }
            if (rand() % 2 == 1)
            {
                tmp = -tmp;
            }
        }
        while (tmp == 0);
        type[i] = tmp;
        big[i]  = tmp;
    }
}

void Output()
{
    std::ofstream fout;
    int           cnt;
    clock_t       start, stop;
    double        time_of_type, time_of_big;

    fout.open(typefile);
    cnt   = 0;
    start = clock();
    while (cnt != TESTS)
    {
        fout << type[cnt++] << std::endl;
    }
    fout << ZeroType << std::endl;
    stop = clock();
    fout.close();
    time_of_type = stop - start;

    fout.open(bigfile);
    cnt   = 0;
    start = clock();
    while (cnt != TESTS)
    {
        fout << big[cnt++] << std::endl;
    }
    fout << ZeroBig << std::endl;
    stop = clock();
    fout.close();
    time_of_big = stop - start;

    std::cout << "Output time of type: " << time_of_type << std::endl
              << "Output time of big: " << time_of_big << std::endl
              << "time_of_big / time_of_type = " << time_of_big / time_of_type << std::endl
              << std::endl;

    system((std::string("fc ") + typefile + " " + bigfile).c_str());
}

void Input()
{
    std::ifstream fin;
    Type          typetmp;
    BigInteger    bigtmp;
    clock_t       start, stop;
    double        time_of_type, time_of_big;

    fin.open(typefile);
    start = clock();
    while (fin >> typetmp);
    stop = clock();
    fin.close();
    time_of_type = stop - start;

    fin.open(bigfile);
    start = clock();
    while (fin >> bigtmp);
    stop = clock();
    fin.close();
    time_of_big = stop - start;

    std::cout << "Input time of type: " << time_of_type << std::endl
              << "Input time of big: " << time_of_big << std::endl
              << "time_of_big / time_of_type = " << time_of_big / time_of_type << std::endl
              << std::endl;
}

void Absolute()
{
    int           cnt;
    std::ofstream fout;
    clock_t       start, stop;
    double        time_of_type, time_of_big;

    fout.open(typefile);
    cnt   = 0;
    start = clock();
    while (cnt != TESTS)
    {
        if (type[cnt] == BigInteger::_LONG_LONG_MIN_)
        {
            fout << "9223372036854775808" << std::endl;
        }
        else if (type[cnt] == INT_MIN)
        {
            fout << "2147483648" << std::endl;
        }
        else
        {
            fout << abs(type[cnt]) << std::endl;
        }
        ++cnt;
    }
    fout << abs(ZeroType) << std::endl;
    stop = clock();
    fout.close();
    time_of_type = stop - start;

    fout.open(bigfile);
    cnt   = 0;
    start = clock();
    while (cnt != TESTS)
    {
        fout << big[cnt].abs() << std::endl;
        ++cnt;
    }
    fout << ZeroBig.abs() << std::endl;
    stop = clock();
    fout.close();
    time_of_big = stop - start;

    std::cout << "Abs time of type: " << time_of_type << std::endl
              << "Abs time of big: " << time_of_big << std::endl
              << "time_of_big / time_of_type = " << time_of_big / time_of_type << std::endl
              << std::endl;

    system((std::string("fc ") + typefile + " " + bigfile).c_str());
}

template<typename T>
void cmpcal(T *arr, int command, std::ofstream &fout)
{
    int     cnt;
    const T ZERO = 0;
    cnt          = 0;
    while (cnt != TESTS)
    {
        switch (command)
        {
        case 1:
        {
            fout << (arr[cnt] == arr[cnt + 1]) << std::endl
                 << (arr[cnt] != arr[cnt + 1]) << std::endl
                 << (arr[cnt] < arr[cnt + 1]) << std::endl
                 << (arr[cnt] <= arr[cnt + 1]) << std::endl
                 << (arr[cnt] > arr[cnt + 1]) << std::endl
                 << (arr[cnt] >= arr[cnt + 1]) << std::endl;
            break;
        }
        case 2:
            fout << arr[cnt] / arr[cnt + 1] << std::endl;
            break;
        case 3:
            fout << arr[cnt] % arr[cnt + 1] << std::endl;
            break;
        case 4:
            fout << arr[cnt] + arr[cnt + 1] << std::endl;
            break;
        case 5:
            fout << arr[cnt] - arr[cnt + 1] << std::endl;
            break;
        case 6:
            fout << arr[cnt] * arr[cnt + 1] << std::endl;
            break;
        default:
            break;
        }
        cnt += 2;
    }
    switch (command)
    {
    case 2:
        fout << ZERO / arr[0] << std::endl;
        break;
    case 3:
        fout << ZERO % arr[0] << std::endl;
        break;
    case 4:
        fout << ZERO + arr[0] << std::endl << arr[0] + ZERO << std::endl;
        break;
    case 5:
        fout << ZERO - arr[0] << std::endl << arr[0] - ZERO << std::endl;
        break;
    case 6:
        fout << ZERO * arr[0] << std::endl << arr[0] * ZERO << std::endl;
        break;
    default:
        break;
    }
}

void Compare_Calculate()
{
    std::ofstream fout;
    clock_t       start, stop;
    double        time_of_type, time_of_big;
    std::string   command[6] = {"Compare", "Div", "Mod", "Add", "Cut", "Multiply"};

    for (int i = 1; i <= 6; ++i)
    {
        switch (i)
        {
        case 4:
            Rand(true, false);
            break;
        case 6:
            Rand(false, true);
            break;
        default:
            break;
        }

        fout.open(typefile);
        start = clock();
        cmpcal(type, i, fout);
        stop = clock();
        fout.close();
        time_of_type = stop - start;

        fout.open(bigfile);
        start = clock();
        cmpcal(big, i, fout);
        stop = clock();
        fout.close();
        time_of_big = stop - start;

        std::cout << command[i - 1] << " time of type: " << time_of_type << std::endl
                  << command[i - 1] << " time of big: " << time_of_big << std::endl
                  << "time_of_big / time_of_type = " << time_of_big / time_of_type << std::endl
                  << std::endl;

        system((std::string("fc ") + typefile + " " + bigfile).c_str());
    }
}
