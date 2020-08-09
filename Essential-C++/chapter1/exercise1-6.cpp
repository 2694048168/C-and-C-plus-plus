#include <iostream>
#include <vector>

int main(int argc, char*argv[])
{
    std::vector<int> ivec;
    int ival = 0, sum = 0, average = 0;

    // solution logistic 1 to compute sum
    while (std::cin >> ival)
    {
        ivec.push_back(ival);
        sum += ival;
    }

    // solution logistic 2 to compute sum
    for (size_t i = 0; i < ivec.size(); ++i)
    {
        sum += ivec[i];
    }
    

    average = sum / ivec.size();

    std::cout << "Sum of " << ivec.size()
              << " elements: " << sum
              << " Average: " << average << std::endl;

    return 0;
}