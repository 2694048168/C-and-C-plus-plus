#include <iostream>
#include <vector>
#include <string>
#include <fstream>
#include <algorithm>

int main(int argc, char*argv[])
{
    std::ifstream in_file("./text.txt");
    if (! in_file)
    {
        std::cerr << "oops! unable to open input file\n";
        return -1;
    }

    std::ofstream out_file("./text_sort.txt");
    if (! out_file)
    {
        std::cerr << "oops! unable to open output fiel\n";
        return -2;
    }
    
    std::string word;
    std::vector<std::string> text;

    while (in_file >> word)
        text.push_back(word);
    
    int ix;
    std::cout << "unsorted text: \n";

    for ( ix = 0; ix < text.size(); ++ix)
        std::cout << text[ix] << " ";
    std::cout << std::endl;  

    sort(text.begin(), text.end());

    out_file << "sorted text: \n";
    for (ix = 0; ix < text.size(); ++ix)
    {
        out_file << text[ix] << " ";
    }
    out_file << std::endl;
    

    return 0;
}