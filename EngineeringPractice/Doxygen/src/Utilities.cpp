#include "Utilities.h"

std::vector<std::string> split(const std::string &str, char delimiter)
{
    std::vector<std::string> elements;
    std::stringstream        ss(str);
    std::string              item;
    while (getline(ss, item, delimiter))
    {
        elements.push_back(item);
    }
    return elements;
}
