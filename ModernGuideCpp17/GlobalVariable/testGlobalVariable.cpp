#include "testGlobalVariable.h"

#include "GlobalVariable.h"
extern int num;
extern std::map<const char *, std::vector<ImgData>> mapTable;

void addOnce(const int &value)
{
    num += value;
}


bool addMapTable(const char* key, std::vector<ImgData> value)
{
    mapTable[key] = value;

    return true;
}