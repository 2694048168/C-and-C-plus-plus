#include "Config/Config.hpp"

ApplicationConfigParam::ApplicationConfigParam()
    : mParam{}
{
}

ApplicationConfigParam::~ApplicationConfigParam() {}

ApplicationConfigParam &ApplicationConfigParam::getInstance()
{
    static ApplicationConfigParam instance;
    return instance;
}
