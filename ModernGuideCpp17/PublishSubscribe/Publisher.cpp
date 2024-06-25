#include "Publisher.hpp"

#include "MessageCenter.hpp"

void Publisher::Publish(const std::string &Topic, void *message, unsigned int datasize)
{
    MessageCenter::getInstance()->RegisterPublish(Topic, message, datasize);
}
