#include "MessageCenter.hpp"

#include "Publisher.hpp"
#include "Subscriber.hpp"

const unsigned int MAX_PUBLISHES{10000};

MessageCenter *MessageCenter::mSgMC = nullptr;
std::mutex     MessageCenter::mMCMutex;

MessageCenter *MessageCenter::getInstance()
{
    if (mSgMC == nullptr)
    {
        std::unique_lock<std::mutex> lock(mMCMutex);
        if (mSgMC == nullptr)
        {
            volatile auto temp = new (std::nothrow) MessageCenter();
            mSgMC              = temp;
        }
    }
    return mSgMC;
}

void MessageCenter::Run()
{
    mCoreProcess.reset(new std::thread(&MessageCenter::CoreProcess, this));
}

void MessageCenter::RegisterPublish(const std::string &tpcKey, void *message, unsigned int datasize)
{
    if ((this->mSubscriber.find(tpcKey) != this->mSubscriber.end())
        && (this->mPublisher[tpcKey].size() > MAX_PUBLISHES))
        return;
    mPublishMutex.lock();
    void *temp_data = new char[datasize];
    memcpy(temp_data, message, datasize);
    this->mPublisher[tpcKey].push_back(temp_data);
    mPublishMutex.unlock();
}

void MessageCenter::RegisterSubscribe(const std::string &tpcKey, Subscriber *subscriber)
{
    mSubscribeMutex.lock();
    this->mSubscriber[tpcKey].remove(subscriber);
    this->mSubscriber[tpcKey].push_back(subscriber);
    mSubscribeMutex.unlock();
}

void MessageCenter::CancelSubscribe(const std::string &tpcKey, Subscriber *subscriber)
{
    mSubscribeMutex.lock();
    if (this->mSubscriber.find(tpcKey) != this->mSubscriber.end())
        this->mSubscriber.find(tpcKey)->second.remove(subscriber);
    mSubscribeMutex.unlock();
}

MessageCenter::MessageCenter()
{
    this->mPublisher.clear();
    this->mSubscriber.clear();
}

MessageCenter::~MessageCenter() {}

void MessageCenter::CoreProcess()
{
    while (true)
    {
        auto it = this->mSubscriber.begin();
        while (it != this->mSubscriber.end())
        {
            if (this->mPublisher.find(it->first) != this->mPublisher.end())
            {
                auto itt = it->second.begin();
                while (itt != it->second.end())
                {
                    auto mp_iter     = this->mPublisher.find(it->first)->second.begin();
                    auto mp_iter_end = this->mPublisher.find(it->first)->second.end();
                    while (mp_iter != mp_iter_end)
                    {
                        (*itt)->HandleEvent(it->first, *mp_iter);
                        ++mp_iter;
                    }
                    ++itt;
                }
                mPublishMutex.lock();
                auto mp_iter     = this->mPublisher.find(it->first)->second.begin();
                auto mp_iter_end = this->mPublisher.find(it->first)->second.end();
                while (mp_iter != mp_iter_end)
                {
                    delete[] (*mp_iter);
                    ++mp_iter;
                }
                this->mPublisher.find(it->first)->second.clear();
                this->mPublisher.erase(it->first);
                mPublishMutex.unlock();
            }
            ++it;
        }
    }
}
