#include "AppSubscriber.hpp"
#include "MessageCenter.hpp"
#include "Publisher.hpp"
#include "TopicMessage.hpp"

// ------------------------------------
int main(int argc, const char **argv)
{
    MessageCenter::getInstance()->Run();

    AppSubscriber appSub;
    Publisher     appPub;

    appSub.Subscribe("Person");
    appSub.Subscribe("Person");

    //appSub.UnSubscribe("Person");

    Person ps{"sma", 18};

    Person ps1{"wxq", 17};

    appPub.Publish("Person", &ps, sizeof(ps));
    while (true)
    {
        //appPub.Publish("Person", &ps, sizeof(ps));
        appPub.Publish("Person", &ps1, sizeof(ps1));
    }
    
    return 0;
}
