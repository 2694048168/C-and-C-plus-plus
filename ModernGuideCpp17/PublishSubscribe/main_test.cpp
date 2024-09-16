/**
 * @file main_test.cpp
 * @author Wei Li (weili_yzzcq@163.com)
 * @brief 
 * @version 0.1
 * @date 2024-09-16
 * 
 * @copyright Copyright (c) 2024
 * 
 */

#include "ObserverMode/Observer.h"

#include <iostream>

// ------------------------------------
int main(int argc, const char **argv)
{
    Morgans *ms     = new Morgans;
    Gossip  *gossip = new Gossip;

    Dragon     *dragon = new Dragon("蒙奇·D·龙", ms);
    Shanks     *shanks = new Shanks("香克斯", ms);
    Bartolomeo *barto  = new Bartolomeo("巴托洛米奥", gossip);

    ms->notify("蒙奇·D·路飞成为新世界的新的四皇之一, 赏金30亿贝里!!!");
    std::cout << "======================================\n";
    gossip->notify("女帝汉库克想要嫁给路飞, 给路飞生猴子, 哈哈哈...");

    delete ms;
    delete gossip;
    delete dragon;
    delete shanks;
    delete barto;

    return 0;
}
