/**
 * @file 18_BranchPrediction.cpp
 * @author Wei Li (weili_yzzcq@163.com)
 * @brief 
 * @version 0.1
 * @date 2024-12-20
 * 
 * @copyright Copyright (c) 2024
 * 
 */

#include <iostream>

/* C++20 的分支预测-代码标记让性能飙升 
 * C++20 的小技巧 -[[likely]] 和[[unlikely]] 属性,
 * 告诉编译器："psst...这条路径超常用的！"或者"嘘...这条路基本上不会走的~" 
 * 
 */
bool validate_user_input(int value)
{
    if ([[likely]] (value >= 0 && value <= 100))
    { // 乖宝宝们都会输入正确的值 😇
        return true;
    }
    return false; // 这条路基本走不到啦~
}

void process_file(const std::string &filename)
{
    std::ifstream file(filename);

    if ([[unlikely]] (!file.is_open()))
    { // 这种倒霉事应该很少发生吧? 🤪
        throw std::runtime_error("哎呀呀,文件打不开啦!");
    }

    // 开心地处理文件去咯... 🎵
}

// [[likely]] 和[[unlikely]] 的精髓所在
// 编译器重新安排代码, 编译器会用一些特殊的暗号来跟CPU交流
bool handle_login(const User &user)
{
    if ([[likely]] (user.has_valid_token()))
    { // 大多数用户都是好孩子呢~ 😇
        return process_login(user);
    }
    // ...
}

void save_document(const Document &doc)
{
    if ([[unlikely]] (disk_space_low()))
    { // 这种倒霉事可不常见 😅
        throw std::runtime_error("哎呀，硬盘空间不够啦！");
    }

    // 开心地保存文档...
}

void process_game_action(const Action &action)
{
    if ([[likely]] (action.is_movement()))
    { // 玩家总是在跑来跑去呢！🏃‍♂️
        handle_movement(action);
    }
    else if ([[unlikely]] (action.is_rare_skill()))
    { // 大招可不是随便放的！✨
        handle_rare_skill(action);
    }
}

// --------------------------------------------------
int main(int /* argc */, const char /* *argv[] */)
{
    std::cout << "The C++20 [[likely]] 和[[unlikely]] magic\n";

    return 0;
}
