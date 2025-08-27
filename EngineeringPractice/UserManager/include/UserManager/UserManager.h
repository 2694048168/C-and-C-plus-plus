/**
 * @file UserManager.h
 * @author Wei Li (Ithaca) (weili_yzzcq@163.com)
 * @brief UserManager类采用单例模式实现, 确保系统中只有一个实例;
 *     主要功能包括用户登录、登出、自动检测无操作等.
 * @version 0.1
 * @date 2025-04-15
 * 
 * @copyright Copyright (c) 2025
 * 
 */

#pragma once

#include <string>

enum UserRole
{
};

class UserManager
{
public:
    // 获取单例实例
    static UserManager &getInstance();

    // 用户登录
    bool login(const std::string &username, const std::string &password, bool rememberMe = false);
    
    // 用户登出
    void logout();

    // 获取用户登录状态
    bool isLoggedIn() const;

    // 创建新用户
    bool createUser(const std::string &username, const std::string &password, UserRole role, int timeoutMinutes = 30,
                    int expirationHours = 72);

    // 自动登出检测
    void initializeIdleDetection(HWND hwnd);
    void terminateIdleDetection();
    bool isInactiveTimeout() const; // 检测是否超时
    void updateActivityTime();      // 更新活动时间
};
