/**
 * @file DatabaseManager.h
 * @author Wei Li (Ithaca) (weili_yzzca@163.com)
 * @brief 数据库管理类（单例模式）
 * @version 0.1
 * @date 2026-04-20
 * 
 * @copyright Copyright (c) 2026
 * 
 */

#pragma once

#include <QMutex>
#include <QSqlDatabase>
#include <memory>

class DatabaseManager
{
public:
    static DatabaseManager &instance();

    bool         initialize(const QString &dbPath);
    QSqlDatabase getConnection();
    void         releaseConnection();

private:
    DatabaseManager() = default;
    ~DatabaseManager();

    QSqlDatabase m_db;
    QMutex       m_mutex;
    bool         m_initialized = false;
};
