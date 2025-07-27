/**
 * @file mainBasic.cpp
 * @author Wei Li (Ithaca) (weili_yzzcq@163.com)
 * @brief the basic OP. for SQLiter via C++
 * @version 0.1
 * @date 2025-07-27
 * 
 * @copyright Copyright (c) 2025
 * 
 */

#include <algorithm>
#include <atomic>
#include <format>
#include <iostream>
#include <string>
#include <unordered_map>
#include <vector>

/** 生成导入库 (.lib) : https://sqlite.org/download.html
* 0. Precompiled Binaries for Windows：sqlite-dll-win64-x64-xxxxxxx.zip（64位）
* 1. 打开 VS 开发人员命令提示符（搜索 "Developer Command Prompt"）
* 2. 执行命令:
* lib /def:sqlite3.def /out:sqlite3.lib /machine:x64
* lib /def:sqlite3.def /out:sqlite3.lib /machine:x86
* 3. download source code and select the header file
* Source Code：sqlite-amalgamation-xxxxxxx.zip
*/
#include "sqlite3.h"

class DatabaseSqlite
{
public:
    enum class ErrorCode
    {
        OK            = 0x0000,
        NG            = 0x0001,
        UNIMPLEMENTED = 0x0002,
    };

    // unordered_map常量存放 ResultStatus 对应字符串
    const std::unordered_map<ErrorCode, std::string> errorCodeToString{
        {ErrorCode::UNIMPLEMENTED,                    "UNIMPLEMENTED: 接口或者功能未实现"},
        {           ErrorCode::OK,                                    "SUCCESS: 操作成功"},
        {           ErrorCode::NG, "FAILED: 操作失败(通用), 对于不便于分类的操作返回错误"},
    };

    std::string GetErrorMessage() noexcept
    {
        return mErrorMessage;
    }

    std::string GetErrorCodeToString(const ErrorCode ret) noexcept
    {
        auto iter = errorCodeToString.find(ret);
        if (iter != errorCodeToString.end())
        {
            return iter->second;
        }
        else
        {
            return "Error Code not find: " + std::to_string((int)ret);
        }
    }

    ErrorCode Open(std::string filepath)
    {
        mDatabaseFilepath = filepath;
        auto ret          = sqlite3_open(mDatabaseFilepath.c_str(), &mpDatabase);
        if (0 == ret)
        {
            mErrorMessage = std::format("Opened {} Database successfully\n", mDatabaseFilepath);
            // log.info(mErrorMessage);
            return ErrorCode::OK;
        }
        else
        {
            auto errmsg   = sqlite3_errmsg(mpDatabase);
            mErrorMessage = std::format("Opened {} Database NOT successfully : {}\n", mDatabaseFilepath, errmsg);
            // log.error(mErrorMessage);
            return ErrorCode::NG;
        }
    }

    ErrorCode Close()
    {
        auto ret = sqlite3_close(mpDatabase);
        if (0 == ret)
        {
            mErrorMessage = std::format("Close {} Database successfully\n", mDatabaseFilepath);
            // log.info(mErrorMessage);
            return ErrorCode::OK;
        }
        else
        {
            auto errmsg   = sqlite3_errmsg(mpDatabase);
            mErrorMessage = std::format("Close {} Database NOT successfully : {}\n", mDatabaseFilepath, errmsg);
            // log.error(mErrorMessage);
            return ErrorCode::NG;
        }
    }

    ErrorCode CreateTable(const std::string &tableName)
    {
        if (nullptr == mpDatabase)
        {
            mErrorMessage = std::format("Please Open {} Database first\n", mDatabaseFilepath);
            // log.info(mErrorMessage);
            return ErrorCode::NG;
        }

        // TODO: 需要查询 database 中 table exist？
        //auto iter = std::find(mTableNameVec.begin(), mTableNameVec.end(), tableName);
        //if (iter != mTableNameVec.end())
        //{
        //    mErrorMessage = std::format("{} in the Database {} already exists\n", tableName, mDatabaseFilepath);
        //    // log.info(mErrorMessage);
        //    return ErrorCode::NG;
        //}
        mTableNameVec.emplace_back(tableName);

        // Create SQL statement
        auto sqlStr = std::format(
            "CREATE TABLE {}"
            "(ID INT PRIMARY KEY     NOT NULL,"
            "NAME           TEXT    NOT NULL,"
            "AGE            INT     NOT NULL,"
            "ADDRESS        CHAR(50),"
            "SALARY         REAL );",
            tableName);

        char *errmsg = nullptr;
        auto  ret    = sqlite3_exec(mpDatabase, sqlStr.c_str(), callback, nullptr, &errmsg);
        if (0 == ret)
        {
            mErrorMessage = std::format("Create {} Database successfully\n", mDatabaseFilepath);
            // log.info(mErrorMessage);
            return ErrorCode::OK;
        }
        else
        {
            mErrorMessage = std::format("Create {} Database NOT successfully : {}\n", mDatabaseFilepath, errmsg);
            // log.error(mErrorMessage);
            return ErrorCode::NG;
        }
    }

    ErrorCode Insert(const std::string &tableName)
    {
        if (nullptr == mpDatabase)
        {
            mErrorMessage = std::format("Please Open {} Database first\n", mDatabaseFilepath);
            // log.info(mErrorMessage);
            return ErrorCode::NG;
        }

        // Create SQL statement
        auto sqlStr = std::format(
            "INSERT INTO {0} (ID,NAME,AGE,ADDRESS,SALARY) "
            "VALUES (1, 'Paul', 32, 'California', 20000.00 ); "
            "INSERT INTO {0} (ID,NAME,AGE,ADDRESS,SALARY) "
            "VALUES (2, 'Allen', 25, 'Texas', 15000.00 ); "
            "INSERT INTO {0} (ID,NAME,AGE,ADDRESS,SALARY)"
            "VALUES (3, 'Teddy', 23, 'Norway', 20000.00 );"
            "INSERT INTO {0} (ID,NAME,AGE,ADDRESS,SALARY)"
            "VALUES (4, 'Mark', 25, 'Rich-Mon', 65000.00 );",
            tableName);

        char *errmsg = nullptr;
        auto  ret    = sqlite3_exec(mpDatabase, sqlStr.c_str(), callback, nullptr, &errmsg);
        if (0 == ret)
        {
            mErrorMessage = std::format("Insert {} Database successfully\n", mDatabaseFilepath);
            // log.info(mErrorMessage);
            return ErrorCode::OK;
        }
        else
        {
            mErrorMessage = std::format("Insert {} Database NOT successfully : {}\n", mDatabaseFilepath, errmsg);
            // log.error(mErrorMessage);
            return ErrorCode::NG;
        }
    }

    ErrorCode Select(const std::string &tableName)
    {
        if (nullptr == mpDatabase)
        {
            mErrorMessage = std::format("Please Open {} Database first\n", mDatabaseFilepath);
            // log.info(mErrorMessage);
            return ErrorCode::NG;
        }

        // Create SQL statement
        auto sqlStr = std::format("SELECT * from {}", tableName);

        char       *errmsg = nullptr;
        const char *data   = "Callback function called";
        auto        ret    = sqlite3_exec(mpDatabase, sqlStr.c_str(), callback, (void *)data, &errmsg);
        if (0 == ret)
        {
            mErrorMessage = std::format("Select from {} Database successfully\n", mDatabaseFilepath);
            // log.info(mErrorMessage);
            return ErrorCode::OK;
        }
        else
        {
            mErrorMessage = std::format("Select from {} Database NOT successfully : {}\n", mDatabaseFilepath, errmsg);
            // log.error(mErrorMessage);
            return ErrorCode::NG;
        }
    }

    ErrorCode Update(const std::string &tableName)
    {
        if (nullptr == mpDatabase)
        {
            mErrorMessage = std::format("Please Open {} Database first\n", mDatabaseFilepath);
            // log.info(mErrorMessage);
            return ErrorCode::NG;
        }

        // Create SQL statement
        auto sqlStr = std::format(
            "UPDATE {0} set SALARY = 75000.00 where ID=1; "
            "SELECT * from {0}",
            tableName);

        char       *errmsg = nullptr;
        const char *data   = "Callback function called";
        auto        ret    = sqlite3_exec(mpDatabase, sqlStr.c_str(), callback, (void *)data, &errmsg);
        if (0 == ret)
        {
            mErrorMessage = std::format("Update {} Database successfully\n", mDatabaseFilepath);
            // log.info(mErrorMessage);
            return ErrorCode::OK;
        }
        else
        {
            mErrorMessage = std::format("Update {} Database NOT successfully : {}\n", mDatabaseFilepath, errmsg);
            // log.error(mErrorMessage);
            return ErrorCode::NG;
        }
    }

    ErrorCode Delete(const std::string &tableName)
    {
        if (nullptr == mpDatabase)
        {
            mErrorMessage = std::format("Please Open {} Database first\n", mDatabaseFilepath);
            // log.info(mErrorMessage);
            return ErrorCode::NG;
        }

        // Create SQL statement
        auto sqlStr = std::format(
            "DELETE from {0} where ID=2; "
            "SELECT * from {0}",
            tableName);

        char       *errmsg = nullptr;
        const char *data   = "Callback function called";
        auto        ret    = sqlite3_exec(mpDatabase, sqlStr.c_str(), callback, (void *)data, &errmsg);
        if (0 == ret)
        {
            mErrorMessage = std::format("Delete {} Database successfully\n", mDatabaseFilepath);
            // log.info(mErrorMessage);
            return ErrorCode::OK;
        }
        else
        {
            mErrorMessage = std::format("Delete {} Database NOT successfully : {}\n", mDatabaseFilepath, errmsg);
            // log.error(mErrorMessage);
            return ErrorCode::NG;
        }
    }

public:
    DatabaseSqlite()
        : mpDatabase{nullptr}
        , mDatabaseFilepath{}
        , mTableNameVec{}
        , mErrorMessage{}
    {
    }

    ~DatabaseSqlite()
    {
        if (mpDatabase)
            Close();
        mTableNameVec.clear();
    }

private:
    static int callback(void *NotUsed, int argc, char **argv, char **azColName)
    {
        for (int i = 0; i < argc; ++i)
        {
            printf("%s = %s\n", azColName[i], argv[i] ? argv[i] : "NULL");
        }
        printf("\n");
        return 0;
    }

private:
    sqlite3                 *mpDatabase;
    std::string              mDatabaseFilepath;
    std::vector<std::string> mTableNameVec;
    std::string              mErrorMessage;
    // std::atomic<std::string> mErrorMessage_mt; // support multi-thread
};

int main(int argc, const char **argv)
{
    DatabaseSqlite database;

    std::string filepath = R"(./database.db)";
    auto        ret      = database.Open(filepath);
    if (ret == DatabaseSqlite::ErrorCode::OK)
        std::cout << "Open database successfully\n";
    else
        std::cout << "Open database NOT successfully\n";

    std::string tableName = "test";
    ret                   = database.CreateTable(tableName);
    if (ret == DatabaseSqlite::ErrorCode::OK)
        std::cout << database.GetErrorMessage();
    else
        std::cout << database.GetErrorMessage();

    tableName = "test2";
    ret       = database.CreateTable(tableName);
    if (ret == DatabaseSqlite::ErrorCode::OK)
        std::cout << database.GetErrorMessage();
    else
        std::cout << database.GetErrorMessage();

    ret = database.Insert(tableName);
    if (ret == DatabaseSqlite::ErrorCode::OK)
        std::cout << database.GetErrorMessage();
    else
        std::cout << database.GetErrorMessage();

    ret = database.Select(tableName);
    if (ret == DatabaseSqlite::ErrorCode::OK)
        std::cout << database.GetErrorMessage();
    else
        std::cout << database.GetErrorMessage();

    ret = database.Update(tableName);
    if (ret == DatabaseSqlite::ErrorCode::OK)
        std::cout << database.GetErrorMessage();
    else
        std::cout << database.GetErrorMessage();

    ret = database.Close();
    if (ret == DatabaseSqlite::ErrorCode::OK)
        std::cout << "Close database successfully\n";
    else
        std::cout << "Close database NOT successfully\n";

    system("pause");
    return 0;
}
