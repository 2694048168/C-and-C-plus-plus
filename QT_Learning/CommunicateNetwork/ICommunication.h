/**
 * @file ICommunication.h
 * @author Wei Li (Ithaca) (weili_yzzcq@163.com)
 * @brief 通讯抽象接口
 * @version 0.1
 * @date 2025-08-26
 *
 * @copyright Copyright (c) 2025
 *
 */

#pragma once

#include <QObject>
#include <QByteArray>
#include <QVariantMap>
#include <QMutex>
#include <QList>

// 通信接口定义（策略模式基类）
class DataObserver : public QObject
{
    Q_OBJECT
public:
    explicit DataObserver(QObject *parent = nullptr)
        : QObject(parent)
    {
    }
    virtual void onDataReceived(const QByteArray &data) = 0;
    virtual void onError(const QString &errorMsg) = 0;
};

class ICommunication : public QObject
{
    Q_OBJECT
public:
    explicit ICommunication(QObject *parent = nullptr)
        : QObject(parent)
    {
    }
    virtual ~ICommunication() = default;

    // 核心通信能力
    virtual bool connect(const QVariantMap &config) = 0; // 连接设备
    virtual void disconnect() = 0;                       // 断开连接
    virtual qint64 write(const QByteArray &data) = 0;    // 发送数据
    virtual QByteArray read(qint32 timeout = 1000) = 0;  // 读取数据（阻塞）
    virtual QVariantMap status() const = 0;              // 获取状态

    // 观察者管理（观察者模式）
    void registerObserver(DataObserver *observer)
    {
        QMutexLocker locker(&m_observerMutex);
        if (!m_observers.contains(observer))
        {
            m_observers.append(observer);
        }
    }

    void unregisterObserver(DataObserver *observer)
    {
        QMutexLocker locker(&m_observerMutex);
        m_observers.removeAll(observer);
    }

signals:
    void dataReceived(const QByteArray &data);   // 数据接收信号
    void errorOccurred(const QString &errorMsg); // 错误信号

protected:
    void notifyDataReceived(const QByteArray &data)
    {
        QMutexLocker locker(&m_observerMutex);
        for (DataObserver *observer : m_observers)
        {
            observer->onDataReceived(data);
        }
    }

    void notifyError(const QString &errorMsg)
    {
        QMutexLocker locker(&m_observerMutex);
        for (DataObserver *observer : m_observers)
        {
            observer->onError(errorMsg);
        }
    }

private:
    QList<DataObserver *> m_observers; // 观察者列表
    QMutex m_observerMutex;            // 观察者锁
};
