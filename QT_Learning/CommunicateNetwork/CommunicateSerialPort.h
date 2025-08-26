/**
 * @file CommunicateSerialPort.h
 * @author Wei Li (Ithaca) (weili_yzzcq@163.com)
 * @brief
 * @version 0.1
 * @date 2025-08-26
 *
 * @copyright Copyright (c) 2025
 *
 */

#pragma once

#include "ICommunication.h"

#include <QSerialPort>
#include <QSerialPortInfo>

class SerialPort : public ICommunication
{
    Q_OBJECT
public:
    explicit SerialPort(QObject *parent = nullptr);
    ~SerialPort() override;

    bool connect(const QVariantMap &config) override;
    void disconnect() override;
    qint64 write(const QByteArray &data) override;
    QByteArray read(qint32 timeout = 1000) override;
    QVariantMap status() const override;

private slots:
    void onReadyRead(); // 串口数据就绪槽函数

private:
    QSerialPort *m_serial = nullptr; // 串口对象
    QMutex m_ioMutex;                // IO操作锁
};
