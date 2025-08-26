/**
 * @file CommunicateModbus.h
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

#include <QModbusDevice>
#include <QModbusDataUnit>

class ModbusRTU : public ICommunication
{
    Q_OBJECT
public:
    explicit ModbusRTU(QObject *parent = nullptr);
    ~ModbusRTU() override;

    bool connect(const QVariantMap &config) override;
    void disconnect() override;
    qint64 write(const QByteArray &data) override;
    QByteArray read(qint32 timeout = 1000) override;
    QVariantMap status() const override;

    // Modbus 专用方法
    bool readHoldingRegisters(quint8 slaveId, quint16 startAddr, quint16 count);

private slots:
    void onReplyFinished();

private:
    QModbusDevice *m_device = nullptr;
    QMutex m_ioMutex;
};
