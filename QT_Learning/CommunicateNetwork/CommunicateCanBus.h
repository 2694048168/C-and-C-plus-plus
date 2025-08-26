/**
 * @file CommunicateCanBus.h
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

#include <QCanBusDevice>
#include <QCanBusFrame>

class CanBus : public ICommunication
{
    Q_OBJECT
public:
    explicit CanBus(QObject *parent = nullptr);
    ~CanBus() override;

    bool connect(const QVariantMap &config) override;
    void disconnect() override;
    qint64 write(const QByteArray &data) override;
    QByteArray read(qint32 timeout = 1000) override;
    QVariantMap status() const override;

private slots:
    void onFramesReceived();

private:
    QCanBusDevice *m_device = nullptr;
    QMutex m_ioMutex;
};
