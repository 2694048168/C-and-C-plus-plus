/**
 * @file SerialRefresh.h
 * @author Wei Li (weili_yzzcq@163.com)
 * @brief 
 * @version 0.1
 * @date 2024-10-01
 * 
 * @copyright Copyright (c) 2024
 * 
 */

#pragma once

#include <qtmetamacros.h>

#include <QComboBox>
#include <QWidget>

class SerialRefresh : public QComboBox
{
    Q_OBJECT

public:
    SerialRefresh(QWidget *parent = nullptr);
    ~SerialRefresh() = default;

protected:
    void mousePressEvent(QMouseEvent *event) override;

signals:
    void refresh();
};
