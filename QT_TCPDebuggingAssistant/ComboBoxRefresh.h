/**
 * @file ComboBoxRefresh.h
 * @author Wei Li (weili_yzzcq@163.com)
 * @brief 
 * @version 0.1
 * @date 2024-10-01
 * 
 * @copyright Copyright (c) 2024
 * 
 */

#pragma once

#include <QComboBox>
#include <QWidget>

class ComboBoxRefresh : public QComboBox
{
    Q_OBJECT

public:
    ComboBoxRefresh(QWidget *parent = nullptr);
    ~ComboBoxRefresh() = default;

protected:
    void mousePressEvent(QMouseEvent *event) override;

signals:
    void comboBox_clicked();
};
