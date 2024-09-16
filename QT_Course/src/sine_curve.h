#ifndef __SINE_CURVE_H__
#define __SINE_CURVE_H__

#include <QPainter>
#include <QWidget>
#include <QtMath>

namespace Ui {
class Widget;
}

class WidgetSineCurve : public QWidget
{
    Q_OBJECT

public:
    explicit WidgetSineCurve(QWidget *parent = nullptr);
    ~WidgetSineCurve();

protected:
    void paintEvent(QPaintEvent *event) override;
    void timerEvent(QTimerEvent *event) override;

private:
    Ui::Widget *ui;

    int             x;       // 当前点的x坐标
    int             y;       // 当前点的y坐标
    int             y_;      // 当前点的y坐标
    QVector<QPoint> points;  // 存储绘制曲线的点
    QVector<QPoint> points_; // 存储绘制曲线的点
};

#endif // __SINE_CURVE_H__