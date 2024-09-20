#ifndef __GRAPH_RENDER_WIDGET_H__
#define __GRAPH_RENDER_WIDGET_H__

#include <QOpenGLFunctions_4_5_Core> // GLAD的功能
#include <QOpenGLWidget>             // GLFW的功能

class GraphRenderWidget
    : public QOpenGLWidget
    , QOpenGLFunctions_4_5_Core
{
    Q_OBJECT // 这个宏是为了能够使用Qt中的信号槽机制

        public : GraphRenderWidget(QWidget *parent = nullptr);
    ~GraphRenderWidget();

protected:
    virtual void initializeGL() override;
    virtual void resizeGL(int w, int h) override;
    virtual void paintGL() override;
};

#endif /* __GRAPH_RENDER_WIDGET_H__ */