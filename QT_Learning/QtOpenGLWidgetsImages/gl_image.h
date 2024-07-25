#pragma once

/**
 * @file WidgetGLImage.h
 * @author Wei Li (Ithaca)
 * @emial weili_yzzcq@163.com
 * @brief QT + OpenGL 高速高效实时渲染刷新图像
 * @version 0.1
 * @date 2024-07-21
 *
 * @copyright Copyright (c) 2024
 *
 * @revise date 2024-07-21
 *
 * 使用QOpenGLWidget(调用GPU)渲染QImage加载的图片
 * https://bbs.huaweicloud.com/blogs/335248
 * Qt使用OpenGL来显示一张图片
 * https://www.cnblogs.com/tony-yang-flutter/p/16252881.html
 * QOpenGLWidget 绘制文字的多种方法实践
 * https://blog.csdn.net/qq_21438461/article/details/135683355
 * QGLWidget、QOpenGLWidget详解及区别
 * https://blog.csdn.net/qq21497936/article/details/94585803
 * An OpenGL-based image viewer
 * https://github.com/AlvinJian/GLImageViewer
 * Using OpenGL in Qt for Processing Images
 * https://amin-ahmadi.com/2019/07/12/using-opengl-in-qt-for-processing-images/
 * Uses Qt and OpenGL to render a selected image on a plane
 * https://github.com/AlexDiru/qt-opengl-imageviewer/tree/master
 * Rendering Text with Qt OpenGL: Step-by-Step Guide
 * https://devcodef1.com/news/1170796/qt-opengl-text-rendering
 *
 */

#include <QImage>
#include <QOpenGLFunctions>
#include <QOpenGLWidget>
#include <QWheelEvent>
#include <atomic>

struct TextInfoWidget
{
    QColor  color;
    QString text;
    int     x;
    int     y;
};

class WidgetGLImage : public QOpenGLWidget
{
    Q_OBJECT

public:
    enum
    {
        Left_Bottom_X,
        Left_Bottom_Y,
        Right_Bottom_X,
        Right_Bottom_Y,
        Right_Top_X,
        Right_Top_Y,
        Left_Top_X,
        Left_Top_Y,
        Pos_Max
    };

    explicit WidgetGLImage(QWidget* parent = nullptr);

    // 设置实时显示的数据源
    void SetImage(QImage* image);

    void addText(int x, int y, QString txt, QColor color);
    //void addRect(SdRenderControl::RectInfo& rectInfo);
    //void addLine(SdRenderControl::LineInfo& lineInfo);
    void clearText();
    void up_paintGL();

protected:
    // 重写QGLWidget类的接口
    void initializeGL();
    void paintGL();
    void resizeGL(int w, int h);

    // 鼠标事件
    void wheelEvent(QWheelEvent* e);
    void mouseMoveEvent(QMouseEvent* e);
    void mousePressEvent(QMouseEvent* e);
    void mouseReleaseEvent(QMouseEvent* e);
    void mouseDoubleClickEvent(QMouseEvent* e);

private:
    QImage* imageData_; //纹理显示的数据源
    QVector<TextInfoWidget> m_txts;     //显示文本数据
    //QVector<SdRenderControl::RectInfo> m_rects;
    //QVector<SdRenderControl::LineInfo> m_lines;
    QSize                   imageSize_;           //图片尺寸
    QSize                   adaptImageSize_;      //适配后图片尺寸
    QSize                   Ortho2DSize_;         //窗口尺寸
    GLuint                  textureId_;           //纹理对象ID
    GLuint                  textureId2_;          //纹理对象ID
    GLuint                  textureId3_;          //纹理对象ID
    int                     vertexPos_[Pos_Max];  //窗口坐标
    float                   texturePos_[Pos_Max]; //纹理坐标
    bool                    dragFlag_;            //鼠标拖拽状态
    QPoint                  dragPos_;             //鼠标拖拽位置
    float                   scaleVal_;            //缩放倍率
    bool                    initTextureFlag = false;
    QPoint                  offset_; //左上角顶点
    float                   scale;   //缩放倍率
    std::atomic<bool>       m_isUpdate;
};
