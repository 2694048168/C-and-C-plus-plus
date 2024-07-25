#include "gl_image.h"
#include <QPainter>

WidgetGLImage::WidgetGLImage(QWidget* parent)
    : QOpenGLWidget(parent)
{
    dragFlag_ = false;
    scaleVal_ = 1.0;
    imageData_ = nullptr;
    m_isUpdate = false;
    scale = 1.0;
    makeCurrent();
}

// 设置待显示的数据源和尺寸
void WidgetGLImage::SetImage(QImage* image)
{
    imageData_ = image;
    initTextureFlag = false;
    imageSize_.setWidth(image->width());
    imageSize_.setHeight(image->height());
    scaleVal_ = 1.0;
    m_isUpdate = true;
    //repaint();
    //update();
}

void WidgetGLImage::addText(int x, int y, QString txt, QColor color)
{
    TextInfoWidget textInfo;
    textInfo.x = x;
    textInfo.y = y;
    textInfo.text = txt;
    textInfo.color = color;
    m_txts.append(textInfo);
}

//
//void WidgetGLImage::addRect(SdRenderControl::RectInfo& rectInfo)
//{
//    m_rects.append(rectInfo);
//}

//void WidgetGLImage::addLine(SdRenderControl::LineInfo& lineInfo)
//{
//    m_lines.append(lineInfo);
//}

void WidgetGLImage::clearText()
{
    m_txts.clear();
    //m_rects.clear();
    //m_lines.clear();
}

void WidgetGLImage::up_paintGL()
{
    if (m_isUpdate)
    {
        m_isUpdate = false;
        this->update();
    }
}

void WidgetGLImage::initializeGL()
{
    // 生成一个纹理ID
    glGenTextures(1, &textureId_);

    // 绑定该纹理ID到二维纹理上
    glBindTexture(GL_TEXTURE_2D, textureId_);
    // 用线性插值实现图像缩放
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
}

// 窗口绘制函数
void WidgetGLImage::paintGL()
{
    // 设置背景颜色
    glClearColor(1.0f, 1.0f, 1.0f, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    if (imageData_ == nullptr)
    {
        return;
    }
    QImage tex = imageData_->mirrored();
    glBindTexture(GL_TEXTURE_2D, textureId_);

    if (!initTextureFlag)
    {
        // 生成纹理
        if (imageData_->depth() == 8)
            glTexImage2D(GL_TEXTURE_2D, 0, GL_DEPTH_COMPONENT, imageSize_.width(), imageSize_.height(), 0,
                GL_DEPTH_COMPONENT, GL_UNSIGNED_BYTE, imageData_->bits());
        else
            glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, imageSize_.width(), imageSize_.height(), 0, GL_RGBA,
                GL_UNSIGNED_BYTE, imageData_->bits());

        // 初始化顶点坐标（居中显示）
        int x_offset = 0;
        int y_offset = 0;
        if (imageSize_.width() < Ortho2DSize_.width() && imageSize_.height() < Ortho2DSize_.height())
        {
            // 当图片宽、高都比窗口尺寸小时，只需计算贴图时在水平和垂直方向上的偏移即可
            x_offset = (Ortho2DSize_.width() - imageSize_.width()) / 2;
            y_offset = (Ortho2DSize_.height() - imageSize_.height()) / 2;
        }
        else
        {
            // 当图片宽或高比窗口尺寸大时，需再要先对图片做等比例缩放，计算贴图时在水平和垂直方向上的偏移
            float w_rate = float(Ortho2DSize_.width()) / imageSize_.width();
            float h_rate = float(Ortho2DSize_.height()) / imageSize_.height();
            if (w_rate < h_rate)
            {
                scale = w_rate;
                x_offset = 0;
                y_offset = (Ortho2DSize_.height() - imageSize_.height() * w_rate) / 2;
            }
            else
            {
                scale = h_rate;
                x_offset = (Ortho2DSize_.width() - imageSize_.width() * h_rate) / 2;
                y_offset = 0;
            }
        }
        offset_.setX(x_offset);
        offset_.setY(y_offset);

        adaptImageSize_.setWidth(Ortho2DSize_.width() - 2 * x_offset);
        adaptImageSize_.setHeight(Ortho2DSize_.height() - 2 * y_offset);

        // 顶点坐标保存的是相对于窗口的位置
        vertexPos_[Left_Bottom_X] = x_offset;
        vertexPos_[Left_Bottom_Y] = y_offset;
        vertexPos_[Right_Bottom_X] = Ortho2DSize_.width() - x_offset;
        vertexPos_[Right_Bottom_Y] = y_offset;
        vertexPos_[Right_Top_X] = Ortho2DSize_.width() - x_offset;
        vertexPos_[Right_Top_Y] = Ortho2DSize_.height() - y_offset;
        vertexPos_[Left_Top_X] = x_offset;
        vertexPos_[Left_Top_Y] = Ortho2DSize_.height() - y_offset;

        // 纹理坐标点保存的是相对于图片的位置
        texturePos_[Left_Bottom_X] = 0.0f;
        texturePos_[Left_Bottom_Y] = 0.0f;
        texturePos_[Right_Bottom_X] = 1.0f;
        texturePos_[Right_Bottom_Y] = 0.0f;
        texturePos_[Right_Top_X] = 1.0f;
        texturePos_[Right_Top_Y] = 1.0f;
        texturePos_[Left_Top_X] = 0.0f;
        texturePos_[Left_Top_Y] = 1.0f;

        initTextureFlag = true;
    }
    else
    {
        // 第一次显示用glTexImage2D方式显示，后面用glTexSubImage2D动态修改纹理数据的方式显示
        if (imageData_->depth() == 8)
            glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, imageSize_.width(), imageSize_.height(), GL_DEPTH_COMPONENT,
                GL_UNSIGNED_BYTE, imageData_->bits());
        else
            glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, imageSize_.width(), imageSize_.height(), GL_RGBA, GL_UNSIGNED_BYTE,
                imageData_->bits());
    }

    glEnable(GL_TEXTURE_2D);
    glBegin(GL_POLYGON);
    glTexCoord2d(texturePos_[Left_Bottom_X], texturePos_[Left_Bottom_Y]);
    glVertex2d(vertexPos_[Left_Bottom_X], vertexPos_[Left_Bottom_Y]);
    glTexCoord2d(texturePos_[Left_Top_X], texturePos_[Left_Top_Y]);
    glVertex2d(vertexPos_[Left_Top_X], vertexPos_[Left_Top_Y]);
    glTexCoord2d(texturePos_[Right_Top_X], texturePos_[Right_Top_Y]);
    glVertex2d(vertexPos_[Right_Top_X], vertexPos_[Right_Top_Y]);
    glTexCoord2d(texturePos_[Right_Bottom_X], texturePos_[Right_Bottom_Y]);
    glVertex2d(vertexPos_[Right_Bottom_X], vertexPos_[Right_Bottom_Y]);
    glEnd();
    glDisable(GL_TEXTURE_2D);
    QPainter painter(this);
    painter.save();
    painter.translate(offset_.x(), offset_.y());
    painter.scale(scaleVal_, scaleVal_);
    QPen pen = painter.pen();
    pen.setWidth(2);
    pen.setColor(QColor(0, 255, 0, 255));
    pen.setCosmetic(true);
    painter.setPen(pen);
    /*for (size_t i = 0; i < m_rects.size(); i++)
    {
        painter.drawRect(m_rects[i].x * scale, m_rects[i].y * scale, m_rects[i].w * scale, m_rects[i].h * scale);
    }
    for (size_t i = 0; i < m_lines.size(); i++)
    {
        painter.drawLine(m_lines[i].start_x * scale, m_lines[i].start_y * scale, m_lines[i].end_x * scale, m_lines[i].end_y * scale);
    }*/
    painter.restore();
    for (size_t i = 0; i < m_txts.size(); i++)
    {
        painter.save();
        QPen pen = painter.pen();
        pen.setWidth(2);
        pen.setColor(m_txts[i].color);
        painter.setPen(pen);
        QFont font(u8"黑体", 12);
        font.setBold(false);
        font.setUnderline(false);
        font.setItalic(false);
        painter.drawText(m_txts[i].x, m_txts[i].y, m_txts[i].text);
        painter.restore();
    }
    // 交换前后缓冲区
    //swapBuffers();
}

void WidgetGLImage::resizeGL(int w, int h)
{
    // 传入的w，h时widget控件的尺寸
    Ortho2DSize_.setWidth(w);
    Ortho2DSize_.setHeight(h);
    glViewport(0, 0, w, h);
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    glOrtho(0, Ortho2DSize_.width(), Ortho2DSize_.height(), 0, -1, 1);
    glMatrixMode(GL_MODELVIEW);
    update();
    //repaint();
}

// 鼠标滚轮实现图片倍率缩放,缩放间隔为0.1，调整绘制位置的偏移，调用paintGL重绘
void WidgetGLImage::wheelEvent(QWheelEvent* e)
{
    if (e->angleDelta().x() > 0)
    {
        scaleVal_ /= 0.9;
    }
    else
    {
        scaleVal_ *= 0.9;
    }

    uint16_t showImgWidth = adaptImageSize_.width() * scaleVal_;
    uint16_t showImgHeight = adaptImageSize_.height() * scaleVal_;

    int xoffset = (Ortho2DSize_.width() - showImgWidth) / 2;
    int yoffset = (Ortho2DSize_.height() - showImgHeight) / 2;
    offset_.setX(xoffset);
    offset_.setY(yoffset);

    vertexPos_[Left_Bottom_X] = xoffset;
    vertexPos_[Left_Bottom_Y] = yoffset;
    vertexPos_[Right_Bottom_X] = xoffset + showImgWidth;
    vertexPos_[Right_Bottom_Y] = yoffset;
    vertexPos_[Right_Top_X] = xoffset + showImgWidth;
    vertexPos_[Right_Top_Y] = yoffset + showImgHeight;
    vertexPos_[Left_Top_X] = xoffset;
    vertexPos_[Left_Top_Y] = yoffset + showImgHeight;

    paintGL();
    update();
    // repaint();
}

// 实现鼠标拖拽图片，鼠标在拖拽过程中会反复调用此函数，因此一个连续的拖拽过程可以
// 分解为多次移动的过程，每次移动都是在上一个位置的基础上进行一次位置调节
void WidgetGLImage::mouseMoveEvent(QMouseEvent* e)
{
    if (dragFlag_)
    {
        int scaledMoveX = e->x() - dragPos_.x();
        int scaledMoveY = e->y() - dragPos_.y();
        offset_.setX(offset_.x() + scaledMoveX);
        offset_.setY(offset_.y() + scaledMoveY);
        vertexPos_[Left_Bottom_X] += scaledMoveX;
        vertexPos_[Left_Bottom_Y] += scaledMoveY;
        vertexPos_[Left_Top_X] += scaledMoveX;
        vertexPos_[Left_Top_Y] += scaledMoveY;
        vertexPos_[Right_Top_X] += scaledMoveX;
        vertexPos_[Right_Top_Y] += scaledMoveY;
        vertexPos_[Right_Bottom_X] += scaledMoveX;
        vertexPos_[Right_Bottom_Y] += scaledMoveY;

        dragPos_.setX(e->x());
        dragPos_.setY(e->y());
        paintGL();
        update();
        // repaint();
    }
}

void WidgetGLImage::mousePressEvent(QMouseEvent* e)
{
    if (scaleVal_ > 0)
    {
        dragFlag_ = true;
        dragPos_.setX(e->x());
        dragPos_.setY(e->y());
    }
}

void WidgetGLImage::mouseReleaseEvent(QMouseEvent* e)
{
    dragFlag_ = false;
}

// 双击实现原比例显示，缩放倍率设置为1.0
void WidgetGLImage::mouseDoubleClickEvent(QMouseEvent* e)
{
    scaleVal_ = 1.0;

    uint16_t showImgWidth = adaptImageSize_.width() * scaleVal_;
    uint16_t showImgHeight = adaptImageSize_.height() * scaleVal_;

    int xoffset = (Ortho2DSize_.width() - showImgWidth) / 2;
    int yoffset = (Ortho2DSize_.height() - showImgHeight) / 2;
    offset_.setX(xoffset);
    offset_.setY(yoffset);

    vertexPos_[Left_Bottom_X] = xoffset;
    vertexPos_[Left_Bottom_Y] = yoffset;
    vertexPos_[Right_Bottom_X] = xoffset + showImgWidth;
    vertexPos_[Right_Bottom_Y] = yoffset;
    vertexPos_[Right_Top_X] = xoffset + showImgWidth;
    vertexPos_[Right_Top_Y] = yoffset + showImgHeight;
    vertexPos_[Left_Top_X] = xoffset;
    vertexPos_[Left_Top_Y] = yoffset + showImgHeight;

    paintGL();
    update();
}
