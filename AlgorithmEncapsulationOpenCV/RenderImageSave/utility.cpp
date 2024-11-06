#include "utility.h"

void __DrawText__::GetStringSize(HDC hDC, const char *str, int *w, int *h)
{
    SIZE size;
    GetTextExtentPoint32A(hDC, str, strlen(str), &size);
    if (w != 0)
        *w = size.cx;
    if (h != 0)
        *h = size.cy;
}

void __DrawText__::PutTextExt(cv::Mat &dst, const char *str, cv::Point org, cv::Scalar color, int fontSize,
                              const char *fn, bool italic, bool underline)
{
    CV_Assert(dst.data != 0 && (dst.channels() == 1 || dst.channels() == 3));

    int x, y, r, b;
    if (org.x > dst.cols || org.y > dst.rows)
        return;
    x = org.x < 0 ? -org.x : 0;
    y = org.y < 0 ? -org.y : 0;

    LOGFONTA lf;
    lf.lfHeight         = -fontSize;
    lf.lfWidth          = 0;
    lf.lfEscapement     = 0;
    lf.lfOrientation    = 0;
    lf.lfWeight         = 5;
    lf.lfItalic         = italic;    //斜体
    lf.lfUnderline      = underline; //下划线
    lf.lfStrikeOut      = 0;
    lf.lfCharSet        = DEFAULT_CHARSET;
    lf.lfOutPrecision   = 0;
    lf.lfClipPrecision  = 0;
    lf.lfQuality        = PROOF_QUALITY;
    lf.lfPitchAndFamily = 0;
    strcpy_s(lf.lfFaceName, fn);

    HFONT hf       = CreateFontIndirectA(&lf);
    HDC   hDC      = CreateCompatibleDC(0);
    HFONT hOldFont = (HFONT)SelectObject(hDC, hf);

    int  strBaseW = 0, strBaseH = 0;
    int  singleRow = 0;
    char buf[1 << 12];
    strcpy_s(buf, str);
    char *bufT[1 << 12]; // 这个用于分隔字符串后剩余的字符，可能会超出。
    //处理多行
    {
        int nnh = 0;
        int cw, ch;

        const char *ln = strtok_s(buf, "\n", bufT);
        while (ln != 0)
        {
            GetStringSize(hDC, ln, &cw, &ch);
            strBaseW = std::max(strBaseW, cw);
            strBaseH = std::max(strBaseH, ch);

            ln = strtok_s(0, "\n", bufT);
            nnh++;
        }
        singleRow = strBaseH;
        strBaseH *= nnh;
    }

    if (org.x + strBaseW < 0 || org.y + strBaseH < 0)
    {
        SelectObject(hDC, hOldFont);
        DeleteObject(hf);
        DeleteObject(hDC);
        return;
    }

    r     = org.x + strBaseW > dst.cols ? dst.cols - org.x - 1 : strBaseW - 1;
    b     = org.y + strBaseH > dst.rows ? dst.rows - org.y - 1 : strBaseH - 1;
    org.x = org.x < 0 ? 0 : org.x;
    org.y = org.y < 0 ? 0 : org.y;

    BITMAPINFO        bmp = {0};
    BITMAPINFOHEADER &bih = bmp.bmiHeader;
    int strDrawLineStep   = strBaseW * 3 % 4 == 0 ? strBaseW * 3 : (strBaseW * 3 + 4 - ((strBaseW * 3) % 4));

    bih.biSize         = sizeof(BITMAPINFOHEADER);
    bih.biWidth        = strBaseW;
    bih.biHeight       = strBaseH;
    bih.biPlanes       = 1;
    bih.biBitCount     = 24;
    bih.biCompression  = BI_RGB;
    bih.biSizeImage    = strBaseH * strDrawLineStep;
    bih.biClrUsed      = 0;
    bih.biClrImportant = 0;

    void   *pDibData = 0;
    HBITMAP hBmp     = CreateDIBSection(hDC, &bmp, DIB_RGB_COLORS, &pDibData, 0, 0);

    CV_Assert(pDibData != 0);
    HBITMAP hOldBmp = (HBITMAP)SelectObject(hDC, hBmp);

    //color.val[2], color.val[1], color.val[0]
    SetTextColor(hDC, RGB(255, 255, 255));
    SetBkColor(hDC, 0);
    //SetStretchBltMode(hDC, COLORONCOLOR);

    strcpy_s(buf, str);
    const char *ln       = strtok_s(buf, "\n", bufT);
    int         outTextY = 0;
    while (ln != 0)
    {
        TextOutA(hDC, 0, outTextY, ln, strlen(ln));
        outTextY += singleRow;
        ln = strtok_s(0, "\n", bufT);
    }
    uchar         *dstData = (uchar *)dst.data;
    int            dstStep = dst.step / sizeof(dstData[0]);
    unsigned char *pImg    = (unsigned char *)dst.data + org.x * dst.channels() + org.y * dstStep;
    unsigned char *pStr    = (unsigned char *)pDibData + x * 3;
    for (int tty = y; tty <= b; ++tty)
    {
        unsigned char *subImg = pImg + (tty - y) * dstStep;
        unsigned char *subStr = pStr + (strBaseH - tty - 1) * strDrawLineStep;
        for (int ttx = x; ttx <= r; ++ttx)
        {
            for (int n = 0; n < dst.channels(); ++n)
            {
                double vtxt = subStr[n] / 255.0;
                int    cvv  = vtxt * color.val[n] + (1 - vtxt) * subImg[n];
                subImg[n]   = cvv > 255 ? 255 : (cvv < 0 ? 0 : cvv);
            }

            subStr += 3;
            subImg += dst.channels();
        }
    }

    SelectObject(hDC, hOldBmp);
    SelectObject(hDC, hOldFont);
    DeleteObject(hf);
    DeleteObject(hBmp);
    DeleteDC(hDC);
}



namespace Utility
{
    bool BufferToHalcon(CameraData* buffer, HalconCpp::HImage* image)
    {
        if (nullptr == buffer->pImgBuf
            || buffer->nImgWidth <= 1
            || buffer->nImgHeight <= 1)
        {
            return false;
        }

        HalconCpp::GenImage1(image, "byte", (int)buffer->nImgWidth, (int)buffer->nImgHeight, (Hlong)buffer->pImgBuf);

        return true;
    }

    bool HImage2QImage(HalconCpp::HImage& from, QImage& to)
    {
        try
        {
            Hlong width;
            Hlong height;
            from.GetImageSize(&width, &height);

            HalconCpp::HTuple channels = from.CountChannels();
            HalconCpp::HTuple type = from.GetImageType();

            if (strcmp(type[0].S(), "byte")) // 如果不是 byte 类型，则失败
            {
                return false;
            }

            QImage::Format format;
            switch (channels[0].I())
            {
            case 1:
                format = QImage::Format_Grayscale8;
                break;
            case 3:
                format = QImage::Format_RGB32;
                break;
            default:
                return false;
            }

            if (to.width() != width || to.height() != height || to.format() != format)
            {
                to = QImage(static_cast<int>(width),
                    static_cast<int>(height),
                    format);
            }
            HalconCpp::HString Type;
            if (channels[0].I() == 1)
            {
                unsigned char* pSrc = reinterpret_cast<unsigned char*>(from.GetImagePointer1(&Type, &width, &height));
                memcpy(to.bits(), pSrc, static_cast<size_t>(width) * static_cast<size_t>(height));
                return true;
            }
            else if (channels[0].I() == 3)
            {
                uchar* R, * G, * B;
                from.GetImagePointer3(reinterpret_cast<void**>(&R),
                    reinterpret_cast<void**>(&G),
                    reinterpret_cast<void**>(&B), &Type, &width, &height);

                for (int row = 0; row < height; row++)
                {
                    QRgb* line = reinterpret_cast<QRgb*>(to.scanLine(row));
                    for (int col = 0; col < width; col++)
                    {
                        line[col] = qRgb(*R++, *G++, *B++);
                    }
                }
                return true;
            }

            return false;
        }
        catch (...)
        {
            return false;
        }      
    }

    bool QImage2HImage(QImage& from, HalconCpp::HImage& to)
    {
        try 
        {
            if (from.isNull()) return false;

            int width = from.width(), height = from.height();
            QImage::Format format = from.format();

            if (format == QImage::Format_RGB32 ||
                format == QImage::Format_ARGB32 ||
                format == QImage::Format_ARGB32_Premultiplied)
            {
                to.GenImageInterleaved(from.bits(), "rgbx", width, height, 0, "byte", width, height, 0, 0, 8, 0);
                return true;
            }
            else if (format == QImage::Format_RGB888)
            {
                to.GenImageInterleaved(from.bits(), "rgb", width, height, 0, "byte", width, height, 0, 0, 8, 0);
                return true;
            }
            else if (format == QImage::Format_Grayscale8 || format == QImage::Format_Indexed8)
            {
                to.GenImage1("byte", width, height, from.bits());
                return true;
            }
            return false;
        }
        catch (...)
        {
            return false;
        }        
    }

    bool SdImage8Grey2QImage(SdVisionCore::SdImage8Grey& from, QImage& to)
    {
        to = QImage((uchar*)from.Data, from.Width, from.Height, QImage::Format_Indexed8);
        return true;
    }

    bool QImage2SdImage8Grey(QImage& from, SdVisionCore::SdImage8Grey& to)
    {
        to.Data = from.bits();
        to.Width = from.width();
        to.Height = from.height();
        to.Stride = to.Width;
        return true;
    }

    void list_image_files(QString dirPath, QStringList* imageFileNames)
    {
        if (dirPath.isEmpty())
            return;
        QDir dir(dirPath);
        dir.setFilter(QDir::Files);
        (*imageFileNames) = dir.entryList();
    }

    void opencvImage2qt(cv::Mat& in_image, QImage& image)
    {
        auto type = in_image.type();
        auto width = in_image.cols;
        auto height = in_image.rows;

        switch (type)
        {
        case CV_8UC4:
        {
            image = QImage(in_image.data, width, height,
                in_image.step, QImage::Format_RGB32);
        }
        case CV_8UC3:
        {
            image = QImage(in_image.data, width, height,
                in_image.step, QImage::Format_RGB888);
        }

        case CV_8UC1:
        {
            static QVector<QRgb>  sColorTable;
            // only create our color table once
            if (sColorTable.isEmpty())
            {
                for (int i = 0; i < 256; ++i)
                    sColorTable.push_back(qRgb(i, i, i));
            }

            image = QImage(in_image.data, width, height,
                in_image.step, QImage::Format_Indexed8).copy();
            image.setColorTable(sColorTable);
        }

        default:
        {
            // spdlog::info();
            std::cout << "The image channel is " << in_image.channels() << " NOT implemention\n";
            break;
        }
        }
    }

    void qtImage2opencv(QImage& in_image, cv::Mat& image)
    {
        auto height = in_image.height();
        auto width = in_image.width();
        auto format = in_image.format();

        switch (format)
        {
        case QImage::Format_ARGB32:
        case QImage::Format_RGB32:
        case QImage::Format_ARGB32_Premultiplied:
        {
            image = cv::Mat(height, width, CV_8UC4,
                (void*)in_image.constBits(),
                in_image.bytesPerLine());
            break;
        }

        case QImage::Format_RGB888:
        {
            image = cv::Mat(height, width, CV_8UC3,
                (void*)in_image.constBits(),
                in_image.bytesPerLine());

            cv::cvtColor(image, image, cv::COLOR_BGR2RGB);
            break;
        }
        case QImage::Format_Indexed8:
        {
            image = cv::Mat(height, width, CV_8UC1,
                (void*)in_image.constBits(),
                in_image.bytesPerLine());

            break;
        }
        default:
        {
            // spdlog::info();
            std::cout << "The image format is " << format << " NOT implemention\n";
            break;
        }
        }
    }
}

