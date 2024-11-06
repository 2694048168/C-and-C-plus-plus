#pragma once

#include <string>

namespace RenderGraphicsSave {

//#define RENDER_COLOR_RED cv::Scalar{0, 0, 255}
#define RENDER_COLOR_RED {0, 0, 255}

//#define RENDER_COLOR_GREEN cv::Scalar{0, 255, 0}
#define RENDER_COLOR_GREEN {0, 255, 0}

//#define RENDER_COLOR_BLUE cv::Scalar{255, 0, 0}
#define RENDER_COLOR_BLUE {255, 0, 0}

//#define RENDER_COLOR_YELLOW cv::Scalar{0, 255, 255}
#define RENDER_COLOR_YELLOW {0, 255, 255}

//#define RENDER_COLOR_BLACK cv::Scalar{0, 0, 0}
#define RENDER_COLOR_BLACK {0, 0, 0}

//#define RENDER_COLOR_WHITE cv::Scalar{255, 255, 255}
#define RENDER_COLOR_WHITE {255, 255, 255}

enum class RenderColor : char
{
    RED    = 0,
    GREEN  = 1,
    BLUE   = 2,
    YELLOW = 3,
    BLACK  = 4,
    WHITE  = 5,

    NUM_COLOR
};

struct TextInfo
{
    int         x;
    int         y;
    std::string text;
    RenderColor color          = RenderColor::GREEN; // default color
    int         text_thickness = 100;                // unit: pixel
};

struct LineInfo
{
    int         start_x;
    int         start_y;
    int         end_x;
    int         end_y;
    RenderColor color          = RenderColor::GREEN; // default color
    int         line_thickness = 12;                 // unit: pixel
};

struct RectInfo
{
    int         x;
    int         y;
    int         w;
    int         h;
    RenderColor color          = RenderColor::GREEN; // default color
    int         rect_thickness = 12;                 // unit: pixel
};

struct CircleInfo
{
    int         center_x;
    int         center_y;
    int         radius;
    RenderColor color            = RenderColor::GREEN; // default color
    int         circle_thickness = 8;                  // unit: pixel
};
} // namespace RenderGraphicsSave
