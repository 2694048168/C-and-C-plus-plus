/**
 * @file main.cpp
 * @author Wei Li (weili_yzzcq@163.com)
 * @brief Printing a list of movies to a PDF
 * @version 0.1
 * @date 2024-01-23
 * 
 * @copyright Copyright (c) 2024
 * 
 */

#include "PDFWriter/PDFPage.h"
#include "PDFWriter/PDFUsedFont.h"
#include "PDFWriter/PDFWriter.h"
#include "PDFWriter/PageContentContext.h"
#include "movies.h"

#include <iomanip>
#include <iostream>
#include <string>
#include <string_view>


#ifdef _WIN32
static const std::string fonts_dir = R"(c:\windows\fonts\)";
#elif defined(__APPLE__)
static const std::string fonts_dir = R"(/Library/Fonts/)";
#else
static const std::string fonts_dir = R"(/usr/share/fonts)";
#endif

/**
 * @brief Printing a list of movies to a PDF
 * 
 * Write a program that can print to a PDF file a list of movies in a tabular form,
 * with the following requirements:
 * 1. There must be a heading to the list with the content List of movies. 
 *    This must appear only on the first page of the document.
 * 2. For each movie, it should display the title, the release year, and the length.
 * 3. The title, followed by the release year in parentheses, must be left-aligned.
 * 4. The length, in hours and minutes (for example, 2:12), must be right-aligned.
 * 5. There must be a line above and below the movie listing on each page.
 * 
 */

/**
 * @brief Solution:
 https://github.com/PacktPublishing/The-Modern-Cpp-Challenge/tree/master/Chapter09/problem_77
------------------------------------------------------ */
void print_pdf(const movie_list &movies, std::string_view path)
{
    const int height      = 842;
    const int width       = 595;
    const int left        = 60;
    const int top         = 770;
    const int right       = 535;
    const int bottom      = 60;
    const int line_height = 28;

    PDFWriter pdf;

    pdf.StartPDF(path.data(), ePDFVersion13);
    auto font = pdf.GetFontForFile(fonts_dir + "Arial.ttf");

    AbstractContentContext::GraphicOptions pathStrokeOptions(AbstractContentContext::eStroke,
                                                             AbstractContentContext::eRGB, 0xff000000, 1);

    PDFPage            *page    = nullptr;
    PageContentContext *context = nullptr;
    int                 index   = 0;
    for (size_t i = 0; i < movies.size(); ++i)
    {
        index = i % 25;
        if (index == 0)
        {
            if (page != nullptr)
            {
                DoubleAndDoublePairList pathPoints;
                pathPoints.push_back(DoubleAndDoublePair(left, bottom));
                pathPoints.push_back(DoubleAndDoublePair(right, bottom));
                context->DrawPath(pathPoints, pathStrokeOptions);

                pdf.EndPageContentContext(context);
                pdf.WritePageAndRelease(page);
            }

            page = new PDFPage();
            page->SetMediaBox(PDFRectangle(0, 0, width, height));
            context = pdf.StartPageContentContext(page);

            {
                DoubleAndDoublePairList pathPoints;
                pathPoints.push_back(DoubleAndDoublePair(left, top));
                pathPoints.push_back(DoubleAndDoublePair(right, top));
                context->DrawPath(pathPoints, pathStrokeOptions);
            }
        }

        if (i == 0)
        {
            const AbstractContentContext::TextOptions textOptions(font, 26, AbstractContentContext::eGray, 0);

            context->WriteText(left, top + 15, "List of movies", textOptions);
        }

        auto textw = 0;
        {
            const AbstractContentContext::TextOptions textOptions(font, 20, AbstractContentContext::eGray, 0);

            context->WriteText(left, top - 20 - line_height * index, movies[i].title, textOptions);
            auto textDimensions = font->CalculateTextDimensions(movies[i].title, 20);
            textw               = textDimensions.width;
        }

        {
            const AbstractContentContext::TextOptions textOptions(font, 16, AbstractContentContext::eGray, 0);

            context->WriteText(left + textw + 5, top - 20 - line_height * index,
                               " (" + std::to_string(movies[i].year) + ")", textOptions);

            std::stringstream s;
            s << movies[i].length / 60 << ':' << std::setw(2) << std::setfill('0') << movies[i].length % 60;

            context->WriteText(right - 30, top - 20 - line_height * index, s.str(), textOptions);
        }
    }

    DoubleAndDoublePairList pathPoints;
    pathPoints.push_back(DoubleAndDoublePair(left, top - line_height * (index + 1)));
    pathPoints.push_back(DoubleAndDoublePair(right, top - line_height * (index + 1)));
    context->DrawPath(pathPoints, pathStrokeOptions);

    if (page != nullptr)
    {
        pdf.EndPageContentContext(context);
        pdf.WritePageAndRelease(page);
    }

    pdf.EndPDF();
}

// ------------------------------
int main(int argc, char **argv)
{
    movie_list movies{
        { 1,                      "The Matrix", 1999, 136},
        { 2,                    "Forrest Gump", 1994, 142},
        { 3,                 "The Truman Show", 1998, 103},
        { 4,        "The Pursuit of Happyness", 2006, 117},
        { 5,                      "Fight Club", 1999, 139},
        { 6,            "Inglourious Basterds", 2009, 153},
        { 7,           "The Dark Knight Rises", 2012, 164},
        { 8,                 "The Dark Knight", 2008, 152},
        { 9,                    "Pulp Fiction", 1994, 154},
        {10,                       "Inception", 2010, 148},
        {11,        "The Shawshank Redemption", 1994, 142},
        {12,        "The Silence of the Lambs", 1991, 118},
        {13,                    "Philadelphia", 1993, 125},
        {14, "One Flew Over the Cuckoo's Nest", 1975,  80},
        {15,                         "Memento", 2000, 113},
        {16,                   "Trainspotting", 1996,  94},
        {17,                           "Fargo", 1998,  98},
        {18,                  "Reservoir Dogs", 1992,  99},
        {19,                    "The Departed", 2006, 151},
        {20,                           "Se7en", 1995, 127},
        {21,              "American History X", 1998, 119},
        {22,         "Silver Linings Playbook", 2012, 122},
        {23,           "2001: A Space Odyssey", 1968, 149},
        {24, "Monty Python and the Holy Grail", 1975,  91},
        {25,                   "Life of Brian", 1979,  94},
        {26,                 "Children of Men", 2006, 109},
        {27,                        "Sin City", 2005, 124},
        {28,               "L.A. Confidential", 1997, 138},
        {29,                  "Shutter Island", 2010, 138},
    };

    print_pdf(movies, "movies.pdf");

    return 0;
}
