/**
 * @file main.cpp
 * @author Wei Li (weili_yzzcq@163.com)
 * @brief Creating a PDF from a collection of images
 * @version 0.1
 * @date 2024-01-24
 * 
 * @copyright Copyright (c) 2024
 * 
 */

/**
 * @brief Creating a PDF from a collection of images
 * 
 * Write a program that can create a PDF document that contains images 
 * from a user-specified directory. The images must be displayed one after another.
 * If an image does not fit on the remainder of a page, 
 * it must be placed on the next page.
 * 
 */

/**
 * @brief Solution:
 https://github.com/PacktPublishing/The-Modern-Cpp-Challenge/tree/master/Chapter09/problem_78
------------------------------------------------------ */
#include "PDFWriter/PDFFormXObject.h"
#include "PDFWriter/PDFPage.h"
#include "PDFWriter/PDFWriter.h"
#include "PDFWriter/PageContentContext.h"

#include <iostream>
#include <string_view>


#ifdef USE_BOOST_FILESYSTEM
#    include <boost/filesystem/operations.hpp>
#    include <boost/filesystem/path.hpp>

namespace fs = boost::filesystem;
#else
#    include <filesystem>
#    ifdef FILESYSTEM_EXPERIMENTAL
namespace fs = std::experimental::filesystem;
#    else
namespace fs = std::filesystem;
#    endif
#endif

std::vector<std::string> get_images(const fs::path &dirpath)
{
    std::vector<std::string> paths;

    for (const auto &p : fs::directory_iterator(dirpath))
    {
        if (p.path().extension() == ".jpg")
            paths.push_back(p.path().string());
    }

    return paths;
}

void print_pdf(const fs::path &pdfpath, const fs::path &dirpath)
{
    const int height = 842;
    const int width  = 595;
    const int margin = 20;

    auto image_paths = get_images(dirpath);

    PDFWriter pdf;
    pdf.StartPDF(pdfpath.string(), ePDFVersion13);

    PDFPage            *page    = nullptr;
    PageContentContext *context = nullptr;

    auto top = height - margin;
    for (size_t i = 0; i < image_paths.size(); ++i)
    {
        auto dims = pdf.GetImageDimensions(image_paths[i]);

        if (i == 0 || top - dims.second < margin)
        {
            if (page != nullptr)
            {
                pdf.EndPageContentContext(context);
                pdf.WritePageAndRelease(page);
            }

            page = new PDFPage();
            page->SetMediaBox(PDFRectangle(0, 0, width, height));
            context = pdf.StartPageContentContext(page);

            top = height - margin;
        }

        context->DrawImage(margin, top - dims.second, image_paths[i]);

        top -= dims.second + margin;
    }

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
    print_pdf("sample.pdf", "res");

    return 0;
}
