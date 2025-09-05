/**
 * @file main.cpp
 * @author Wei Li (Ithaca) (weili_yzzca@163.com)
 * @brief 
 * @version 0.1
 * @date 2025-08-31
 * 
 * @copyright Copyright (c) 2025
 * 
 */

#include "test/TestImageEdgeDetector.hpp"
#include "test/TestImageEmphasize.hpp"
#include "test/TestImageEnhancementTAGC.hpp"

// ------------------------------------
int main(int argc, const char *argv[])
{
    // Ithaca::TestImageEmphasize();
    // Ithaca::TestImageEnhancementTAGC();
    // Ithaca::TestImageEnhancementTAGCFolder();
    Ithaca::TestImageEdgeDetector();

    return 0;
}
