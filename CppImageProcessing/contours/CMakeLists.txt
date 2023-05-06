# 把需要拷贝的资源路径都放在 "IMAGES" 变量里面
file(GLOB IMAGES 
    "${PROJECT_SOURCE_DIR}/images/HappyFish.jpg"
    "${PROJECT_SOURCE_DIR}/images/stuff.jpg"
    "${PROJECT_SOURCE_DIR}/images/Moments_Source_Image.jpg"
)
# 把 "IMAGES" 指代的所有内容都拷贝到可执行文件目录下
if(CMAKE_SYSTEM_NAME MATCHES "Windows")
    file(COPY ${IMAGES} DESTINATION "${EXECUTABLE_OUTPUT_PATH}/Debug")
else()
    file(COPY ${IMAGES} DESTINATION "${EXECUTABLE_OUTPUT_PATH}")
endif()

add_executable(FindingContours)
target_sources(FindingContours
    PRIVATE
        "finding_contours.cpp"
)
target_link_libraries(FindingContours
    PRIVATE
        "${OpenCV_LIBS}"
)

add_executable(ConvexHull)
target_sources(ConvexHull
    PRIVATE
        "convex_hull.cpp"
)
target_link_libraries(ConvexHull
    PRIVATE
        "${OpenCV_LIBS}"
)

add_executable(BoundingBoxesCircles)
target_sources(BoundingBoxesCircles
    PRIVATE
        "bounding_boxes_circles.cpp"
)
target_link_libraries(BoundingBoxesCircles
    PRIVATE
        "${OpenCV_LIBS}"
)

add_executable(BoundingRotatedEllipses)
target_sources(BoundingRotatedEllipses
    PRIVATE
        "bounding_rotated_ellipses.cpp"
)
target_link_libraries(BoundingRotatedEllipses
    PRIVATE
        "${OpenCV_LIBS}"
)

add_executable(ImageMoments)
target_sources(ImageMoments
    PRIVATE
        "image_moments.cpp"
)
target_link_libraries(ImageMoments
    PRIVATE
        "${OpenCV_LIBS}"
)

add_executable(PointPolygon)
target_sources(PointPolygon
    PRIVATE
        "point_polygon.cpp"
)
target_link_libraries(PointPolygon
    PRIVATE
        "${OpenCV_LIBS}"
)
