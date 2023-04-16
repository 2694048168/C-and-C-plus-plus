#ifndef _UTILS_HPP_
#define _UTILS_HPP_

#include <GLFW/glfw3.h>
#include <glm/glm.hpp>

// settings
const unsigned int SCR_WIDTH = 800;
const unsigned int SCR_HEIGHT = 600;

void framebuffer_size_callback(GLFWwindow* window, int width, int height);
void processInput(GLFWwindow *window);

#endif /* _UTILS_HPP_ */