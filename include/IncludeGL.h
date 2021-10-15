//===----------- IncludeGL.h - Utility functions used by OpenGL. ----------===//
//
// Part of the SuperConga Project, under the GNU LGPL v3 license or higher.
// See LICENSE.txt for license information.
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Utility functions used in the OpenGL live visualization.
///
//===----------------------------------------------------------------------===//
#ifndef INCLUDE_GL_H_
#define INCLUDE_GL_H_

#ifdef __linux__
#include <GL/freeglut.h>
#else
#include <GLUT/glut.h>
#endif

namespace conga {
// Key logging functions
namespace Key {
unsigned char g_key_pressed;
int g_key_pressed_x = -1;
int g_key_pressed_y = -1;

void keyboardFunc(unsigned char key, int x, int y) {
  g_key_pressed = key;
  g_key_pressed_x = x;
  g_key_pressed_y = y;
}

unsigned char getKeyPressed() {
  unsigned char key = g_key_pressed;
  g_key_pressed = '\0';
  return key;
}

unsigned char getKeyPressedCoords(int &x, int &y) {
  x = g_key_pressed_x;
  y = g_key_pressed_y;

  return getKeyPressed();
}
} // namespace Key

void handleResize(int w, int h) {
  // viewport
  glViewport(0, 0, w, h);

  // projection
  glMatrixMode(GL_PROJECTION);
  glLoadIdentity();
  gluPerspective(45.0, (GLfloat)w / (GLfloat)h, 1.0, 100.0);
}

int initGL(int argc, char *argv[], const int nx, const int ny) {
  glutInit(&argc, argv);

  glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGB | GLUT_DEPTH);
  glutInitWindowSize(nx, ny);
  const int windowID = glutCreateWindow("Solution");

  glutReshapeFunc(handleResize);
  glutKeyboardFunc(Key::keyboardFunc);

  // default initialization
  glClearColor(0.0, 0.0, 0.0, 1.0);
  glDisable(GL_DEPTH_TEST);

  glutSetOption(GLUT_ACTION_ON_WINDOW_CLOSE, GLUT_ACTION_CONTINUE_EXECUTION);

  return windowID;
}
} // namespace conga

#endif // INCLUDE_GL_H_
