#ifndef INCLUDEGL_H
#define INCLUDEGL_H

#ifdef __linux__
#include <GL/freeglut.h>
#else
#include <GLUT/glut.h>
#endif

#define REFRESH_DELAY 0 // ms

void display();
void cleanup();
void timerEvent(int);
void handleResize(int, int);

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

// void initGL( int argc, char *argv[] ) {
int initGL(int argc, char *argv[], int nx, int ny) {
  glutInit(&argc, argv);

  glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGB | GLUT_DEPTH);
  glutInitWindowSize(nx, ny);
  int winId = glutCreateWindow("Solution");

  glutDisplayFunc(display);
  glutReshapeFunc(handleResize);
  glutTimerFunc(REFRESH_DELAY, timerEvent, 0);
  glutKeyboardFunc(Key::keyboardFunc);

  // default initialization
  glClearColor(0.0, 0.0, 0.0, 1.0);
  glDisable(GL_DEPTH_TEST);

  atexit(cleanup);
  return winId;
}

void handleResize(int w, int h) {
  // viewport
  glViewport(0, 0, w, h);

  // projection
  glMatrixMode(GL_PROJECTION);
  glLoadIdentity();
  gluPerspective(45.0, (GLfloat)w / (GLfloat)h, 1.0, 100.0);
}

void timerEvent(int value) {
  glutPostRedisplay();
  glutTimerFunc(REFRESH_DELAY, timerEvent, 0);
}

#endif // INCLUDEGL_H
