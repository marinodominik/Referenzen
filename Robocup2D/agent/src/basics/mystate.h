#ifndef _MYSTATE_H_
#define _MYSTATE_H_

#include "angle.h"
#include "Cmd.h"
#include "Vector.h"
#include "ws_info.h"

class MyState {

 public:
  Vector my_vel;
  Vector my_pos;
  Vector ball_vel;
  Vector ball_pos;
  ANGLE  my_angle;
  Vector op_pos;
  ANGLE op_bodydir;
  int op_bodydir_age;
  PPlayer op;
  PPlayer me;

  void get_from_WS();

  MyState();
};

#endif
