#ifndef _DEMO_PLAYER_H_
#define _DEMO_PLAYER_H_

#include "Cmd.h"

class DemoPlayer {
public:
  DemoPlayer() { };
  virtual bool behave(Cmd & cmd);
};


#endif
