#include "demo_player.h"


#include "globaldef.h"
#include "ws_info.h"
#include "log_macros.h"
#include <iostream>

using namespace std;
bool DemoPlayer::behave(Cmd & cmd) {
  switch(WSinfo::ws->play_mode){
  case PM_PlayOn:
    cmd.cmd_body.set_dash(100);
    break;
  default:
    cmd.cmd_body.set_turn(1.57);
  }

  LOG_DEB(0, << "now in cycle : " << WSinfo::ws->time);

  LOG_DEB(0, << _2D << VC2D(WSinfo::me->pos, 2, "ff0000") );
  if ( WSinfo::ws->time % 100 == 1 ) {
    cout << "\nnow in cycle " << WSinfo::ws->time << flush;
  }

  return true;
}
