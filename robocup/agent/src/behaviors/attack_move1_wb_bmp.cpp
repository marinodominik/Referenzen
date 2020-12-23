#include "attack_move1_wb_bmp.h"
#include "ws_info.h"
#include "ws_memory.h"
#include "log_macros.h"
#include "mdp_info.h"
#include <stdlib.h>
#include <stdio.h>
#include "tools.h"
#include "valueparser.h"
#include "options.h"
#include "log_macros.h"
#include "geometry2d.h"

#if 1
#define DBLOG_POL(LLL,XXX) LOG_POL(LLL,<<"ATTACK_MOVE1_WB: " XXX)
#define DBLOG_DRAW(LLL,XXX) LOG_POL(LLL,<<_2D <<XXX)
#define DBLOG_ERR(LLL,XXX) LOG_ERR(LLL,XXX)
#define MYGETTIME (Tools::get_current_ms_time())
#else
#define DBLOG_POL(LLL,XXX)
#define DBLOG_DRAW(LLL,XXX)
#define DBLOG_ERR(LLL,XXX) 
#define MYGETTIME (0)
#endif


/* constructor method */
Attack_Move1_Wb::Attack_Move1_Wb() {
  ValueParser vp(CommandLineOptions::policy_conf,"attack_move1_wb");
}

Attack_Move1_Wb::~Attack_Move1_Wb() {
}


bool Attack_Move1_Wb::do_move(){
  DBLOG_POL(0, << "check attack move 1 ");
  return false;
}



bool Attack_Move1_Wb::get_cmd(Cmd & cmd) {
  DBLOG_POL(0, << "get_cmd");
  return false;
}


