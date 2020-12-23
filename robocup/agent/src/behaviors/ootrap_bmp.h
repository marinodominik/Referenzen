#ifndef _ootrap_nb_H_
#define _ootrap_nb_H_

#include "base_bm.h"
#include "skills/neuro_go2pos_bms.h"
#include "skills/neuro_intercept_bms.h"
#include "skills/dribble_straight_bms.h"
#include "skills/neuro_kick05_bms.h"
#include "skills/face_ball_bms.h"
#include "play_pass_bms.h"
#include "intention.h"
#include "neuro_wball.h"


#define MAX_STEPS2INTERCEPT 40



  
class OvercomeOffsideTrap: public BodyBehavior{
  
 
  NeuroGo2Pos * go2pos;
  NeuroIntercept * neuroIntercept;
  NeuroKick05 * neuroKick;
  FaceBall *faceBall;
  //PlayPass *playPass;
  OneTwoHoldTurn *  onetwoholdturn;

  
  //:begin BOTH
  int mode;
  PPlayer player;
  Vector playerpos;
  PPlayer taropp;
  Vector pos1;
  Vector pos2;
  Vector pos3;
  double speed;
  int start_intercept_time;
  
  Vector ball_pos;
  
  int say_time;
  
  
  
  double dist_ol;
  bool got_already;
  bool go_left;
  int run_time;
  

//:begin PLAYER 1
bool p1_hold(Cmd & cmd);
bool p1_pass(Cmd & cmd);
bool p1_run(Cmd & cmd);
bool p1_intercept(Cmd & cmd);
//PLAYER 1    :end



//:begin PLAYER 2
bool p2_go2pos2(Cmd & cmd);
bool p2_wait(Cmd & cmd);
bool p2_intercept(Cmd & cmd);
bool p2_pass(Cmd & cmd);
//PLAYER 2    :end


public:
  virtual ~OvercomeOffsideTrap() {
    delete faceBall;
    delete go2pos;
    delete neuroKick;
    delete neuroIntercept;
    delete onetwoholdturn;

  }
  static bool init(char const * conf_file, int argc, char const* const* argv);
  OvercomeOffsideTrap();
  bool test_nb(Cmd & cmd);
  bool test_wb(Cmd & cmd);
  bool get_cmd(Cmd & cmd);
  //void reset_intention();
};


#endif
