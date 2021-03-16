#ifndef _ONEORTWO_STEP_KICK_BMS_H_
#define _ONEORTWO_STEP_KICK_BMS_H_

/* This behavior is a port of Move_1or2_Step_Kick into the behavior
   framework. get_cmd will try to kick in one step and otherwise return
   a cmd that will start a two-step kick. There are some functions
   that return information about the reachable velocities and the needed
   steps, so that you can prepare in your code for what this behavior will
   do.

   This behavior usually takes the current player position as reference point,
   but you can override this by calling set_state() prior to any other function.
   This makes it possible to "fake" the player's position during the current cycle.
   Use reset_state to re-read the WS information, or wait until the next cycle.

   Note that kick_to_pos_with_final_vel() is rather unprecise concerning the
   final velocity of the ball. I don't know how to calculate the needed starting
   vel precisely, so I have taken the formula from the original Neuro_Kick2 move
   (which was even more unprecise...) and tweaked it a bit, but it is still
   not perfect.

   Note also that this behavior, as opposed to the original move, has a working
   collision check - the original move ignored the player's vel...

   (w) 2002 Manuel Nickschas
*/


#include "../../basics/Cmd.h"
#include "one_step_kick_bms.h"
#include "base_bm.h"
#include "angle.h"
#include "Vector.h"
#include "tools.h"
#include "n++.h"
#include "macro_msg.h"
#include "valueparser.h"
#include "options.h"
#include "ws_info.h"
#include "log_macros.h"
#include "mystate.h"
#include "abstract_mdp.h"

class OneStepKick;

class OneOrTwoStepKickItrActions {
  // JTS define constants -> INITIALIZATION in oneortwo_step_kick_bms.c
  static const double kick_pwr_min;
  static const double kick_pwr_inc;
  static const double kick_pwr_max;
  static const double kick_ang_min;
  static const double kick_ang_inc;
  //  static const double kick_ang_inc = 2*PI/36.; // ridi: I think it should be as much! too much
  static const double kick_ang_max;
  static const double dash_pwr_min;
  static const double dash_pwr_inc;
  static const double dash_pwr_max;
  static const double turn_ang_min;
  static const double turn_ang_inc;
  //  static const double turn_ang_max = 2*PI-turn_ang_inc;
  static const double turn_ang_max;  // ridi: do not allow turns

  static const double kick_pwr_steps;
  static const double dash_pwr_steps;
  static const double turn_ang_steps;
  static const double kick_ang_steps;
  
  Cmd_Body action;

  double kick_pwr,dash_pwr;
  ANGLE kick_ang,turn_ang;
  int kick_pwr_done,kick_ang_done,dash_pwr_done,turn_ang_done;

 public:
    void reset();

  Cmd_Body* next();
};

class OneOrTwoStepKick: public BodyBehavior {
  static bool initialized;

  OneStepKick *onestepkick;

  OneOrTwoStepKickItrActions itr_actions;
  
  Cmd_Body result_cmd1,result_cmd2;
  double result_vel1,result_vel2;
  bool result_status;
  bool need_2_steps;
  long set_in_cycle;
  Vector target_pos;
  ANGLE target_dir;
  double target_vel;
  bool kick_to_pos;
  bool calc_done;

  MyState fake_state;
  long fake_state_time; 
  void get_ws_state(MyState &state);

  MyState get_cur_state();
  
  bool calculate(const MyState &state,double vel,const ANGLE &dir,const Vector &pos,bool to_pos,
		 Cmd_Body &res_cmd1,double &res_vel1,Cmd_Body &res_cmd2,double &res_vel2,
		 bool &need_2steps); 
  bool do_calc();

 public:

  /** This makes it possible to "fake" WS information.
      This must be called _BEFORE_ any of the kick functions, and is valid for 
      the current cycle only.
  */
  void set_state(const Vector &mypos,const Vector &myvel,const ANGLE &myang,
		 const Vector &ballpos,const Vector &ballvel, 
		 const Vector &op_pos = Vector(1000,1000), 
		 const ANGLE &op_bodydir = ANGLE(0),
		 const int op_bodydir_age = 1000);

  void set_state( const AState &state );
  
  /** Resets the current state to that found in WS.
      This must be called _BEFORE_ any of the kick functions.
  */
  void reset_state();
  
  void kick_in_dir_with_initial_vel(double vel,const ANGLE &dir);
  void kick_in_dir_with_max_vel(const ANGLE &dir);
  void kick_to_pos_with_initial_vel(double vel,const Vector &point);
  void kick_to_pos_with_final_vel(double vel,const Vector &point);
  void kick_to_pos_with_max_vel(const Vector &point);

  /** false is returned if we do not reach our desired vel within two cycles.
     Note that velocities are set to zero if the resulting pos is not ok,
     meaning that even if a cmd would reach the desired vel, we will ignore
     it if the resulting pos is not ok.
  */
  bool get_vel(double &vel_1step,double &vel_2step);
  bool get_cmd(Cmd &cmd_1step,Cmd &cmd_2step);
  bool get_vel(double &best_vel);    // get best possible vel (1 or 2 step)
  bool get_cmd(Cmd &best_cmd);      // get best possible cmd (1 or 2 step)

  // returns 0 if kick is not possible, 1 if kick in 1 step is possible, 2 if in 2 steps. probably modifies vel
  int is_kick_possible(double &speed,const ANGLE &dir);

  bool need_two_steps();

  bool  can_keep_ball_in_kickrange();
    
  static bool init(char const * conf_file, int argc, char const* const* argv);

  OneOrTwoStepKick();
  virtual ~OneOrTwoStepKick();
};


#endif
