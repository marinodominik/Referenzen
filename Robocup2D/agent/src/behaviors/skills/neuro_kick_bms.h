#ifndef _NEURO_KICK_BMS_H_
#define _NEURO_KICK_BMS_H_

/* This behavior is derived from the (now deprecated) move neuro_kick2.
   Only its operation mode 1 (using the BS01 nets) is implemented.
   Operation mode 0 (using BS2k nets) has not been converted, since this
   mode has not been used for quite a while now.
   Operation mode 2 is not possible either, meaning that there are no learning
   functions in this behavior.

   NeuroKick usually takes the current player position as reference point,
   but you can override this by calling set_state() prior to any other function.
   This makes it possible to "fake" the player's position during the current cycle.
   Use reset_state to re-read the WS information, or wait until the next cycle.

   Note that kick_to_pos_with_final_vel() is rather unprecise concerning the
   final velocity of the ball. I don't know how to calculate the needed starting
   vel precisely, so I have taken the formula from the original Neuro_Kick2 move
   (which was even more unprecise...) and tweaked it a bit, but it is still
   not perfect.

   Some bugs of the original Neuro_Kick2 move have been found and fixed:

   * Collision check did not work correctly (player's vel was ignored)
   * kick_to_pos used the player as reference point instead of the ball pos,
     resulting in rather unprecise kicks.
   
   The NeuroKick behavior can make use of the OneOrTwoStepKick (default).
   
   (w) 2002 by Manuel Nickschas
*/

#include "../../basics/Cmd.h"
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
#include "oneortwo_step_kick_bms.h"
#include "mystate.h"


class NeuroKickItrActions {
  /* set params */

  static const double kick_pwr_min;
  static const double kick_pwr_inc;
  static const double kick_pwr_steps;
  static const double kick_fine_pwr_inc;
  static const double kick_fine_pwr_steps;
  static const double kick_ang_min;
  static const double kick_ang_inc;
  static const double kick_ang_steps;
  static const double kick_fine_ang_inc;
  static const double kick_fine_ang_steps;

  double pwr_min,pwr_inc,pwr_steps,ang_steps;
  ANGLE ang_min,ang_inc;
  
  Cmd_Body action;
  int ang_done,pwr_done;
  double pwr;ANGLE ang;
 public:
  void reset(bool finetune=false,double orig_pwr=0,ANGLE orig_ang=ANGLE(0));
  
  Cmd_Body *next();
};

class NeuroKick: public BodyBehavior {
  static bool initialized;

#if 0
  struct State {
    Vector my_vel;
    Vector my_pos;
    ANGLE  my_ang;
    Vector ball_pos;
    Vector ball_vel;
  };
#endif
  
  static const double kick_finetune;
  static const double kick_finetune_power;
  static const double turn_finetune;
  
  static const double tolerance_velocity;
  static const double tolerance_direction;
  static const double min_ball_dist;
  static const double kickable_tolerance;

  static bool use_12step_def;
  static bool do_finetuning;
  
  static Net *nets[2];
  Net *net; // chosen net
    
  NeuroKickItrActions itr_actions;

  OneOrTwoStepKick *twostepkick;

  long   init_in_cycle;
  double target_vel;
  ANGLE  target_dir;
  Vector target_pos;
  bool   do_target_tracking;
  bool   use_12step;

  MyState fake_state;
  long fake_state_time;
  void get_ws_state(MyState &);
  MyState get_cur_state();
  
  void get_features(const MyState &state, ANGLE dir, double vel,float *net_in);
  Net *choose_net();
  bool is_success(const MyState&);
  bool is_failure(const MyState&);

  double evaluate(MyState const &state);
  bool decide(Cmd &cmd);

 public:
  NeuroKick();
  virtual ~NeuroKick();
  static bool init(char const *conf_file, int argc, char const* const* argv);

  /** This makes it possible to "fake" WS information.
      This must be called _BEFORE_ any of the kick functions, and is valid for 
      the current cycle only.
  */
  void set_state(const Vector &mypos,const Vector &myvel,const ANGLE &myang,
		 const Vector &ballpos,const Vector &ballvel);
  
  /** Resets the current state to that found in WS.
      This must be called _BEFORE_ any of the kick functions.
  */
  void reset_state();
  
  void kick_to_pos_with_initial_vel(double vel,const Vector &pos,bool use_12step = use_12step_def);
  void kick_to_pos_with_final_vel(double vel, const Vector &pos,bool use_12step = use_12step_def);
  void kick_to_pos_with_max_vel(const Vector &pos,bool use_12step = use_12step_def); 
  void kick_in_dir_with_initial_vel(double vel, const ANGLE &dir,bool use_12step = use_12step_def);
  void kick_in_dir_with_max_vel(const ANGLE &dir,bool use_12step = use_12step_def);
  
  bool get_cmd(Cmd & cmd);
};
    
#endif
