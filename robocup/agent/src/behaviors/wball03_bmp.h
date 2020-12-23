#ifndef _WBALL03_BMC_H_
#define _WBALL03_BMC_H_

#include "base_bm.h"
#include "skills/basic_cmd_bms.h"
#include "skills/neuro_kick05_bms.h"
#include "skills/selfpass_bms.h"
#include "skills/selfpass2_bms.h"
#include "skills/dribble_straight_bms.h"
#include "skills/dribble_between.h"
#include "skills/one_step_kick_bms.h"
#include "skills/onetwo_holdturn_bms.h"
#include "skills/oneortwo_step_kick_bms.h"
#include "skills/score_bms.h"
#include "skills/score05_sequence_bms.h"
#include "../policy/abstract_mdp.h"
#include "intention.h"
#include "neuro_wball.h"

class Wball03: public BodyBehavior {
  NeuroKick05 *neurokick;
  DribbleStraight *dribblestraight;
  DribbleBetween *dribble_between;
  Selfpass *selfpass;
  Selfpass2 *selfpass2;
  BasicCmd *basiccmd;
  OneStepKick *onestepkick;
  OneOrTwoStepKick *oneortwo;
  OneTwoHoldTurn *onetwoholdturn;
  Score *score;
  Score05_Sequence *score05_sequence;
  NeuroWball *neuro_wball;

 private:
  bool in_penalty_mode;
  bool reconsider_goalshot;
  int wait_and_see_patience;
  double wait_and_see_clearanceline;
  int at_ball_patience;
  long last_at_ball;
  int at_ball_for_cycles;
  double flank_param;
  int lastTimeLookedForGoalie;
  int lastTimeLookedForGoal;
  float intendedLookDirection;
  int lasttime_in_waitandsee;
  int cyclesI_looked2goal;
  int cycles_in_waitandsee;
  long last_waitandsee_at;
  long last_heavy_attack_at;
  int evaluation_mode;
  int check_action_mode;
  int action_set_type;
  int exploration_mode;
  float exploration;
  float success_threshold;
  float dribble_success_threshold;
	bool is_dribble_ok;
	bool is_dribble_straight_ok;
	bool is_dribble_between_ok;
	bool is_dribble_between_insecure;
  int my_role;
  PPlayer closest_opponent; // warning: might be 0!

  bool is_selfpass_possible();
  void precompute_best_selfpass();  // side effect: fill out scheduled selfpass formular
  void precompute_best_selfpass(const Vector mypos, const Vector myvel, const ANGLE myang,
				const double mystamina,
				const Vector ballpos, const Vector ballvel);
  void precompute_best_selfpass_in_goalarea(const Vector mypos, const Vector myvel, const ANGLE myang,
				const double mystamina,
				const Vector ballpos, const Vector ballvel);

  struct{
    ANGLE targetdir;
    long valid_at;
    Vector targetpos;
    double kickspeed;
    int steps2go;
    double evaluation;
    Vector attackerpos;
    int attacker_num;
  } scheduled_selfpass;

  struct{
    Intention pass_or_dribble_intention;
    Intention intention;
    NeckRequest neckreq;
  } my_blackboard;


  bool get_turn_and_dash(Cmd &cmd);

  bool selfpass_potential2score(Vector pos);


  bool test_dribbling(Intention &intention);
  bool test_solo(Intention &intention);
  bool test_selfpasses(Intention &intention);
  bool test_selfpasses_in_scoring_area(Intention &intention);
  bool test_advance_in_scoring_area(Intention &intention);
  bool test_priority_pass2(Intention &intention);
  bool test_default(Intention &intention);
  bool test_holdturn(Intention &intention);
  bool test_holdturn2(Intention &intention);
  bool test_kicknrush(Intention &intention);
  bool test_two_teammates_control_ball(Intention &intention);
  bool test_in_trouble(Intention &intention);
  bool test_opening_seq(Intention &intention);
  bool test_dribble_straight(Intention &intention);

  bool get_best_panic_selfpass(const double testdir[],const int num_dirs,double &speed, double &dir);


  bool I_can_advance_behind_offside_line();
  bool selfpass_dir_ok(const ANGLE dir );
  bool selfpass_dir_ok2(const ANGLE dir );
  bool selfpass_area_is_free(const ANGLE dir);

  bool aggressive_selfpass_dir_ok(const ANGLE dir);

  bool check_selfpass(const ANGLE targetdir, double &ballspeed, Vector &target, int &steps,
		      Vector &op_pos, int &op_num);
  void set_neck_selfpass(const ANGLE targetdir, const Vector &op_pos);
  void set_neck_selfpass2();

  bool I_am_in_selfpass_goalarea();


  bool is_dribblestraight_possible();
  bool is_dribblestraight_possible2();
  int is_dribblebetween_possible();


  int I_am_heavily_attacked_since();

  bool test_pass_or_dribble(Intention &intention);
  bool check_previous_intention(Intention prev_intention, Intention  &new_intention);

  void check_write2blackboard();

  bool get_pass_or_dribble_intention(Intention &intention);
  bool get_pass_or_dribble_intention(Intention &intention, const Vector newmypos, const Vector newmyvel,
				     const Vector newballpos,const Vector newballvel);
  bool get_pass_or_dribble_intention(Intention &intention, AState &state);
  
  void aaction2intention(const AAction& aaction, Intention &intention);


  // auxillary functions for offensive_move:
  void get_onestepkick_params(double &speed, double &dir);
  void get_kickrush_params(double &speed, double &dir);
  void get_kickrush_params(double &speed, double &dir, Vector &ipos, int &advantage,
			   int & closest_teammate);
  void get_opening_pass_params(double &speed, double &dir, Vector &ipos, int &advantage,
			       int & closest_teammate);

  void get_clearance_params(double &speed, double &dir);
  bool check_kicknrush(double &speed, double &dir, bool &safe, Vector &resulting_pos);


  bool get_opening_seq_cmd( const float  speed, const Vector target,Cmd &cmd);
  double adjust_speed(const Vector ballpos, const double dir, const double speed);
  bool is_pass_a_killer();

  bool is_my_passway_in_danger();
  bool am_I_attacked();
  bool am_I_attacked2();
  int howmany_kicks2pass();
  bool can_advance_behind_offsideline(const Vector pos);
  bool I_am_close_to_offsideline(const double howclose=5.0);


 protected:
 public:
  void foresee(const Vector newmypos, const Vector newmyvel,const ANGLE newmyang,
	       const Vector newballpos,const Vector newballvel, ANGLE & targetdir);
  bool get_intention(Intention &intention);
  static bool init(char const * conf_file, int argc, char const* const* argv) {
    return (
	    NeuroKick05::init(conf_file,argc,argv) &&
	    DribbleStraight::init(conf_file,argc,argv) &&
	    Selfpass::init(conf_file,argc,argv) &&
	    Selfpass2::init(conf_file,argc,argv) &&
	    BasicCmd::init(conf_file,argc,argv) &&
	    OneStepKick::init(conf_file,argc,argv) &&
	    OneOrTwoStepKick::init(conf_file,argc,argv) &&
	    OneTwoHoldTurn::init(conf_file,argc,argv)
	    );
  }
  Wball03();
  virtual ~Wball03();
  bool get_cmd(Cmd & cmd);
  void reset_intention();
  Intention oot_intention; //hauke
  bool intention2cmd(Intention &intention, Cmd &cmd);  //set public 04/06 HS
};


#endif
