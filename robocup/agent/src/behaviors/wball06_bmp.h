#ifndef _WBALL06_BMC_H_
#define _WBALL06_BMC_H_

#include "base_bm.h"
#include "skills/basic_cmd_bms.h"
#include "skills/neuro_kick05_bms.h"
#include "skills/neuro_kick_wrapper_bms.h"
#include "skills/selfpass2_bms.h"
#include "skills/ballzauber_bms.h"
#include "skills/dribble_straight_bms.h"
#include "skills/dribble_between06.h"
#include "skills/dribble_between.h"
#include "skills/one_step_kick_bms.h"
#include "skills/onetwo_holdturn_bms.h"
#include "skills/oneortwo_step_kick_bms.h"
#include "skills/score_bms.h"
#include "skills/score05_sequence_bms.h"
#include "skills/one_step_score_bms.h"
#include "../policy/abstract_mdp.h"
#include "intention.h"
#include "skills/tingletangle.h"
#include "skills/neuro_dribble_2017.h"
//#include "neuro_wball.h"
#include "pass_selection.h"

class Wball06: public BodyBehavior
{
  friend class StandardSituation;
  friend class StandardSituation07;
  friend class MyGoalKick2016;
  NeuroKick05 *neurokick;
  NeuroKickWrapper *neurokickwrapper;
  //DribbleBetween06 *dribble_between;
  DribbleBetween *dribble_between;
  Selfpass2 *selfpass2;
  Ballzauber *ballzauber;
  BasicCmd *basiccmd;
  OneStepKick *onestepkick;
  OneOrTwoStepKick *oneortwo;
  OneTwoHoldTurn *onetwoholdturn;
  Score *score;
  Score05_Sequence *score05_sequence;
  PassSelection *pass_selection;
  TingleTangle *tingletangle;
  OneStepScore* onestepscore;
  NeuroDribble2017* neurodribble2017;

 private:
  
  struct{
    bool is_dribble_ok;
    bool is_dribble_safe;
    NeckRequest dribble_neckreq;
    bool is_holdturn_safe;
    bool is_tt_safe;
    double tt_dir;
  } status;
  
  typedef struct{
    long communicated_at;  
    int type;
    double speed;
    double passdir;
  } CommunicationT;
  
  struct{
    Intention intention;
    CommunicationT communication;
  } memory;


  int cycles_in_waitandsee;
  long last_waitandsee_at;
  long last_heavy_attack_at;

  
 public:
  static bool init(char const * conf_file, int argc, char const* const* argv) {
    return (
      NeuroKick05::init(conf_file,argc,argv) &&
      NeuroKickWrapper::init(conf_file,argc,argv) &&
	    DribbleStraight::init(conf_file,argc,argv) &&
	    Selfpass2::init(conf_file,argc,argv) &&
	    Ballzauber::init(conf_file,argc,argv) &&
	    BasicCmd::init(conf_file,argc,argv) &&
	    OneStepKick::init(conf_file,argc,argv) &&
	    OneOrTwoStepKick::init(conf_file,argc,argv) &&
	    OneTwoHoldTurn::init(conf_file,argc,argv) &&
	    OneStepScore::init(conf_file,argc,argv) &&
	    TingleTangle::init(conf_file,argc,argv) &&
	    NeuroDribble2017::init(conf_file,argc,argv) 
           );
  }
  Wball06();
  virtual ~Wball06();
  
  void foresee(Cmd &cmd);

  bool get_intention(Intention &intention, Intention &pass_option, Intention &selfpass_option );
  bool get_simple_intention(Intention &intention, Intention &pass_option, Intention &selfpass_option );
  bool get_cmd(Cmd & cmd);
  BodyBehavior * getScoreBehavior();
  BodyBehavior * getOneStepScoreBehavior();
  void reset_intention();
  bool check_previous_intention(Intention prev_intention, Intention  &new_intention, 
				Intention & pass_option, Intention & selfpass_option);

 private:
  long tt_last_active_at;
  bool holdturn_intention2cmd(Intention &intention, Cmd &cmd);
  bool intention2cmd(Intention &intention, Cmd &cmd); 

  /* determining options */
  void determine_pass_option(Intention &pass_option, const Vector mypos, const Vector myvel,const ANGLE myang, 
			 const Vector ballpos,const Vector ballvel);
  void determine_best_pass(Intention &pass_option, AState &state);
  void determine_best_selfpass(Intention &selfpass_option,const Vector mypos, const Vector myvel, const ANGLE myang,
			       const double mystamina,
			       const Vector ballpos, const Vector ballvel);

  void determine_advance_selfpass(Intention &selfpass_option,const Vector mypos, const Vector myvel, const ANGLE myang,
			       const double mystamina,
			       const Vector ballpos, const Vector ballvel);
  
  void determine_best_selfpass_close2goalline(Intention &selfpass_option, const Vector mypos, const Vector myvel, const ANGLE myang,
					   const double mystamina,
					   const Vector ballpos, const Vector ballvel); 
  void determine_best_selfpass_in_goalarea(Intention &selfpass_option, const Vector mypos, const Vector myvel, const ANGLE myang,
					   const double mystamina,
					   const Vector ballpos, const Vector ballvel); 

  bool prefered_direction_is_free(const ANGLE dir, const double length, const double width_factor = 1.0, const bool consider_goalie = false);
  bool I_am_close2goalline();
  bool selfpass_dir_ok(const ANGLE dir);
  bool selfpass_area_is_free(const ANGLE dir);
  bool selfpass_goalarea_is_free(const ANGLE dir, const Vector ballpos);
  bool selfpass_close2goalline_is_free(const ANGLE dir, const Vector ballpos);
  bool advance_selfpass_area_is_free(const ANGLE dir, const Vector ballpos);

  /* test for intention */
  bool test_selfpasses(Intention &intention, Intention & selfpass_option);
  bool test_advance_selfpasses(Intention &intention, Intention & selfpass_option);
  bool test_save_turn_in_opp_pen_area(Intention &intention, Intention & selfpass_option);
  bool test_priority_pass(Intention &intention, Intention & pass_option, Intention & selfpass_option);
  bool test_in_trouble(Intention &intention);
  bool test_two_teammates_control_ball(Intention &intention);
  bool test_default(Intention &intention);
  bool test_dribbling(Intention& intention);
  bool test_holdturn(Intention &intention);
  bool test_tingletangle(Intention& intention);
  bool test_prefer_holdturn_over_dribbling(Intention &intention);
  bool test_pass(Intention &intention, Intention & pass_option, Intention & selfpass_option);
  bool test_pass_under_attack(Intention &intention, Intention &pass_option, Intention &selfpass_option);
  bool test_pass_when_playing_against_aggressive_opponents_08
       (Intention &intention, Intention &pass_option, Intention &selfpass_option);
  bool test_1vs1(Intention &intention, Intention & pass_option, Intention & selfpass_option);
  bool test_dream_selfpass(Intention &intention, const Vector mypos, const Vector ballpos, const Vector ballvel);
  bool test_tingletangle_in_scoring_area(Intention &intention);

  void check_if_wait_then_pass(Intention &pass_intention);

  bool can_tingletangle2dir(double dir);


  /* auxillary */
  bool check2choose_tt_dir(double &dir);

  bool is_dream_selfpass_to_target_possible(Intention &intention, 
                                            Vector targetpos, 
                                            const Vector mypos, 
                                                 const Vector ballpos, 
                                            const Vector ballvel,
                                            bool checkPrevIntentionMode=false,
                                            const double kickSpeed=0.0);
  bool I_am_in_right_corner();
  bool I_am_in_left_corner();
  bool I_am_in_goalarea();

  bool is_goalarea_free();

  void foresee_modify_cmd(Cmd &cmd, const double targetdir);
  bool is_selfpass_lookfirst_possible(Intention &intention, Intention & selfpass_option);
  bool is_selfpass_lookfirst_needed(Intention & selfpass_option);
  void display_intention(Intention & intention);
  void aaction2intention(const AAction &aaction, Intention &intention);
  void get_onestepkick_params(double &speed, double &dir);
  bool get_best_panic_selfpass(const double testdir[],const int num_dirs,double &speed, double &dir);
  int I_am_heavily_attacked_since();
  bool am_I_attacked(const double factor = 2.0, const double backExtraFactor = -1.0);
  bool is_my_passway_in_danger(Intention & pass_intention);
  int howmany_kicks2pass(Intention & pass_intention);
  int is_dribblebetween_possible();
  //* if I can play a pass with less kicks that is nearly as fast, lower pass speed a bit */
  void check_adapt_pass_speed(Intention &pass_intention);
  double get_prefered_tingletangle_dir_close2goalline();
  double get_prefered_tingletangle_dir();
  double get_prefered_tingletangle_dir_infield();
  bool targetpos_keeps_dist2goalie(const double dir);

  /* communicate and neck */
  void set_neck(Intention &intention, Intention &pass_option, Intention & selfpass_option);
  void set_neck_selfpass(Intention & selfpass_intention);
  void set_communication(Intention &intention, Intention &pass_intention, const Vector mypos, Cmd &cmd);
  void set_neck_default(Intention & intention, Intention &pass_option, Intention & selfpass_option);
  void set_neck_pass(Intention & pass_intention);
  void set_neck_dribble();
  bool is_pass_announced(Intention &pass_intention);
};


#endif
