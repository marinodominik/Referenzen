#ifndef _NOBALL03_ATTACK_BMC_H_
#define _NOBALL03_ATTACK_BMC_H_

#include "base_bm.h"
#include "skills/basic_cmd_bms.h"
#include "skills/neuro_go2pos_bms.h"
#include "skills/neuro_kick05_bms.h"
#include "skills/one_step_kick_bms.h"
#include "skills/oneortwo_step_kick_bms.h"
#include "skills/intercept_ball_bms.h"
#include "skills/neuro_intercept_bms.h"
#include "skills/face_ball_bms.h"

#include "neuro_wball.h"

#include "attack_move1_nb_bmp.h"


#include "neuro_positioning.h"

#include "../policy/policy_tools.h"
#include "../policy/positioning.h"
#include "../policy/abstract_mdp.h"

/** This is a test player, solely for Sputnick, so don't mess around here ;-) */

class Noball03_Attack: public BodyBehavior {
  static bool initialized;
  NeuroKick05 *neurokick;
  NeuroGo2Pos *go2pos;
  OneStepKick *onestepkick;
  OneOrTwoStepKick *oneortwo;
  InterceptBall *intercept;
  NeuroIntercept *neurointercept;
  FaceBall *face_ball;
  BasicCmd *basiccmd;

  Attack_Move1_Nb *attack_move1_nb;
  NeuroWball *neuro_wball;

  NeuroPositioning *neuro_positioning;

 protected:

  struct{
    int valid_at;
    Vector basepos;
    Vector relpos;
  } pinfo;
  
  void update_pinfo(XYRectangle2d rect, const Vector basepos, Vector *pos_arr, int size);
  void select_new_pinfo_pos(XYRectangle2d rect, const Vector basepos, Vector *pos_arr, int size);
  Vector select_good_position(XYRectangle2d rect, const Vector basepos, Vector *pos_arr, int size);
  bool is_position_covered(const Vector pos);
  bool choose_free_position(XYRectangle2d rect, Vector &res, Vector *pos_arr, int size);
  bool is_position_free(Vector position);

 /* test move methods */
  bool test_tackle(Cmd &cmd);
  bool test_go2ball_value(Cmd &cmd);
  bool test_offside(Cmd &cmd);
  bool test_gohome(Cmd &cmd);
  bool test_support_attack(Cmd &cmd);
  bool test_support_attack_for_midfielders(Cmd &cmd);
  bool test_receive_pass_val(Cmd &cmd);
  bool test_default(Cmd &cmd);
  bool test_neuro_positioning(Cmd &cmd);
  bool test_analytical_positioning(Cmd &cmd);
  bool test_ballpos_valid(Cmd &cmd);
  bool test_wait4clearance(Cmd &cmd);
  bool test_scoringarea_positioning(Cmd &cmd);

  bool is_mypos_ok(const Vector & targetpos);
  
  bool scoringarea_positioning_for_lwa(Cmd &cmd);
  bool scoringarea_positioning_for_ma(Cmd &cmd);
  bool scoringarea_positioning_for_rwa(Cmd &cmd);
  
  void determine_positioning_constraints(XYRectangle2d *constraints_P, Vector *home_positions_P);

  bool do_waitandsee(Cmd &cmd);
  
  bool go2pos_economical(Cmd &cmd, const Vector target);
  bool shall_I_go(const Vector target);
  
  
  DashPosition positioning_for_offence_player( const DashPosition & pos );
  DashPosition positioning_for_middlefield_player( const DashPosition & pos );
  DashPosition attack_positioning_for_middlefield_player( const DashPosition & pos ) ;

  /* stuff concerning the go2ball decision process */
  static const float GO2BALL_TOLERANCE;
  Go2Ball_Steps* go2ball_list;
  struct{
    int me;
    int my_goalie;
    int teammate;
    int teammate_number;
    int opponent;
    int opponent_number;
    Vector teammate_pos;
    bool ball_kickable_for_teammate;
    Vector opponent_pos;
    bool ball_kickable_for_opponent;
  } steps2go;
  static int num_pj_attackers;
  float y_homepos_tolerance;
  float attacker_y_tolerance;
  float attacker_gowing_tolerance;
  float attacker_x_backward_tolerance, midfielder_x_forward_tolerance, midfielder_x_backward_tolerance;
  float midfielder_behind_offsideline;
  float go2ball_tolerance_teammate;
  float go2ball_tolerance_opponent;
  float min_dist2teammate;
  void compute_steps2go();
  bool do_receive_pass;
  int time_of_last_receive_pass;
  bool do_stay_active;
  bool go_home_is_possible;
  bool my_free_kick_situation();
  int kick_in_by_formation ();
  int kick_in_by_midfielder();

  double critical_offside_line;
  Vector my_homepos;
  int my_role;
  bool should_care4clearance;

  Vector target;
  int last_neuro_positioning;

  bool i_am_fastest_teammate();
  bool intercept_ball(Cmd &cmd);
  void get_permutation(const int m, const int no_sets, const int n[], int idx[]);
  float evaluate(const AState& state,const AAction jointaction[]);
  void get_jointaction(const int idx[],AAction jointaction[]);
  int determine_all_jointactions(const AState& state);
  // is the new player position a valid position
  void code_pos(const AState& state, const Vector& orig, Vector & feat);
  void code_ball(const AState& state, const Vector& orig, Vector & feat);
  void code_ball_vel(const AState& state, const Vector& orig, Vector & feat);
  bool position_check(const Vector targetpos);
  bool check_goto_absolute(const AState& state,int player, Vector targetpos, 
			   AAction &result);
  bool check_goto(const AState& state,int player, Vector deltapos, AAction &result);
  bool check_gohome(const AState& state,int player, AAction &result);
  bool is_relevant(const AState& state,int player);
  bool is_stuck();
  bool am_I_passcandidate();
  bool playon(Cmd &cmd);
  bool aaction2cmd(AAction action, Cmd &cmd);
  bool opening_seq(const Vector ballpos, const Vector ballvel, const Vector ipos, Cmd &cmd);
  bool am_I_neuroattacker();
  bool shall_I_do_neuropositioning();
  void check_correct_homepos(Vector &homepos);
  bool go2pos_withcare(Cmd &cmd, const Vector target);
  bool Iam_a_neuroattacker;

 public:
  bool test_go4pass(Cmd &cmd);

  static bool init(char const * conf_file, int argc, char const* const* argv) {
    if ( initialized )
      return true;
    initialized= true;
    

	  bool b1 =  Attack_Move1_Nb::init(conf_file,argc,argv) ;
	  bool b2 =  NeuroKick05::init(conf_file,argc,argv) ;
	  bool b3 =  NeuroGo2Pos::init(conf_file,argc,argv) ;
	  bool b4 =  OneStepKick::init(conf_file,argc,argv) ;
	  bool b5 =  OneOrTwoStepKick::init(conf_file,argc,argv) ;
	  bool b6 =  InterceptBall::init(conf_file,argc,argv) ;
	  bool b7 =  NeuroIntercept::init(conf_file,argc,argv) ;
	  bool b8 =  BasicCmd::init(conf_file,argc,argv) ;
	  bool b9 =  FaceBall::init(conf_file,argc,argv);

    return b1 && b2 && b3 && b4 && b5 && b6 && b7 && b8 && b9;
  }
  Noball03_Attack();
  virtual ~Noball03_Attack();
  bool get_cmd(Cmd & cmd);
  void reset_intention();
  static bool am_I_attacker();
};


#endif
