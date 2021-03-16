#ifndef _NOBALL03_BMC_H_
#define _NOBALL03_BMC_H_

#include "../basics/Cmd.h"
#include "../policy/policy_tools.h"
#include "../policy/positioning.h"
#include "base_bm.h"
#include "skills/neuro_go2pos_bms.h"
#include "skills/intercept_ball_bms.h"
#include "skills/basic_cmd_bms.h"
#include "../policy/abstract_mdp.h"
#include "intention.h"
#include "skills/face_ball_bms.h"
#include "skills/neuro_intercept_bms.h"
#include "blackboard.h"
#include "noball03_attack_bmp.h"

class Noball03 : public BodyBehavior {
  static bool initialized;

  NeuroGo2Pos *go2pos;
  InterceptBall *intercept;
  BasicCmd *basic_cmd;
  NeuroIntercept *neurointercept;
  FaceBall *face_ball;
  Noball03_Attack *noball03_attack;

 protected:

 /* test move methods */
 
  bool test_go2ball_value(AAction &aaction);
  bool test_tackle(AAction &aaction);
  bool test_offside(AAction &aaction);
  bool test_look2ball(AAction &aaction);
  bool test_formations(AAction &aaction);
  bool test_receive_pass_val(AAction &aaction);
  bool test_ballpos_valid(Cmd &cmd, AAction &aaction);
  bool test_help_attack(AAction &aaction);
  bool test_help_blocking(AAction &aaction);
  bool test_go4pass(AAction &aaction);

  void set_aaction_go4pass(AAction &aaction, const Vector vballpos, const Vector vballvel);
  void set_aaction_go2ball(AAction &aaction);
  void set_aaction_turn_inertia(AAction &aaction, Angle ang);
  void set_aaction_tackle(AAction &aaction, double power);
  void set_aaction_face_ball(AAction &aaction);
  void set_aaction_face_ball_no_turn_neck(AAction &aaction);
  void set_aaction_goto(AAction &aaction, Vector target, double target_tolerance = 1.0, int consider_obstacles = 1, int use_old_go2pos = 0);
  void set_aaction_score(AAction &aaction, Vector target, double speed);
  void set_aaction_kicknrush(AAction &aaction, double speed, double dir);
  void set_aaction_panic_kick(AAction &aaction, Vector target);
  void set_aaction_backup(AAction &aaction, Vector target);
  bool aaction2cmd(AAction &aaction, Cmd &cmd);

  // ridi04: new methods:
  void check_goto_position(AAction &aaction);
  bool test_tackle_aggressive(AAction &aaction) ;
  DashPosition attack_positioning_for_middlefield_player( const DashPosition & pos ) ;
  DashPosition defense_positioning_for_middlefield_player( const DashPosition & pos ) ;



  DashPosition positioning_for_defence_player( const DashPosition & pos );
  DashPosition positioning_for_middlefield_player( const DashPosition & pos );
  DashPosition positioning_for_offence_player( const DashPosition & pos );

  /* stuff concerning the go2ball decision process */
  static const float GO2BALL_TOLERANCE;
  Go2Ball_Steps* go2ball_list;
  struct{
    int me;
    int my_goalie;
    int teammate;
    int teammate_number;
    Vector teammate_pos;
    bool ball_kickable_for_teammate;
    int opponent;
    int opponent_number;
    Vector opponent_pos;
    bool ball_kickable_for_opponent;
  } steps2go;
  float go2ball_tolerance_teammate;
  float go2ball_tolerance_opponent;
  void compute_steps2go();
  typedef struct {
    int player;
    int static_position;
    int number;
    Vector pos;
  } Cover_Position;
  
  long local_time;
  
  bool our_team_is_faster();
  bool i_am_fastest_teammate();
  bool opponent_free_kick_situation();
  bool my_free_kick_situation();
  bool intercepting_is_dangerous();
  int kick_in_by_midfielder();
  int kick_in_by_formation();
  bool kick_in(AAction &aaction, int player_to_go);
  Vector get_block_position();
  Vector get_block_position(int &use_long_go2pos);
  Vector get_block_position_DANIEL();
  Vector get_block_position_DANIEL(int &use_long_go2pos);
  bool block_ball_holder(AAction &aaction);
  bool intercept_ball(AAction &aaction);

  void set_attention(Cmd &cmd);
  void set_players_to_communicate(Cmd &cmd);

  /* stuff concerning the defense handling */
  double last_player_line;
  int my_role;
  bool test_cover_attacker(DashPosition & pos);
  bool test_save_stamina(DashPosition & pos);
  bool test_save_stamina_wm2001(DashPosition & pos);
  bool test_disable_moveup(DashPosition & pos);
  bool test_blocking_ball_holder(DashPosition & pos);
  bool surpress_intercept();
  bool do_surpress_intercept;
  bool do_tackle;
  int time_of_last_kickin;

  int teammate_pos_closest_to(Vector pos);
  void get_opponent_attackers(int *attacker_number, int max_attackers, int op_ball_holder);
  double get_opponent_danger(int opponent_number);
  Vector next_intercept_pos_NN();
  Vector ball_pos_abs_after_steps(int steps);
  void get_attacker_to_cover1(const DashPosition & pos, Cover_Position &cover_pos);
  
  Vector intersection_point(Vector p1, Vector steigung1, Vector p2, Vector steigung2);
  Vector point_on_line(Vector steigung, Vector line_point, double x);
  void get_positions_to_cover(Cover_Position *c_pos_Arr, int num_c_players, int num_defenders);
  int get_opponent_attackers(Cover_Position *c_pos_Arr, int max_attackers, int op_ball_holder);
  void get_opponent_attackers_in_midfield(int *attacker_number, int max_attackers, int op_ball_holder);

  bool attack_fastest_opponent_as_defender();

  Vector get_cover_position(const DashPosition & pos, int attacker_number);
  Vector get_cover_position_DANIEL(const DashPosition &pos, const Cover_Position &cover_pos);
  
  int inside_range_of_defender(int attacker_number);
  float nearest_unmatched_defender(int attacker_number, int *matched_defenders,  int max_opponent_attackers);

  bool test_move2ballside(DashPosition & pos);
  bool move2ballside;
  
  double pos_offence_eval(const Vector & pos, const Vector ball_pos, int size, Vector * opponents);

  /* stuff for resolving deadlock situations */
  int time_of_last_chance;
  int last_opponent;

  int time_of_last_receive_pass;

  int time_of_last_go2pos;
  Vector pos_of_last_go2pos;

  Vector next_i_pos;

  float teamformation_tolerance;
  int fine_positioning;

  struct{
    Intention intention;
  } my_blackboard;

  int cover_number;
  int cover_number_valid_at;
  int last_look_to_opponent;

  /* defenders do not move up, if stamina is insufficient to run back */ 
  double stamina_run_back_line;

  double get_last_player_line();


 public:

  static bool init(char const * conf_file, int argc, char const* const* argv) {
    if ( initialized )
      return true;
    initialized= true;
    
    bool b1 =     BasicCmd::init(conf_file, argc, argv);
    bool b2 =     NeuroGo2Pos::init(conf_file,argc,argv);
    bool b3 =     InterceptBall::init(conf_file,argc,argv);
    bool b4 =     NeuroIntercept::init(conf_file,argc,argv);
    bool b5 =     Noball03_Attack::init(conf_file,argc,argv);
    bool b6 =     FaceBall::init(conf_file,argc,argv);
    
    return b1 && b2 && b3 && b4 && b5 && b6;
  }
  void reset_intention();
  Noball03();
  virtual ~Noball03();

  bool get_cmd(Cmd &cmd);


  //virtual Main_Move* playon();
  //virtual Main_Move* reconsider_move(Main_Move *current_move);

};

#endif //_NOBALL03_BMC_H_

