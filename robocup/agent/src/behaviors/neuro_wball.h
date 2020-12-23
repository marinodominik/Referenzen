#ifndef _NEURO_WBALL_H_
#define _NEURO_WBALL_H_

#include "globaldef.h"

#include "abstract_mdp.h"
#include "planning.h"
#include "positioning.h"
#include "planning2.h"

#include "intention.h"
#include "skills/selfpass_bms.h"
#include "skills/dribble_straight_bms.h"

#include "mdp_info.h" //for STAMINA_STATE_*
#include "ws_memory.h"


class NeuroWball {

  Selfpass *selfpass;
  DribbleStraight *dribblestraight;

 private:

  //NetDescription P1net;

  bool is_better( const float V,const float Vbest, const AState & state, const AAction *actions, const int a, const int abest );

  void refine_laufpass( const AState state, AAction &action, const float Vaction );

  void print_action_data( const AState current_state, const AAction action, const int idx, const float V, const float display_value, const int log_level=2 );

  float do_evaluation( const AState current_state, const AAction action );

  float select_best_aaction( const AState current_state, const AAction *action_set, int action_set_size, AAction & best_action );

  void generate_safe_passes( const AState & state, AAction *actions, int & num_actions );

  void generate_risky_passes( const AState & state, AAction *actions, int & num_actions );

  void generate_passes( const AState & state, AAction *actions, int & num_actions, const int save_time );

  void generate_laufpasses( const AState & state, AAction *actions, int & num_actions, const int save_time );

  void generate_laufpasses2( const AState & state, AAction *actions, int & num_actions, const int save_time );

  void generate_penaltyarea_passes(AAction *actions, int &num_actions);

  int generate_action_set( const AState & state, AAction *actions );

  bool selfpass_dir_ok( const AState & state, const ANGLE dir );

  bool check_selfpass( const AState & state, const ANGLE targetdir, double &ballspeed, Vector &target, int &steps, Vector &op_pos, int &op_num );

  void generate_selfpasses( const AState & state, AAction *actions,int &num_actions);

  bool is_dribblestraight_possible( const AState & state );


  // new for OSAKA (everything that is needed for new version)
  Vector compute_potential_pos(const AState current_state,const AAction action);
  bool select_best_aaction2(const AState current_state,  AAction *action_set, int action_set_size, AAction &best_action);
  bool is_better2(const AState &state, const AAction *actions, const int idx, const int best_idx);
  void check_to_improve_action(AAction &action);

  void check_for_best(const AAction *actions, const int idx, int & best_safe,  int & best_risky );
  bool is_safe(const AAction action);

 public:

  double exploration;

  int exploration_mode;
  int evaluation_mode;

  NeuroWball();
  
  virtual ~NeuroWball();


  bool evaluate_passes_and_dribblings( AAction & best_aaction, AState & current_state );

  double adjust_speed( const Vector ballpos, const double dir, const double speed );
  double adjust_speed2( const Vector ballpos, const double dir, const double speed );


  // new for OSAKA (everything that is needed for new version):
  bool evaluate_passes( AAction & best_aaction, AState & current_state );

  void generate_safe_passes06(const AState &state, AAction *actions, 
			      int & num_actions);
  int generate_action_set06(const AState &state, AAction *actions);
  bool evaluate_passes06(AAction & best_aaction_P, AState &current_state);
  bool select_best_aaction06(const AState current_state,AAction *action_set,int action_set_size,AAction &best_action);
  void generate_laufpasses06( const AState & state, AAction *actions, int & num_actions, const int save_time );
  void generate_penaltyarea_passes06(AAction *actions, int &num_actions);


};

#endif // _NEURO_WBALL_H_
