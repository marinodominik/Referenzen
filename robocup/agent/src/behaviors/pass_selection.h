#ifndef _PASS_SELECTION_H_
#define _PASS_SELECTION_H_

#include "../basics/globaldef.h"

#include "intention.h"
#include "skills/selfpass_bms.h"
#include "skills/dribble_straight_bms.h"

#include "../policy/abstract_mdp.h"
#include "../policy/planning.h"
#include "../policy/policy_tools.h"
#include "../policy/positioning.h"
#include "../policy/planning2.h"

//#include "mdp_info.h" //for STAMINA_STATE_*
#include "ws_memory.h"


class PassSelection {

  Selfpass *selfpass;
  DribbleStraight *dribblestraight;

 private:

  //NetDescription P1net;

  void print_action_data(const Vector ballpos, const AAction action, const int color = 0);

  void check_to_improve_action(AAction &action);
  //  void check_for_best(const AAction *actions, const int idx, int & best_safe,  int & best_risky );
  bool is_safe(const AAction action);
  Vector compute_potential_pos(const AState current_state,const AAction action);


  void generate_safe_passes06(const AState &state, AAction *actions, 
			      int & num_actions);
  void generate_dream_passes06(const AState &state, AAction *actions, int & num_actions);
  int generate_action_set06(const AState &state, AAction *actions);
  bool select_best_aaction06(const AState current_state,AAction *action_set,int action_set_size,AAction &best_action);
  void generate_laufpasses06( const AState & state, AAction *actions, int & num_actions, const int save_time );
  void generate_penaltyarea_passes06(const AState & state, AAction *actions, int &num_actions);

  /* improvement of passes */

  void try_to_improve_pass(const AState &current_state, AAction &current_best);
  void improve_direct_pass(const AState &current_state, AAction &direct_pass);
  //  void check_to_replace_bestpass(const AState &state, AAction &best_pass,  AAction &candidate_pass);
  bool is_pass1_better (const AState &state,  AAction &candidate_pass, AAction &best_pass);
  bool is_pass_an_improvement(const AState &state, AAction &candidate_action, AAction &current_pass);

 public:

  PassSelection();
  virtual ~PassSelection();

  bool evaluate_passes06(AAction & best_aaction_P, AState &current_state);
  double adjust_speed2(const Vector ballpos, const double dir, const double speed);

  double compute_direct_pass_speed(const Vector ballpos,
				  const Vector teammatepos);


};

#endif // _PASS_SELECTION_H_
