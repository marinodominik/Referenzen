#ifndef _PLANNING_H
#define _PLANNING_H

#include <stdlib.h>
#include <stdio.h>
#include <sys/time.h>

#include "PlayerSet.h"
#include "globaldef.h"
#include "Vector.h"
#include "abstract_mdp.h"
#include "n++.h"
#include "oneortwo_step_kick_bms.h"
#include "policy_tools.h"
#include "mdp_info.h"
#include "tools.h"
#include "log_macros.h"
#include "options.h"
#include "valueparser.h"
#include "positioning.h"
#include "intercept.h"
#include "sort.h"

#define MIN_PASS_DISTANCE 5.0 // this is the minimum distance between to players for passing


#define MAX_AACTIONS 4000
#define SCORE_REWARD 1000
#define PASS_ARRIVAL_SPEED 1.1
#define PASS_SAFETY_MARGIN 2.0
#define SELFPASS_SAFETY_MARGIN 4.0
#define DRIBBLE_SAFETY_MARGIN 2.0
#define MAX_PASS_SPEED 2.5
#define MIN_POSITION_NET -20.

class OneOrTwoStepKick;

class Netdescription
{
public:
  Net net;
  bool loaded;
  int opponents_num;
  int teammates_num;
  int sort_type;

  Netdescription();
};


class Planning {
 private:
  static int risky_mytime2react;
  static int risky_optime2react;

  static int    laufpass_min_advantage;
  static bool   dribblings_allowed;
  static double pass_opponent_bonusstep;
  static double pass_goalie_bonusstep;
  static double pass_ownhalf_factor;
  static double pass_attack_factor;
  static double selfpass_myhandicap;
  static int    valuefunction;
  static double dribble_velocity_loss; // slower than without ball
  static double dribble_opponent_bonusstep;
  static double dribble_goalie_bonusstep;
  static double goalie_pass_min_advantage;
  static double goalie_maxspeed_percentage;
  static double goalshot_goalie_bonusstep;  // goalie is 1 step closer to ball
  static double extremeshot_goalie_maxspeed_percentage;
  static double pass_arrival_speed;
  static double pass_inrun_arrival_speed;
  static double consider_age;
  static double selfpass_consider_age;
  static double selfpass_ok_zone;
  static double consider_freedom;
  static double multikick_opponent_speed;
  static bool   pass_receiver_array[11];
  /**
  static void code_pos(const AState& state, const Vector& orig, Vector & feat, 
		       const int type =0);
  static void code_ball(const AState& state, const Vector& orig, Vector & feat, 
			const int type=0);
  */
  static Netdescription Qnet, P1net, P2net, Jnn1, Jnn2, J1vs1;
  static float Jnn1_active_thresh,J1vs1_active_thresh, Jnn2_active_thresh, Jnn2_deactive_thresh;
  static int activeJnn;
  static OneOrTwoStepKick *twostepkick;

 public:
  static void init_params();
  /* Planning stuff */
  /* checks whether opponent is in kickrange or can be if op. moves in the next step
     default value is that he moves with .8 * max_speed; if 0, then only op. in kickrange
     are considered
  */
  static bool is_kick_possible(float &speed, Angle dir);

  static bool is_kick_possible(float &speed, Vector target);
  
  static bool is_kick_possible( const AState & state, float &speed, Vector target );
  
  static bool is_kick_possible( const AState & state, float &speed, Angle dir );
  
  static float evaluate(const AState &state,const int last_action_type = -1);

  static float evaluate_action(const AState &state, AAction& action, int steps2go,
			       AAction solution_path[]);

  static bool check_action_solo(const AState &state,AAction &candidate_action);
  static bool check_action_laufpass(const AState &state,AAction &candidate_action, 
				    float speed, const float dir,const bool risky_pass=false);
  static bool check_action_laufpass2(AAction &candidate_action, 
				    float speed, const float dir);

  static bool is_laufpass_successful2(const Vector ballpos, const float speed, const float dir, Vector & interceptpos,
				      int & number, int & advantage, Vector & playerpos);

  static bool is_laufpass_successful2(const Vector ballpos, float speed, const float dir);

  static bool is_penaltyareapass_successful(const Vector ballpos,float speed, const float dir, int & advantage, int &number,
				       Vector & playerpos);

  static bool is_penaltyareapass_successful(const Vector ballpos,float speed, const float dir);

  static bool check_action_penaltyareapass(AAction &candidate_action, float speed, const float dir);

  static bool check_action_pass(const AState &state,
				AAction &candidate_action, 
				int recv_idx, Vector rel_target, 
				double min_pass_dist=MIN_PASS_DISTANCE);


  static float evaluate_state(const AState& state);
  static float evaluate_byQnn(const AState& state, const AAction &action);
  static float evaluate_byP1nn(const AState& state, const AAction &action);
  static float evaluate_byP2nn(const AState& state, const AAction &action);
  static float evaluate_byJnn(const AState &state,const int last_action_type = -1);
  static float evaluate_byJnn_1vs1(const AState &state);
  static float evaluate_state1(const AState& state);
  static int generate_pactions(const AState &state, AAction *actions);

  /**  returns a value between 0 and V_ballmax/V_playermax, if its bigger than 1, than
   goalshot is considered safe (without considering goalies catch capabilities */
  static float goalshot_chance_vs_optigoalie(const Vector ballpos);
    
  inline static void mark_pass_receiver(int pass_receiver){pass_receiver_array[pass_receiver] = true;};
  static void unmark_all_pass_receiver();
  inline static bool is_pass_receiver_marked(int pass_receiver){return(pass_receiver_array[pass_receiver]);};

  static float compute_pass_speed(Vector ballpos,Vector receiverpos,Vector targetpos,
				  const double maxspeed = MAX_PASS_SPEED);
  static float compute_pass_speed_with_arrival_vel(Vector ballpos,Vector receiverpos,Vector targetpos,
				  const double maxspeed, double & arrival_vel );

  /** returns true if our team gets the ball. advantage is the number of steps
      we are faster than our opponent
  */
  static bool is_laufpass_successful(const Vector ballpos,
				     float & speed, const float dir,
				     Vector &interceptpos, int &advantage, 
				     int &number,Vector &playerpos,const bool risky_pass=false);

  static bool is_laufpass_successful(const AState & state, const Vector ballpos,
				     float & speed, const float dir,
				     Vector &interceptpos, int &advantage, 
				     int &number,Vector &playerpos,const bool risky_pass=false);

  static bool is_laufpass_successful(const Vector ballpos,
				     float & speed, const float dir,const bool risky_pass=false);

  static bool is_pass_successful(const AState & state, const Vector ballpos,
				 float & speed, const float dir,
				 const Vector playerpos,
				 Vector &interceptpos, int &advantage);

  static bool is_pass_successful_hetero(const AState & state, const Vector ballpos,
					float & speed, const float dir,
					const PPlayer player,
					Vector &interceptpos, 
					int &advantage);

  static bool is_pass_successful(const Vector ballpos,
				 float & speed, const float dir,
				 const Vector playerpos,
				 Vector &interceptpos, int &advantage);

  static bool is_pass_successful_hetero(const Vector ballpos,
					float & speed, const float dir,
					const PPlayer player,
					Vector &interceptpos, 
					int &advantage);

  /** returns -1 if opponent is faster or steps_diff if receiver is faster
      in either case, intercept_pos contains the position where the ball is caught
      uses getball chance */
  static float pass_success_level(const Vector ballpos, Vector receiverpos, 
				  const double speed, const double dir,
				  Vector &intercept_pos);

  /** returns -1 if opponent is faster or steps_diff if receiver is faster
      in either case, intercept_pos contains the position where the ball is caught
      uses getball chance with option 'do not consider close players' */
  static float receive_pass_success_level(const Vector ballpos, Vector receiverpos, 
					  const double speed, const double dir,
					  Vector &intercept_pos);

  /** returns -1 if opponent is faster or steps_diff if receiver is faster
      in either case, intercept_pos contains the position where the ball is caught */
  static float getball_chance(const Vector ballpos, Vector receiverpos, 
			      const double speed, const double dir,
			      Vector &intercept_pos,
			      const double receiverbonus=0,
			      bool consider_close_players = true,
			      double consider_age_factor = -1,
			      bool selfpass = false);
  /** checks if the player at position playerpos can receive a pass from the current ballpos */
  static bool is_player_a_passcandidate(Vector playerpos);
  static void generate_input(const AState& state, const Netdescription &netinfo);
  static float Qnn(const AState& state, const bool testoutput = false);
  static float P1nn(const AState& state, const bool testoutput = false);
  static float P2nn(const AState& state, const bool testoutput = false);

  static void code_pos(const AState& state, const Vector& orig, Vector & feat,
		const int net_sort_type=0, const float min_x = MIN_POSITION_NET,
		const bool use_x_symmetry=false);

  static void code_ball(const AState& state, const Vector& orig, Vector & feat,
		 const int net_sort_type=0, const float min_x = MIN_POSITION_NET,
		 const bool use_x_symmetry=false);

  static void code_ball_vel(const AState& state, const Vector& orig, Vector & feat,
		     const int net_sort_type=0,
		     const bool use_x_symmetry=false);

  static float Jnn(const AState& state, const bool testoutput = false, 
	    const int index_of_relevant_player = -1);

  static bool is_offside( const Vector pos);

  static float compute_Jnn(Netdescription &netinfo,const AState& state, 
		    const bool testoutput = false, 
		    const int index_of_relevant_player = -1);

  static bool check_candidate_pass(AAction &candidate_action, const Vector ballpos, const double speed, const double dir,
				   const int desired_receiver = -1);
  static bool is_pass_safe(const Vector ballpos, double & speed, const double dir, Vector & interceptpos, int & advantage, int & receiver,
			   const int desired_receiver = -1);
  static bool check_kick(double &speed, const double dir );

  static bool check_action_penaltyareapass06(const Vector ballpos, AAction &candidate_action, float speed, const float dir);
  static bool is_penaltyareapass_safe(const Vector ballpos,float speed, const float dir, int & advantage, int &number,
				      Vector & playerpos);

  static bool check_pass_with_known_ipos(AAction &candidate_pass, const Vector ballpos);
  static bool compute_ipos_of_candidate_pass(AAction &candidate_pass, const Vector ballpos, const double speed, const double dir);
  static bool check_if_pass_in_back(const Vector ballpos, const Vector ipos, const Vector receiverpos, const double dir );


};

#endif
