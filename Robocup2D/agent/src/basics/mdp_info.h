#ifndef _MDP_INFO_
#define _MDP_INFO_

#include "Cmd.h"
#include "globaldef.h"
#include "mdpstate.h"
#include "mdp_memory.h"
#include "Vector.h"
#include "options.h"
#include "intentiontype.h"
//#include "goals_admin.h"

//#include "types.h"


/** @ralf:
    This class allow access to the values of mdp-state all over the program, 
    because all variables and functions are static. You have only call 
    update(mdp-state) normaly at begin of  behave(...) to have actuall data.

    Some additional functions like "is_ball_kickable()" will help you to 
    program easier. 
*/

/** States of Stamina Level
 */
#define STAMINA_STATE_FULL 0
#define STAMINA_STATE_OK 1
#define STAMINA_STATE_ECONOMY 2
#define STAMINA_STATE_RESERVE 3


typedef struct {
  double ball_near_pos_avg,ball_dist_pos_avg,ball_near_age_avg,ball_dist_age_avg;
  double ball_near_vel_avg,ball_dist_vel_avg;
  double ball_kr_pos_avg,ball_kr_age_avg,ball_kr_vel_avg;
  long ball_near_cyc,ball_dist_cyc,ball_kr_cyc,ball_near_unk,ball_dist_unk,ball_kr_unk;
  double me_pos_avg,me_vel_avg,me_age_avg;
  long me_cyc;
  double team_near_pos_avg,team_dist_pos_avg,team_unk_avg,team_near_age_avg,team_dist_age_avg;
  double opp_near_pos_avg,opp_dist_pos_avg,opp_unk_avg,opp_near_age_avg,opp_dist_age_avg;
  long all_cycles,called_cyc;
  long team_near_cyc,opp_near_cyc,team_dist_cyc,opp_dist_cyc;
  double goalie_near_pos_avg,goalie_dist_pos_avg,goalie_near_age_avg,goalie_dist_age_avg;
  long goalie_near_cyc,goalie_dist_cyc,goalie_unk;
  double my_offside_l_avg,my_offside_r_avg,his_offside_l_avg,his_offside_r_avg;
  long my_offside_l_cyc,my_offside_r_cyc,his_offside_l_cyc,his_offside_r_cyc;
  long ball_in_kr_err,ball_out_kr_err,ball_kr_vel_wrong,ball_kr_pos_wrong;
  double ang_with_ball_avg,ang_without_ball_avg;
  long ang_with_ball_cyc,ang_without_ball_cyc;
  
} StatData;

class mdpInfo {

 public:
  /** constant meaning that any number of  steps can be done until stamina is below recovery_threshold
      occurs for example near the end of a halftime.
   */
  static const int stamina4infinite_steps;
   /** To have access of all values of the MDPstate
   */
  static /*const*/ MDPstate* mdp; //const kommt wieder rein ART!

  /** information about current decision
   */
  //static IntentionType decision_info;
  static IntentionType intention[11];
    
  static Cmd *my_current_cmd;

  /** To have access of all values of the server state - special feature
   */
  static MDPstate* server_state;

  /** To have access of all values of the MDPmemory
   */
  static MDPmemory* memory;

  /** To set the actual MDPstate, which be used by all functions and in the hole agent.
   */
  static void update( MDPstate &m);
  /** To init mdpInfo.
   */
  static void init();
  /** Clear intentions of mine 
   */
  static void clear_my_intention();
  /** Clear my neck intention */
  static void clear_my_neck_intention();
  
  /** Clear intentions of my team
   */
  static void clear_intentions();
  /** set intention of teammate player_no */
  static void set_intention(int player_no, int type, int time, float p1=0., float p2=0., float p3=0., 
			    float p4=0., float p5 = 0.);
  /** set intention of teammate player_no by parsing a heart information */
  static void set_intention_by_heart_info(char* heartinfo);
  /** returns intention type of teammate player_no and also returns its intention 
      if player == me, then my current decision is returned, otherwise heart information from teammates
      is given */
  static int get_intention(int player_no, IntentionType &player_intention);
  /** returns intention type of teammate player_no 
   if player == me, then my current decision is returned, otherwise heart information from teammates
   is given */  
  static int get_intention(int player_no);
  
  /** returns my intention type also returns intention */ 
  static int get_my_intention(IntentionType &player_intention);

  /** NECK INTENTIONS ARE OBSOLETE AND SHOULD NO LONGER BE USED!   */
  /*  Please see basics/blackboard.h instead!                      */
  
  /** returns my neck intention type also returns intention */ 
  static int get_my_neck_intention(IntentionType &player_intention);
  
  /** returns my intention type */
  static int get_my_intention();
  /** returns my neck intention type */
  static int get_my_neck_intention();
  
  /** set my intention */
  static void set_my_intention(int type, float p1=0., float p2=0., float p3=0., 
			       float p4=0., float p5 = 0.);
  /** set my neck intention */
  static void set_my_neck_intention(int type, float p1=0.);
  
  /** To set the actual MDPmemory, which be used by all functions and in the hole agent.
   */
  static void update_memory( MDPmemory &m);

  // ***********************************************************************************
  // **   Now follows additional functions
  // **
  // **   pos = position
  // **   vel = velosity
  // **   abs = absolute
  // **   rel = relative to the player itself (depends on own angle and positon)

  // 1. Functions for the player itself:

  static Vector my_expected_pos_abs_with_dash(double dash_power);
  static Vector my_expected_vel_with_dash(double dash_power);
  /** gets width of view angle  of the player as a radian
      wide -> PI
      normal -> PI/2.0
      narrow -> PI/4.0
      ServerOptions::visible_angle can influence this values
        (PI == ServerOptions::visible_angle * PI/90.0);
      NOTE: This now returns the view angle of the NEXT cycle if BS02_View strategy is used!!!
  */
  static Angle view_angle_width_rad();
  static ANGLE cur_view_angle_width();
  static ANGLE next_view_angle_width();
  
  /** calculates my angle to another object. */
  static Angle my_angle_to(Vector target);
  /** calculates my absolute angle to another object. */
  static Angle my_abs_angle_to(Vector target);
  /** calculates my neck angle to another object. */
  static Angle my_neck_angle_to(Vector target);
  /** checks whether the ball can be sensed.*/
  static bool is_ball_infeelrange();
  /** checks whether the goalie can catch the ball
      note the ban_cycle is not considered, so please do it in your policy 
  */
  static bool is_ball_catchable();
  static bool is_ball_catchable_exact();
  static bool is_ball_catchable_next_time();
  inline static Vector opponent_goalpos(){return Vector(ServerOptions::pitch_length/2.,0);};


  /** Returns if or if not ball is kickable in the next simulator step, when no kick command effects it. */
  static bool is_ball_kickable_next_time();
  static bool is_ball_infeelrange_next_time();
  static bool is_ball_outfeelrange_next_time();

  // These functions also care for an issued body turn command.
  
  /** calculates the expected relative turn angle (if Cmd is already issued) */
  static Angle my_expected_turn_angle();
  /** calculates the needed turn neck moment to achieve desired (abs) angle */
  static double moment_to_turn_neck_to_abs(Angle abs_dir);
  static double moment_to_turn_neck_to_rel(Angle rel_dir);
  /** gets the maximum angle I could see (abs, right) */
  static Angle my_maximum_abs_angle_to_see();
  /** gets the minimum angle I could see (abs, left) */
  static Angle my_minimum_abs_angle_to_see();
  /** returns true if abs target direction could be seen with appropriate neck turn */
  static bool could_see_in_direction(Angle target);
  
  // 3. Functions for the player's teammates:
  /** gets the index in the array at MDPstate for the teammate with uniform number 'number'.*/
  static int teammate_array_index(int number); 
  /** checks whether the position of a teammate has been accurately determined. */
  static bool is_teammate_pos_valid(int number);
  /** use this method, if you already have a reference to the player O(1) **/
  static bool is_teammate_pos_valid(const FPlayer &);
  /** gets the absolute position of a teammate. */
  static Vector teammate_pos_abs(int number);
  /** checks whether the velocity of a teammate has been accurately determined. */
  static bool is_teammate_vel_valid(int number);
  /** gets the absolute velocity of a teammate. */
  static Vector teammate_vel_abs(int number);

  //x-pos of last field player
  static double last_player_line();


  // 4. Functions for the player's opponents:
  /** gets the index in the array at MDPstate for the opponent with uniform number 'number'.*/
  static int opponent_array_index(int number); 
  /** checks whether the position of an opponent has been accurately determined. O(n) */
  static bool is_opponent_pos_valid(int number);
  /** use this method, if you already have a reference to the player. O(1) **/
  static bool is_opponent_pos_valid( const FPlayer & );

  /** gets the absolute position of an opponent. */
  static Vector opponent_pos_abs(int number);
  /** gets the absolute velocity of a teammate. */
  static Vector opponent_vel_abs(int number);
  /** gets the absolute orientation of a teammate. */
  static Angle opponent_ang_abs(int number);
  /** gets the distance of an opponent to another object. */
  static double opponent_distance_to(int number, Vector taret);
  /** gets the distance of an opponent to the ball .*/
  static double opponent_distance_to_ball(int number);
  /** gets the distance of the opponent to me. */
  static double opponent_distance_to_me(int number);
  /** gets the number of the closest opponent to an object DOES NOT CONSIDER FELT PLAYERS!*/
  static int opponent_closest_to(Vector target);
  /** gets the closest opponent to an arbitrary target, returns op. dist and dir from target */
  static int opponent_closest_to(Vector target,double &dist,double &dir);
  /** gets the closest opponent to an arbitrary target, returns op. dist and dir from target */
  static int opponent_closest_to(Vector target,double &dist,double &dir,Vector &pos);
  /** gets the closest opponent to the ball. */
  static int opponent_closest_to_ball(double &dist, double &dir, Vector &pos);
  /** gets the closest opponent to the goal - probably the goalie */
  static int opponent_closest_to_opponent_goal();
  /** gets the goalie age - if goalie not known -1 is returned */
  static int opponent_goalie_age();

  /** checks whether the ball is kickable for an opponent. */
  static bool is_ball_kickable_for_opponent(int number);


  // 5a. functions to get knowledge of the goalie numbers:
  /** return our goalies player number */
  static int our_goalie_number() {return 1;}

  /** return opponent goalies player number */
  //D static int their_goalie_number(){return opponent_goalie_number();}

  // 5b. Common functions for all players:

  /** gets the absolute position of an object .*/
  static Vector get_pos_abs(FPlayer player);

  /** gets the age of a player position .*/
  static int teammate_age(int number);
  /** gets the age of player position .*/
  static int age_playerpos(const FPlayer *player);
  /** gets the age of player velocity .*/
  static int age_playervel(const FPlayer *player);
  static int age_opponent_ang(int number);

  // 6. Functions to change representations (e.g. relative to absolute)
  /** convert a position from polar to vector representation. */
  static Vector polar2global_pos(double distance, Angle angle);
  /** checks whether a position is located inside of the pitch. */
  static bool is_position_in_pitch(Vector position, float safety_margin = 0.0);

  // 7. Fuctions which treat properties of whole teams
  /**determines which team got the player who is fastest to ball (ART's method!) */
  static int fastest_team_to_ball();
  /**determines which team is fastest to ball with momentum */
  static int fastest_team_to_ball_momentum();
  /**determines whether my team is attacking */
  static bool is_my_team_attacking();

  // 8. Functions who check positions of objects.
  /** check whether an object is in our penalty area.*/
  static bool is_object_in_my_penalty_area(Vector obj, double safety_margin = 0.0);

  //9. mdp related get functions.

  /** get self.
      @return pointerx to the player himself
  */
 static FPlayer* me();

  /** computes the number of steps that are possible if player dashes with power 
      avoiding that stamina goes below recover_dec_thr * stamina_max
      if no restrictions apply then mdpInfo::stamina4infinite_steps is returned
      @return int number of steps that are possible
  */
 static int stamina4steps_if_dashing(double power);
 /** computes the number of meters that are possible if player dashes with power 
      uses stamina4steps_if_dashing()
 */
 static double stamina4meters_if_dashing(double power);
  /** computes the available Power regarding recoverz threshold
  */
 static double stamina_left();
  /** returns the current state of stamina level
  */
 static int get_stamina_state();

  /** get the current playmode */
 static int play_mode();

  /** get a player from our team.
     @param player_nbr (range = 0-10)
     @return a pointer to a player
  */
 static const FPlayer* my_team(int player_nbr);
  /** get a player from their team.
     @param player_nbr (range = 0-10)
     @return a pointer to a player
  */
 static const FPlayer* his_team(int player_nbr);


 static bool do_I_play_with_fullstate();

 static void visualize_mdp_state();
 static void visualize_server_state();
 static void clear_mdp_statistics();
 static void print_mdp_statistics();

 private:

 static StatData stats;
 
 public:
/*  private: */
/*   mdpInfo(){} */
/*   ~mdpInfo(){} */
};

#endif
