#ifndef _TOOLS_H
#define _TOOLS_H

#include <stdio.h>

#include "server_options.h"
#include "abstract_mdp.h"
#include "mystate.h"


#define NONE 0
#define LOW 1
#define MEDIUM 2
#define HIGH 3
#define VERY_HIGH 4

#define EQUAL 0
#define FIRST 1
#define SECOND 2



class Tools {
  static const int MAX_NUM_POWERS= 100;
  static int num_powers;
  static double ball_decay_powers[MAX_NUM_POWERS];
  static AState state;
  static int state_valid_at;
  static PPlayer modelled_player;
  
 public:

  static bool deduce_action_from_states( Cmd_Body&     action,
                                         const PPlayer player,                double distToOtherPlayer,
                                         const Vector& oldPlayerPos,          Vector oldPlayerVel, ANGLE oldPlayerAng,
                                         const Vector& newPlayerPos,          Vector newPlayerVel, ANGLE newPlayerAng,
                                         const bool    ignoreBall,            double distToBall = 0.0,
                                         const Vector& oldBallPos = Vector(), Vector oldBallVel = Vector(),
                                         const Vector& newBallPos = Vector(), Vector newBallVel = Vector()
                                       );

  static double getViewNoiseFromDist( double distance );

  static Vector ip_with_fieldborder(const Vector p, const float dir);
  static Vector ip_with_right_penaltyarea(const Vector p, const float dir);
  static double ballspeed_for_dist(const double dist);

  /** Return the angle in [-Pi,+Pi) */
  static Angle get_angle_between_mPI_pPI(Angle angle);
  /** Return the angle in [0,+2*Pi) */
  static Angle get_angle_between_null_2PI(Angle angle);

  static double get_tackle_success_probability(Vector my_pos,
                                              Vector ball_pos, 
                                              double my_angle,
                                              bool foul=false);
  static double get_foul_success_probability( Vector my_pos,
                                             Vector ball_pos,
                                             double  my_angle);

  /** Return the absolut size of angle */
  static Angle get_abs_angle(Angle angle);

  /** calculates my angle to another object. */
  static ANGLE my_angle_to(Vector target);

  static ANGLE my_abs_angle_to(Vector target); 
  
  /** calculates my neck angle to another object. */
  static ANGLE my_neck_angle_to(Vector target);

  // >>>>>>>>>>>>>>>>>>>>
  // These functions work only if Cmd_Main has already been set!!
  
  /** returns expected relative body turn angle */
  static ANGLE my_expected_turn_angle();

  static double moment_to_turn_neck_to_abs(ANGLE abs_dir);

  /** gets the maximum angle I could see (abs, right) */
  static ANGLE my_maximum_abs_angle_to_see();

  /** gets the minimum angle I could see (abs, left) */
  static ANGLE my_minimum_abs_angle_to_see();

  /** returns true if abs direction can be in view_angle with appropriate neck turn */
  static bool could_see_in_direction(ANGLE target_ang);
  static bool could_see_in_direction(Angle target_ang);
  
  // <<<<<<<<<<<<<<<<<<<<<
  
  /** Return a random integer with 0 <= int < range. */
  static int int_random(int range);

  static double range_random(double lo, double hi);
  
  static int very_random_int(int n);

  inline static bool equal(double x,double y,double epsilon = 0.00001)
    {return(fabs(x-y)<epsilon);};

  /** Return the maximum */
  static double max(double a, double b);
  static double min(double a, double b);
  static int max(int a, int b);
  static int min(int a, int b);


  static double get_dash2stop();
  static bool get_cmd_to_stop(Cmd & cmd);

  static bool is_a_scoring_position(Vector pos);


  static Vector get_Lotfuss(Vector x1, Vector x2, Vector p);
  static double get_dist2_line(Vector x1, Vector x2, Vector p);

  //art: returns time in ms since first call to this routine 
  static long get_current_ms_time();
  
  // returns info string. info[] must have length of 50 
  static void cmd2infostring(const Cmd_Body & cmd, char *info);

  // predicts player movement only. calls model_cmd_main.
  static void model_player_movement(const Vector & old_my_pos, const Vector & old_my_vel, 
				    const Angle & old_my_ang,
				    const Cmd_Body & cmd,
				    Vector & new_my_pos, Vector & new_my_vel,
				    Angle & new_my_ang,
				    const bool do_random=false);

  /** this method can model the soccerserver if you apply a basic command
      dash and turn concern always the my_* parameters
   */
  static void model_cmd_main(const Vector & my_pos, 
			     const Vector & my_vel, 
			     const Angle & my_ang,
			     const int &old_my_stamina,
			     const Vector & ball_pos,
			     const Vector & ball_vel,
			     const Cmd_Body & cmd, 
			     Vector & new_my_pos,
			     Vector & new_my_vel,
			     Angle & new_my_ang,
			     int &new_my_stamina,
			     Vector & new_ball_pos,
			     Vector & new_ball_vel, const bool do_random = false);

  static void model_cmd_main(const Vector & my_pos, 
			     const Vector & my_vel, 
			     const Angle & my_ang,
			     const Vector & ball_pos,
			     const Vector & ball_vel,
			     const Cmd_Body & cmd, 
			     Vector & new_my_pos,
			     Vector & new_my_vel,
			     Angle & new_my_ang,
			     Vector & new_ball_pos,
			     Vector & new_ball_vel, const bool do_random = false);
  
  /* same as above, but using ANGLE instead of Angle */
  static void model_cmd_main(const Vector & my_pos, 
			     const Vector & my_vel, 
			     const ANGLE & my_ang,
			     const Vector & ball_pos,
			     const Vector & ball_vel,
			     const Cmd_Body & cmd, 
			     Vector & new_my_pos,
			     Vector & new_my_vel,
			     ANGLE & new_my_ang,
			     Vector & new_ball_pos,
			     Vector & new_ball_vel, const bool do_random = false);


  /* same as model_... as above, uses mystate*/
  static void get_successor_state(MyState const &state, Cmd_Body const &cmd, MyState &next_state,
				  const bool do_random = false);

  /* the model_cmd_main series of methods does not cover tackle
     commands; this method can be used for version 12 of
     soccer server only (angular tackle)*/
  static void model_tackle_V12( const Vector & playerPos,
                                const ANGLE  & playerANG,
                                const Vector & ballPos,
                                const Vector & ballVel,
                                const int    & tackleAngle,
                                Vector       & ballNewPos,
                                Vector       & ballNewVel);

  /* Berechnet den Schnittpunkt zweier Geraden */
  static Vector intersection_point(Vector p1, Vector steigung1, 
				   Vector p2, Vector steigung2);

  static Vector point_on_line(Vector steigung, Vector line_point, double x);

  static bool intersection(const Vector & r_center, double size_x, double size_y,
				  const Vector & l_start, const Vector & l_end);

  //a triagle is defined as the convex hull of the three points t1,..,t3, which can be chosen arbitrary
  static bool point_in_triangle(const Vector & p, const Vector & t1, const Vector & t2, const Vector & t3);

  //a rectangle is defined as the convex hull of the four points r1,..,r4, which can be chosen arbitrary
  static bool point_in_rectangle(const Vector & p, const Vector & r1, const Vector & r2, const Vector & r3, const Vector & r4);

  //extra_margin can be negative to make the field smaller
  static bool point_in_field(const Vector & p, double extra_margin= 0.0);

  /** it's a cached and lazy method to computed powers of ball_decay, (questions: ask art!)
      if the power is bigger then the max cache size, a warning will inform you about it.
   */
  static double get_ball_decay_to_the_power(int power);

  inline static Vector opponent_goalpos()
  {
      return Vector(ServerOptions::pitch_length/2.,0);
  }

  static double get_max_expected_pointto_error(double dist);
  /* here comes stuff used by neck and view behaviors   */
  
  /* calculate ANGLE from NARROW, NORMAL or WIDE */
  static ANGLE get_view_angle_width(int view_ang);

  /* These use the blackboard to get their information. */
  static ANGLE cur_view_angle_width();
  static ANGLE next_view_angle_width();
  static int get_next_view_angle();
  static int get_next_view_quality();
  static long get_last_lowq_cycle();
  static Vector get_last_known_ball_pos();
  static void force_highq_view();
  
  static void set_neck_request(int req_type, double param = 0, bool force = false);
  static void set_neck_request(int req_type, ANGLE param, bool force = false);
  static int get_neck_request(double &param);    // returns NECK_REQ_NONE if not set
  static int get_neck_request(ANGLE &param);    // returns NECK_REQ_NONE if not set

  static void set_attention_to_request(int plNr, bool force = false);
  static int  get_attention_to_request();

  static bool is_ball_kickable_next_cycle(const Cmd &cmd, Vector & mypos,Vector & myvel, ANGLE &newmyang,
					  Vector & ballpos,Vector & ballvel);

  static bool is_ballpos_safe(const Vector &oppos, const ANGLE &opbodydir,
			      const Vector &ballpos, int bodydir_age);
  static bool is_ballpos_safe(const PPlayer opp,const Vector &ballpos,bool consider_tackles=false);
  static bool is_ballpos_safe(const PlayerSet &opps,const Vector &ballpos,bool cons_tackles=false);
  static bool is_ball_safe_and_kickable(const Vector &mypos, const Vector &oppos, const ANGLE &opbodydir,
					const Vector &ballpos, int bodydir_age);
  static bool is_ball_safe_and_kickable(const Vector &mypos, const PPlayer opp,const Vector &ballpos,
					bool consider_tackles=false);
  static bool is_ball_safe_and_kickable(const Vector &mypos, const PlayerSet &opps,const Vector &ball,
					bool consider_tackles=false);
  static bool is_position_in_pitch(Vector position,  const float safety_margin = 1.0 );

  static bool shall_I_wait_for_ball(const Vector ballpos, const Vector ballvel, int &steps);
  
  static double speed_after_n_cycles(const int n, const double dash_power_rate,
				    const double effort, const double decay);

  static bool is_pos_free(const Vector & pos);
  static double eval_pos_wrt_position(const Vector & pos,const Vector & targetpos, const double mindist2teammate = 2.0);
  static double eval_pos(const Vector & pos,const double mindist2teammate = 3.0);

  static double evaluate_wrt_position(const Vector & pos,const Vector & targetpos);
  static double get_closest_op_dist(const Vector pos);
  static double get_closest_teammate_dist(const Vector pos);
  static double get_optimal_position(Vector & result, Vector * testpos,
				    const int num_testpos,const PPlayer &teammate);

  static PPlayer get_our_fastest_player2ball(Vector &intercept_pos, int & steps2go);
  static bool is_pos_occupied_by_ballholder(const Vector &pos);
  static int num_teammates_in_circle(const Vector pos, const double radius);
  static Vector check_potential_pos(const Vector pos, const double max_advance= 10);
  static double evaluate_pos(const Vector query_pos);
  static double evaluate_pos_selfpass_neuro(const Vector query_pos);
  static double evaluate_pos_selfpass_neuro06(const Vector query_pos);

  static double evaluate_pos_analytically(const Vector query_pos);
  static double evaluate_potential_of_pos(const Vector pos);
  static int compare_two_positions(Vector pos1, Vector pos2);
  static double min_distance_to_border(const Vector position);
  static void display_direction(const Vector pos, const ANGLE dir, const double length, const int color = 0);
  static bool can_advance_behind_offsideline(const Vector pos);
  static int potential_to_score(Vector pos);

  static int compare_positions(const Vector pos1, const Vector pos2, double & difference);

  static bool close2_goalline(const Vector pos);

  static bool is_pos1_better(const Vector pos1, const Vector pos2);

  static bool can_score(const Vector pos, const bool consider_goalie = false); // ridi: before 9.6.06, goal was always not considered
  static bool opp_can_score(const Vector pos);


  static void display_astate();
  static int get_interceptor_in_astate(AState & state);

  static void simulate_player(const Vector & old_pos, const Vector & old_vel, 
			      const ANGLE & old_ang, const int &old_stamina, 
			      const Cmd_Body & cmd,
			      Vector & new_pos, Vector & new_vel,
			      ANGLE & new_ang, int &new_stamina,
			      const double stamina_inc_max,
			      const double inertia_moment,
			      const double dash_power_rate, const double effort, const double decay);

  static bool is_pass_behind_receiver(const Vector mypos, const Vector receiverpos, const double passdir);

  static double get_ballspeed_for_dist_and_steps(const double dist, const int steps);

  static bool can_actually_score(const Vector pos);

  static int get_steps_for_ball_distance(const double dist, double ballspeed);

  static bool willMyStaminaCapacitySufficeForThisHalftime();
  static bool willStaminaCapacityOfAllPlayersSufficeForThisHalftime();

  static ANGLE degreeInRadian( double degree );
  static double radianInDegree( ANGLE radian );

  static void setModelledPlayer( PPlayer p );
};

extern std::ostream& operator<< (std::ostream& o, const MyState& s);

#endif
