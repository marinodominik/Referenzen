#ifndef _POLICY_TOOLS_H
#define _POLICY_TOOLS_H

#include "../basics/PlayerSet.h"
#include "globaldef.h"
#include "Vector.h"


#define DEFENSE_AREA_X 16.0
#define DEFENSE_AREA_Y 16.0
#define DEFENSE_OFFSET 8.0
#define MAX_STEPS2INTERCEPT 40


/*****************************************************************************/

struct Pass{
  int player_num;
  Vector target;
  float velocity;
};

/*****************************************************************************/

struct Go2Ball_Steps {
  const static int MY_TEAM = 1;
  const static int THEIR_TEAM = -1;
  const static int NO_TEAM = 0;
  
  int number;
  int side;
  int steps;
};

/*****************************************************************************/

/** The class MachtPositions implements am matching mechanism. 
    It can match up ton num_players_MAX to the maximum of num_targets_MAX 
    of targets. Both players and targets are represented as vectors.

    The matching mechnism does use a heuristic. It doesn't consider all
    possible matchings, but instead every target can just be matched by
    one of the nearest 3 players.

    if num_targets_MAX=6, then ther are
    3^num_targets_MAX= 3^6= 729  possible solutions
    
    You can also use more then 3 nearest neighbours, but the code is optimized
    just for 3, so you will have to change some details (i.e. the computation of
    nearest neighbours for the targets)
*/
class MatchPositions {
  static const int num_players_MAX= NUM_PLAYERS;
  static const int num_targets_MAX= 6;

  int best_sol[num_players_MAX];

  struct Dist_2_Player {
    Dist_2_Player() { player= -1; dist= 4*FIELD_BORDER_X; }
    Dist_2_Player(int p,double d) { player= p; dist= d; }
    int player;
    double dist;
  };
  Dist_2_Player nearest_from_target[num_targets_MAX][3];
 public:
  MatchPositions() {}
  /** if num_players < num_targets  the will also a solution, but targets
      with lower index will be taken first.
  */

  void match(const Vector* players, int num_players,
	     const Vector* targets, int num_targets);
  /** get the index idx of target, so that players[num] is matched to
      target[idx]. 
      If idx is < 0, the players[num] has no target to go!
   */
  int get_match_for(int num) { return best_sol[num]; }
};

/*****************************************************************************/

/** We need this class for our matching algorithm */
class PlayerMatch{
 public:
  void init(int index, 
	    const Vector me,
	    const Vector* targets, int num_targets);
  double *distances;
  int mapped_to;
  int my_index;
  int shortest_index;
};

/** The class MatchPositions implements another matching algorithm.
    It matchs 10 Players to 10 Positions. First the player with the longest way 
    to go to his closest Position gets assigned that position. We repeat this process
    with the remaining Players and Positions until everyone has a position.
    Note that this is a suboptimal heuristic, but in many cases it works quite 
    well, because everybody will have to run a bit and not, like in many other 
    matching algorithms, one player has to run a very long way while the rest does'nt
    have to move. Additionally the algorithm is quite fast...
*/
class MatchPositionsAlternative{
 private:
  int n_players, n_targets;
  int *map;
  PlayerMatch *player;
  int min_element(double *array, int array_length);
  int max_element(PlayerMatch *array, int array_length);
  int temp_max;
 public:
  /** constructor.
      Please be sure to have at least as many positions as you have players !
  */
  MatchPositionsAlternative(int num_players=10, int num_targets=10);
  /** we try to match num_players players to num_targets targets */
  void match(const Vector* players, int num_players,
	     const Vector* targets, int num_targets);
  /** get the index idx of target, so that players[num] is matched to
      target[idx]. 
      If idx is < 0, the players[num] has no target to go!
   */
  int get_match_for(int num) { return map[num]; }
};

/*****************************************************************************/

/** The third Matching Algorithm does not optimize a certain property for all
    vertices and edges, instead it tries to optimize this property in a way
    that its distribution across the vertices is optimized.

    Example for betting understanding of the above gibberish:
    property=distance, thus not the matching for minimized sum over all
    edges is searched for, but the matching for minimizing the maximum
    distance one(!) player has to walk to its assigned position.

    The class uses the QuickSort Algorithm (by Artur Merke).
*/

class MatchDistribution{

 protected:

  struct val2ord{
    float value;
    int player;
    int position;
  };
  int *map;
  val2ord *v;

  void exchange(int i,int j, val2ord * v);
  void qsort(int size,val2ord * v);
  void convert(float * values, int num_players, int num_positions);

 public:
  void match(float * values, int num_players, int num_positions);

  int get_match_for(int num){return map[num];}

};


/***************************************************************************/
class Policy_Tools {
 private:
  static double goalshot_goalie_bonusstep;  // goalie is 1 step closer to ball
  static double goalshot_corner_tolerance;
  static double goalshot_consider_age;
  static double goalshot_worstcase_consider_age;
  static double extremeshot_goalie_maxspeed_percentage;
  static double extremeshot_goalie_playersize_percentage;
  static bool go2ball_list_analytical;
  static Go2Ball_Steps go2ball_list[22];
  static int selfpass_optime2react;
  static int selfpass_my_minadvantage;
  static int pass_optime2react;
  static int pass_my_minadvantage;
  static float defense_area;
  static bool turnneck2ballholder;
  static bool use_go2ball_classic;

  static int last_go2ball_steps_update;
  static int last_intercept_pos_update;
  static Vector saved_intercept_pos;

  static int opteam_time2intercept(const Vector ballpos,const double speed,
				   const double dir,
				   Vector &interceptpos,
				   const int op_time2react = 1,
				   const int critical_value = 0,
				   const bool consider_opdir=false,
				   const Vector centre_of_interest = Vector(0),
				   const float radius_of_interest = 1000.,
				   const int max_age = 1000);

  static int opteam_time2intercept06(const Vector ballpos,const double speed,
				   const double dir,
				   Vector &interceptpos,
				   const int op_time2react = 1,
				   const int critical_value = 0,
				   const bool consider_opdir=false,
				   const Vector centre_of_interest = Vector(0),
				   const float radius_of_interest = 1000.,
				   const int max_age = 1000);


 public:
  static int pass_mytime2react;
  static void init_params();

  /* computes the number of steps to intercept the ball for every player on the pitch and stores the results
     in an array of Go2Ball_Steps structs. set will be sorted by the number of steps to intercept the ball in 
     ascending order.
     Use this set to make a go2ball decision
  */
  static void go2ball_steps_update();
  static Go2Ball_Steps* go2ball_steps_list();
  
  /** returns true if my team can intercept the ball within the pitch
      advantage gives the number of steps before fastest opponent */
  static bool myteam_intercepts_ball(const Vector ballpos,const double speed,
				     const double dir,
				     Vector &intercept_pos,
				     int &advantage,
				     int &number, Vector &playerpos);

  static bool myteam_intercepts_ball_hetero(const Vector ballpos,const double speed,
					    const double dir,
					    Vector &intercept_pos,
					    int &advantage,
					    int &number, Vector &playerpos,
					    const double myteam_time2react = 1000,
					    const double opteam_time2react = 1000);

  static bool myteam_intercepts_ball_hetero06(const Vector ballpos,const double speed,
					    const double dir,
					    Vector &intercept_pos,
					    int &advantage,
					    int &number, Vector &playerpos,
					    const double myteam_time2react = 1000,
					      const double opteam_time2react = 1000,
					      const int desired_receiver = -1);

  static bool opteam_intercepts_ball_faster06(const Vector ballpos,const double speed,
					      const double dir,
					      const int my_num_steps,
					      const Vector my_ipos,
					      int & advantage,
					      const double opteam_time2react = 1000);


  static bool compute_ipos_for_teammate06(const Vector ballpos,const double speed, const double dir, const int  teammate,
					  int &num_steps, Vector & intercept_pos, const int my_time2react=0);


  static bool myplayer_intercepts_ball(const Vector ballpos,const double speed,
				       const double dir,
				       const Vector playerpos,
				       Vector &intercept_pos,
				       int &advantage);

  /** returns the time to intercept the ball from a given position.
      time2react gives the number of cycles the player needs to react
      returns the number of time steps needed.
      if number of time steps exceeds max_steps 70, -1 is returned */
  static int get_time2intercept(Vector & interceptpos,
					      const Vector & ballpos, 
					      const double ballspeed,
					      const double balldir,
					      const Vector & playerpos, 
					      const int time2react=1,
					      int max_steps = MAX_STEPS2INTERCEPT,
					      const bool goalie = false,const float kickrange_percentage = 0.9);

  static int get_time2intercept(Vector & interceptpos,
					      const Vector & ballpos, 
					      const Vector initballvel, 
					      const Vector & playerpos, 
					      const int time2react=1,
					      int max_steps = MAX_STEPS2INTERCEPT,
					      const bool goalie = false,const float kickrange_percentage = 0.9);

  static int get_time2intercept_hetero(Vector & interceptpos,
					      const Vector & ballpos, 
					      const Vector initballvel, 
					      const PPlayer & player, 
					      const int time2react =1,
					      int max_steps= MAX_STEPS2INTERCEPT ,
					      const bool goalie = false,
					      const float kickrange_percentage = 0.9 );

  static int get_time2intercept_hetero(Vector & interceptpos,
					    const Vector & ballpos, 
				     const double ballspeed,
				     const double balldir,
					    const PPlayer & player, 
					    const int time2react = 1,
					    int max_steps= MAX_STEPS2INTERCEPT  ,
					    const bool goalie  = false,
				       const float kickrange_percentage = 0.9 );



  /** computes the earlriest time and position at which the player can
      intercept the ball, the default value player_vel_max= -1.0
      means, that a default value for player velocity will be assumed.  */
  static void intercept_min_time_and_pos_hetero(int & res_time, Vector & res_pos,
					 const Vector & ball_pos, 
					 const Vector & ball_vel, 
					 const Vector & player_pos,
					 int player_number,
					 bool my_team,
					 double player_vel_max,
					 double player_size);

  static void intercept_min_time_and_pos_hetero(int & res_time, Vector & res_pos,
					 const Vector & ball_pos, 
					 const double ballspeed,
					 const double balldir,
					 const Vector & player_pos, 
					 int player_number,
					 bool my_team,
					 double player_vel_max,
					 double player_size);
  /** computes the earlriest time and position at which 
      player A, dashing with speed in direction dir can be
      intercepted by player B */
  static void intercept_player(int & res_time, Vector & res_pos,
					 const Vector A_pos, 
					 const double A_speed,
					 const double A_dir,
					 const Vector B_pos,
					 double player_vel_max= -1.0,
					 double player_size= -1000.0);
  

  /** compute position where the ball is caught if shot with speed in direction dir
      returns true, if intercept position is inside pitch
      returns if myteam gets the ball, the intercept position and the player number 
  */
  static bool earliest_intercept_pos(const Vector ballpos,const double speed, const double dir,
				     Vector &intercept_pos,bool &myteam, int &number, int &advantage,
				     double speed_factor = -1.0);
  static bool earliest_intercept_pos(const Vector ballpos,const double speed, const double dir,
				     Vector &intercept_pos,bool &myteam, int &number,
				     double speed_factor = -1.0);

  /** compute position where the ball intercepted wrt. to the actual game state by calling 
      earliest_intercept_pos(...)
  */
  static Vector next_intercept_pos(double speed_factor = -1.0);

  /** compute the best direction for clearing out of testdir list.
      returns true if either teammate has a chance of getting the ball or ballgain > min_gain (20m)
  */
  static bool get_best_clearance_kick(const Vector ballpos, const int testdirs,
				      const double testdir[], double testspeed[],
				      double &speed, double &dir);
  /** compute the best direction for kick and rush of testdir list.
      returns true if our teammate has a chance to get the ball at first
      and false else
  */
  static bool get_best_kicknrush(const Vector ballpos, const int testdirs,
				 const double testdir[], double testspeed[],
				 double &speed, double &dir, Vector &resulting_pos);
  static bool get_best_kicknrush(const Vector ballpos, const int testdirs,
				 const double testdir[], double testspeed[],
				 double &speed, double &dir, Vector &resulting_pos,
				 int &advantage, int &closest_teammate);
  static bool isKicknRushInterceptionPointAcceptable( Vector ipos, 
                                                      int    advantage );  
  /* defense stuff */
  static void get_opponent_attackers(int *attacker_number, int max_attackers, int op_ball_holder=0);
  static void check_if_turnneck2ball(int teammate_steps2ball, int opponent_steps2ball);

  /* communicate intentions */
  static bool check_go4pass(Vector &ipos, Vector &ballpos, Vector &ballvel);
  static bool check_go4pass(Vector &ipos);
  static bool I_am_fastest4pass(const Vector ballpos,const double speed,
				const double dir, Vector &intercept_pos);

  static bool myplayer_intercepts_ball_hetero(const Vector ballpos,const double speed,
					      const double dir,
					      const PPlayer player,
					      Vector &intercept_pos,
					      int &advantage);

  static int get_selfpass_advantage(const double speed, const double dir, Vector &interceptpos);


  static bool use_clever_moves;
  static Vector ballpos;
  static bool goaliestandards;
};


#endif
