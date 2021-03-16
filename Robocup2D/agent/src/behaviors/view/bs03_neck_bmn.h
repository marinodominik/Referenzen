#ifndef _BS03_NECK_H_
#define _BS03_NECK_H_

#include "../base_bm.h"
#include "globaldef.h"
#include "angle.h"
#include "ws_info.h"
#include "neck_cmd_bms.h"
#include "goalie_neck_bmn.h"

#define BS02_NECK_SECT_X 21
#define BS02_NECK_SECT_Y 13

/**
 */
class  BS03Neck : public NeckBehavior {
  static bool initialized;
  NeckCmd *neck_cmd;
  GoalieNeck *goalie_neck;

  struct Neck_Info {
    int sector[BS02_NECK_SECT_X][BS02_NECK_SECT_Y];  // age of field sectors
    double sector_avg,penalty_avg;
    int opp_player[NUM_PLAYERS];
    int own_player[NUM_PLAYERS];
    int ball;
    int opp_goal,opp_goal_left,opp_goal_right,own_goal;
    int opp_goalie;
    int opp_offside,own_offside;
    double opponents;
    double teammates;
    
    Neck_Info();
  };
  


 public:

  BS03Neck();
  virtual ~BS03Neck();

  bool get_cmd(Cmd &cmd);
  
  static bool init(char const * conf_file, int argc, char const* const* argv);

 private:

  int intercept_look_mode;
  int opp_has_ball_look_mode;
  int own_has_ball_look_mode;
  int ball_holder_look_mode;
  bool ignore_neck_intentions;
  bool use_1v1_mode;
  
  double xunit;
  double yunit;

  Angle norm_view_ang;
  Angle minang,maxang;
  bool center_target;
  
  bool got_update;
  bool ball_already_searched;
  bool need_lookto_ball;

  long last_looked_to_goalie;
  
  /* next cycle positions */
  Vector my_new_pos,my_new_vel,new_ball_pos,new_ball_vel;
  Angle my_new_angle;
  bool ballinfeelrange;
  bool potentialcollision;
  double distance_to_ball;
  ANGLE dir_to_ball;

  PlayerSet players_near_ball,own_players_near_ball,opp_players_near_ball;

  void init_cycle(Cmd &cmd);
  
  /* weights for value function */
  double ball_weight,goalie_weight,sector_weight,opp_weight,team_weight;
  double single_sector_weight[BS02_NECK_SECT_X][BS02_NECK_SECT_Y];
  double single_sector_divisor;
  
  IntentionType neck_intention;  
  double turnback2_scandir;
  
  Neck_Info cur_info,next_info;

  Neck_Info get_neck_info(Vector mypos,ANGLE neckdir,Vector ballpos, Neck_Info neckinfo);
  double get_neckinfo_value(Neck_Info);
  void get_neckinfo_weights();
  void get_single_sector_weights();
  
  Angle get_best_angle(Angle &target);
  Angle neck_value_based(Angle target);
  
  bool can_see_object(Vector mypos,ANGLE neckdir,Vector objpos);
  Angle get_dir_of_nearest_opponent();

  Angle check_neck_1v1();
  Angle check_neck_intention();
  Angle check_intercept();
  Angle check_players_near_ball();
  Angle check_direct_opponent_defense();
  Angle check_search_ball();
  Angle check_block_ball_holder();
  Angle check_offside();
  Angle check_goalie();
  Angle check_relevant_teammates();

  Angle neck_default(Angle preset);
  
  bool neck_lock(Cmd &cmd);        /** locks neck dir to body dir */

  Angle intention_lookindirection();
  Angle intention_direct_opponent_defense();
  Angle intention_passindirection();
  Angle intention_blockballholder();
  Angle intention_faceball();
};

#endif //_BS03_NECK_H_

