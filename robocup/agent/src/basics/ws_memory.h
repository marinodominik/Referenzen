#ifndef _WS_MEMORY_H_
#define _WS_MEMORY_H_

/* Author: Manuel Nickschas */

#include "tools.h"

#define MAX_VIEW_INFO 10          // number of view infos stored (one for every cycle)
#define MAX_ATTENTIONTO_INFO 10
#define MAX_INTERCEPT_INFO_PLAYERS 5

//TG09: Selfpass-related constants // CR16 moved from globaldef.h
#define MAX_SELFPASS_RISK_LEVELS 10


class WSmemory {

  struct ViewInfo {
    ANGLE view_width;
    ANGLE view_dir;
    Vector ppos;
    long time;
  };
  struct AttentionToInfo
  {
    int  playerNumber;
    long time;
  };

  static ViewInfo view_info[];
  static AttentionToInfo attentionto_info[];

  static void add_view_info(long cyc, int view_ang, int view_qty, ANGLE view_dir,Vector ppos);
  static void add_attentionto_info(long cyc, int plNr);

  static const int MAX_MOMENTUM=5;
  static float momentum[MAX_MOMENTUM];
  static int counter;
  static int saved_team_last_at_ball;
  static int saved_team_in_attack; //gibt an, welches Team im Angriff ist
  static double his_offside_line_lag2,his_offside_line_lag1,his_offside_line_lag0;
  static int cvTimesHisGoalieSeen, cvTimesHisGoalieOutsidePenaltyArea;
  static int last_update_at;
  static long cvOurLastSelfPass, cvOurLastSelfPassWithBallLeavingKickRange;
  static int  cvOurLastSelfPassRiskLevel;
  static int  cvSelfpassAssessment[MAX_SELFPASS_RISK_LEVELS][3];
  static int  cvLastTimeFouled;
  static int  cvLastFouledOpponentNumber;
  static int  cvLastHisGoalKick;
  
 public:

  //jk change: was private
  static long teammate_last_at_ball_time;
  static int teammate_last_at_ball_number; // include myself!
  static int opponent_last_at_ball_number;
  static long opponent_last_at_ball_time;

  static InterceptResult cvCurrentInterceptResult[MAX_INTERCEPT_INFO_PLAYERS];
  static PlayerSet          cvCurrentInterceptPeople;

  static long last_kick_off_at;
  static long ball_was_kickable4me_at;

  /* ridi 18.6.03: these return (ws->time +1) if we haven't seen object within
     the last MAX_VIEW_INFO cycles, this makes the use much easier */
  static long last_seen_in_dir(ANGLE direction);
  static long last_seen_to_point(Vector point);
 
  static long last_attentionto_to(int plNr);

  static int team_last_at_ball() {
    return saved_team_last_at_ball;
  }
  static int team_in_attack() {
    return saved_team_in_attack;
  }
  static int get_teammate_last_at_ball_number() {
    return teammate_last_at_ball_number;
  }
  static int get_teammate_last_at_ball_time() {
    return teammate_last_at_ball_time;
  }
  static int get_opponent_last_at_ball_number() {
    return opponent_last_at_ball_number;
  }
  static int get_opponent_last_at_ball_time() {
    return opponent_last_at_ball_time;
  }
  static int get_last_fouled_time() {
    return cvLastTimeFouled;
  }
  static int get_last_his_goal_kick_time() {
    return cvLastHisGoalKick;
  }
  static int get_last_fouled_opponent_number() {
    return cvLastFouledOpponentNumber;
  }

  static bool get_view_info_before_n_steps(int n, 
                                           ANGLE & viewWidth, 
                                           ANGLE & viewDir, 
                                           Vector & playerPos, 
                                           long & globalTime); 

  static double get_his_offsideline_movement();
  static int   get_his_goalie_classification();
  static bool  getSelfpassMemoryForRiskLevel( int riskLevel,
                                              int & numberOfSuccessfulSelfpasses,
                                              int & numberOfUndecidedSelfpasses,
                                              int & numberOfFaultySelfpasses
                                             );

  static void update_offside_line_history();
  static void update_his_goalie_classification();
  static void update_fastest_to_ball();
  static void update_our_selfpass_assessment();
  static void update_our_selfpass_memory();
  static void update();

  static void init();
  static void incorporateCurrentCmd( const Cmd& cmd );


};

#endif
