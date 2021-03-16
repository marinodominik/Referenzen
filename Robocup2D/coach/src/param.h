/* Author: Manuel "Sputnick" Nickschas
 *
 * This file contains structs that hold the server and player parameters as
 * well as the player types.
 * Each of those struct has a function that parses the param message sent from the
 * server. The ServerParam struct also reads default values from the config file
 * during initialisation (most options will be overwritten by the server_param message
 * afterwards).
 *
 * NOTE: The server config file won't be read for now, maybe it will implemented later... 
 *
 */

#ifndef _PARAM_H
#define _PARAM_H

#include "defs.h"

/** Contains all data sent with the player_param message by the server. */
struct PlayerParam {

  static int player_types;
  static int subs_max;
  static int pt_max;
  static double player_speed_max_delta_min;
  static double player_speed_max_delta_max;
  static double stamina_inc_max_delta_factor;
  static double new_stamina_inc_max_delta_factor;
  static double player_decay_delta_min;
  static double player_decay_delta_max;
  static double inertia_moment_delta_factor;
  static double dash_power_rate_delta_min;
  static double dash_power_rate_delta_max;
  static double new_dash_power_rate_delta_min;
  static double new_dash_power_rate_delta_max;
  static double player_size_delta_factor;
  static double kickable_margin_delta_min;
  static double kickable_margin_delta_max;
  static double kick_rand_delta_factor;
  static double extra_stamina_delta_min;
  static double extra_stamina_delta_max;
  static double effort_max_delta_factor;
  static double effort_min_delta_factor;
  static double random_seed;
  
  static void init();
  static bool parseMsg(const char*);
};

/** Contains all data sent with the player_type message by the server. */
struct PlayerType {
  
  int id;
  double player_speed_max;
  double stamina_inc_max;
  double player_decay;
  double inertia_moment;
  double dash_power_rate;
  double player_size;
  double kickable_margin;
  double kick_rand;
  double extra_stamina;
  double effort_max;
  double effort_min;

  /* these values are calculated */
  double real_player_speed_max;
  double dash_to_keep_max_speed;
  double stamina_demand_per_meter;
  double speed_progress[SPEED_PROGRESS_MAX];
  double max_likelihood_max_speed_progress[SPEED_PROGRESS_MAX];
  double max_speed_progress[SPEED_PROGRESS_MAX];

  double stamina_10m;
  double stamina_20m;
  double stamina_30m;
  
  void init();
  bool parseMsg(const char*);
};

/** Contains all data sent with the server_param message by the server.
    WARNING: These are not all parameters that server.conf may contain!
*/

struct ServerParam {

  static double goal_width;
  static double inertia_moment;
  static double player_size;
  static double player_decay;
  static double player_rand;
  static double player_weight;
  static double player_speed_max;
  static double player_accel_max;
  static double stamina_max;
  static double stamina_inc_max;
  static double recover_init;
  static double recover_dec_thr;
  static double recover_min;
  static double recover_dec;
  static double effort_init;
  static double effort_dec_thr;
  static double effort_min;
  static double effort_dec;
  static double effort_inc_thr;
  static double effort_inc;
  static double kick_rand;
  static bool team_actuator_noise;
  static double prand_factor_l;
  static double prand_factor_r;
  static double kick_rand_factor_l;
  static double kick_rand_factor_r;
  static double ball_size;
  static double ball_decay;
  static double ball_rand;
  static double ball_weight;
  static double ball_speed_max;
  static double ball_accel_max;
  static double dash_power_rate;
  static double kick_power_rate;
  static double kickable_margin;
  static double control_radius;
  static double control_radius_width;
  static double maxpower;
  static double minpower;
  static double maxmoment;
  static double minmoment;
  static double maxneckmoment;
  static double minneckmoment;
  static double maxneckang;
  static double minneckang;
  static double visible_angle;
  static double visible_distance;
  static double wind_dir;
  static double wind_force;
  static double wind_ang;
  static double wind_rand;
  static double kickable_area;
  static double catchable_area_l;
  static double catchable_area_w;
  static double catch_probability;
  static int goalie_max_moves;
  static double ckick_margin;
  static double offside_active_area_size;
  static bool wind_none;
  static bool wind_random;
  static int say_coach_cnt_max;
  static int say_coach_msg_size;
  static int clang_win_size;
  static int clang_define_win;
  static int clang_meta_win;
  static int clang_advice_win;
  static int clang_info_win;
  static int clang_mess_delay;
  static int clang_mess_per_cycle;
  static int clang_del_win;
  static int clang_rule_win;
  static int freeform_send_period;
  static int freeform_wait_period;
  static int half_time;
  static int simulator_step;
  static int send_step;
  static int recv_step;
  static int sense_body_step;
  static int lcm_step;
  static int say_msg_size;
  static int hear_max;
  static int hear_inc;
  static int hear_decay;
  static int catch_ban_cycle;
  static int slow_down_factor;
  static bool use_offside;
  static bool forbid_kick_off_offside;
  static double offside_kick_margin;
  static double audio_cut_dist;
  static double quantize_step;
  static double quantize_step_l;
  static double quantize_step_dir;
  static double quantize_step_dist_team_l;
  static double quantize_step_dist_team_r;
  static double quantize_step_dist_l_team_l;
  static double quantize_step_dist_l_team_r;
  static double quantize_step_dir_team_l;
  static double quantize_step_dir_team_r;
  static bool coach;
  static bool coach_w_referee;
  static bool old_coach_hear;
  static int send_vi_step;
  static int start_goal_l;
  static int start_goal_r;
  static bool fullstate_l;
  static bool fullstate_r;
  static int drop_ball_time;
  static int port;
  static int coach_port;
  static int olcoach_port;
  static int verbose;
  static int replay;
  static int synch_mode;
  static int synch_offset;
  static int synch_micro_sleep;

  static int max_goal_kicks;
  static int point_to_ban;
  static int point_to_duration;
  static int tackle_cycles;
  static int back_passes;
  static int free_kick_faults;
  static int proper_goal_kicks;
  static int record_messages;
  static int send_comms;
  static double stopped_ball_vel;
  static double tackle_back_dist;
  static double tackle_dist;
  static double tackle_exponent;
  static double tackle_power_rate;
  static double tackle_width;

  static void init();
  static bool parseMsg(const char*);
};
  
  
  


#if 0
struct ServerParam {

  static double maxneckmoment;
  static double minneckmoment;
  static double maxneckang;
  static double minneckang;
  static double offside_kick_margin;

  static double goal_width;
  static double player_size;
  static double player_decay;
  static double player_rand;
  static double player_weight;
  static double player_speed_max;
  static double player_accel_max;
  static double stamina_max;
  static double stamina_inc_max;
  static double recover_dec_thr;
  static double recover_dec;
  static double recover_min;
  static double effort_dec_thr;
  static double effort_dec;
  static double effort_inc_thr;
  static double effort_inc;
  static double effort_min;
  static int hear_max;
  static int hear_inc;
  static int hear_decay;
  static double audio_cut_dist;
  static double inertia_moment;
  static double catchable_area_l;
  static double catchable_area_w;
  static double catch_probability;
  static int catch_ban_cycle;
  static int goalie_max_moves;
  static double ball_size;
  static double ball_decay;
  static double ball_rand;
  static double ball_weight;
  static double ball_speed_max;
  static double ball_accel_max;
  static double wind_force;
  static double wind_dir;
  static double wind_rand;
  static double kick_margin;
  static double kickable_margin;
  static double kick_rand;
  static double ckick_margin;
  static double corner_kick_margin;
  static double dash_power_rate;
  static double kick_power_rate;
  static double visible_angle;
  static double quantize_step;
  static double quantize_step_l;
  static double maxpower;
  static double minpower;
  static double maxmoment;
  static double minmoment;
  static int port;
  static int coach_port;
  static int simulator_step;
  static int send_step;
  static int recv_step;
  static int half_time;
  static int say_msg_size;
  static bool use_offside;
  static double offside_active_area_size;
  static bool forbid_kick_off_offside;
  static bool verbose;
  static int record_version;
  static bool record_log;
  static bool send_log;
  static int sense_body_step;
  static int say_coach_msg_size;
  static int say_coach_cnt_max;
  static int send_vi_step;
  static double server_port;
  static double kickable_area;

  /* these are not from server.conf... */
  static Vector own_goal_pos;
  static Vector their_goal_pos;
  static double pitch_length;
  static double pitch_width;
  static double penalty_area_width;
  static double penalty_area_length;
  static double ctlradius ;				/*< control radius */
  static double ctlradius_width ;			/*< (control radius) - (plyaer size) */
  static double maxn ;					/*< max neck angle */
  static double minn ;					/*< min neck angle */
  static double visible_distance ;			/*< visible distance */
  static int wind_no ;					/*< wind factor is none */
  static int wind_random ;				/*< wind factor is random */

  static int log_times;
  static int clang_win_size;
  static int clang_define_win;
  static int clang_meta_win;
  static int clang_advice_win;
  static int clang_info_win;
  static int clang_mess_delay;
  static int clang_mess_per_cycle;

  void init();
  bool readFromFile(const char *name);
  bool parseMsg(const char*);
};

#endif
#endif
