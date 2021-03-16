#ifndef _SERVER_OPTIONS_H_
#define _SERVER_OPTIONS_H_

#include "Vector.h"
#include "angle.h"

/* 
 don't edit this file unless YOU know what changes have to be done in other parts
 of the code, so that this options are set correctly!

 Otherwise important parts of the Agent initialization might be damaged.
 */
struct ServerOptions
{
    static double ball_accel_max;
    static double ball_decay;        // Decay rate of speed of the ball.
    static double ball_rand;         // Amount of noise added in the movements of the ball.
    static double ball_size;         // Radius of the ball.
    static double ball_speed_max;    // Maximum speed of the ball during one simulation cycle
    //static double ball_weight;       // Weight of the ball. This parameter concerns the wind factor
    static double catchable_area_l;  // Goalie catchable area length.
    static double catchable_area_w;  // Goalie catchable area width.
    static int    catch_ban_cycle;   // The number of cycles the goalie is banned from catching the ball after a successful catch.
    //static double catch_probability; // The probability for a goalie to catch the ball (if it is not during the catch ban static interval)
    static double dash_power_rate;   // Rate by which Power argument in dash command is multiplied.
    static int    drop_ball_time;    // Time a Player has to do a standard situation action, until the ball is getting dropped and the game goes on.
    static double effort_dec;        // Decrement step for player's effort capacity.
    static double effort_dec_thr;    // Decrement threshold for player's effort capacity.
    static double effort_inc;        // Increment step for player's effort capacity.
    static double effort_inc_thr;    // Increment threshold for player's effort capacity.
    static double effort_min;        // Minimum value for player's effort capacity.
    static bool   fullstate_l;       // Fullstate flag for left side.
    static bool   fullstate_r;       // Fullstate flag for right side.
    //static int    goalie_max_moves;
    static double goal_width;        // Width of the goal. For acquiring higher scores 14.02 was used in most cases
    static int    half_time;         // The length of a half time of a match. Unit is simulation cycle.
    static double inertia_moment;    // Inertia moment of a player. It affects it's moves
    static double kickable_area;     // The area within which the ball is kickable. kickable_area = kickable_margin + ball_size + player_size
    static double kickable_margin;
    static double kick_power_rate;   // Rate by which Power argument in kick command is multiplied.
    static ANGLE  maxneckang;
    static ANGLE  maxneckmoment;
    static double maxpower;          // Maximum value of Power argument in dash and kick commands.
    static ANGLE  minneckang;
    static ANGLE  minneckmoment;
    static double minpower;          // Minimum value of Power argument in dash and kick commands.
    //static double player_accel_max;
    static double player_decay;      // player decay
    static double player_rand;       // Amount of noise added in player's movements and turns.
    static double player_size;       // Radius of a player.
    static double player_speed_max;  // Maximum speed of a player during one simulation cycle
    //static double player_weight;     // Weight of a player. This parameter concerns the wind factor
    static double recover_dec;       // Decrement step for player's recovery.
    static double recover_dec_thr;   // Decrement threshold for player's recovery.
    static double recover_min;       // Minimum player recovery.
    static int    simulator_step;    // Length of period of simulation cycle.
    static int    send_step;
    static int    slow_down_factor;  // Slow down factor used by the soccer server.
    static double stamina_max;       // Maximum stamina of a player
    static double stamina_inc_max;   // Amount of stamina that a player gains in a simulation cycle.
    static bool   use_offside;       // Flag for using offside rule [on/off]
    static double visible_angle;     // Angle of view cone of a player in the standard view mode.
    static double visible_distance;  // visible distance
    static double tackle_dist;
    static double tackle_back_dist;
    static double tackle_width;
    static int    tackle_exponent;
    static int    tackle_cycles;
    static double tackle_power_rate;
    // new in order to comply with v12 protocol
    static double player_speed_max_min;
    static double max_tackle_power;
    static double max_back_tackle_power;
    static int    extra_stamina;
    static int    ball_stuck_area;
    static int    synch_see_offset;
    static bool   pen_allow_mult_kicks;
    static int    pen_nr_kicks;
    static int    pen_max_extra_kicks;
    static double pen_max_goalie_dist_x;
    static double pen_dist_x;
    //  JTS: new in order to comply with v13 protocol
    static double stamina_capacity;
    static double max_dash_angle;
    static double min_dash_angle;
    static double dash_angle_step;
    static double side_dash_rate;
    static double back_dash_rate;
    static double max_dash_power;
    static double min_dash_power;
    static int    extra_half_time;
    //  JTS 10: new for v14
    static double tackle_rand_factor;
    static double foul_detect_probability;
    static double foul_exponent;
    static int    foul_cycles;
    static int    golden_goal;

    /* these are not from server.conf... */
    static double penalty_area_width;  // the width of the penalty area.
    static double penalty_area_length; // the length of the penalty area.
    static double pitch_length;        // the length of the field.
    static double pitch_width;         // the width of the field.
    static Vector own_goal_pos;        // the position of our goal.
    static Vector their_goal_pos;      // the position of their goal.
    static Vector their_left_goal_corner;
    static Vector their_right_goal_corner;
};

#endif

