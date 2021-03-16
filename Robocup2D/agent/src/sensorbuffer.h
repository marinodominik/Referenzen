#ifndef _SENSORBUFFER_H_
#define _SENSORBUFFER_H_

#include <iostream>

#include "../../lib/src/Vector.h"

#include "basics/globaldef.h"
#include "basics/comm_msg.h"
#include "basics/wmdef.h"

//////////////////////////////////////////////////
/** This class can contain am see message       */
struct Msg_see
{
    struct _see_marker
    {
        _see_marker()
        {
            see_position = false;
            x = 0.0;
            y = 0.0;
            how_many = -1;
            dist = -1.0;
            dir = 0.0;
            dist_change = 0.0;
            dir_change = 0.0;
        }

        /** in special cases as (Goal) or (Flag) there is no information about the Goal or Flag id,
        in such a situation see_position is set to false
        */
        bool see_position;
        /** the marker's x coordinate. */
        double x;
        /** the marker's y coordinate. */
        double y;
        /** information amount: gives the number of following entries which have valid values
        so e.g. 2 means that dist and dir have valid values, but dir_change and dist_change not*/
        int how_many;
        /** distance to the player*/
        double dist;
        /** direction from the player*/
        double dir;
        /** change of distance. */
        double dist_change;
        /** change of direction. */
        double dir_change;
    };

    typedef  _see_marker _see_line;

    struct _see_player
    {
        _see_player()
        {
            team         =  unknown_TEAM;
            number       =  0;
            goalie       =  false;
            tackle_flag  =  false;
            kick_flag    =  false;
            pointto_flag =  false;
            pointto_dir  =  0.0;
            how_many     = -1;
            dist         = -1.0;
            dir          =  0.0;
            dist_change  =  0.0;
            dir_change   =  0.0;
            body_dir     =  0.0;
            head_dir     =  0.0;
        }

        /** which values for team: my_TEAM, his_TEAM, unknown_TEAM */
        int team;
        /** Uniform number: 0 means unknown number */
        int number;
        /** is seen as goalie */
        bool goalie;
        /** tackle flag is set for this player */
        bool tackle_flag;
        /** kick flag is set for this player */
        bool kick_flag;
        /** pointto is set by player */
        bool pointto_flag;
        /** if pointto_flag is true, here you can find the corresponding dir (distance is not propageted in ver. 8.04) */
        double pointto_dir;
        /** information amount */
        int how_many;
        /** distance to the player*/
        double dist;
        /** direction from the player*/
        double dir;
        /** change of distance.*/
        double dist_change;
        /** change of direction.*/
        double dir_change;
        /** direction of the body. */
        double body_dir;
        /** direction of the head. */
        double head_dir;
    };

    struct _see_ball
    {
        _see_ball()
        {
            how_many    = -1;
            dist        = -1.0;
            dir         =  0.0;
            dist_change =  0.0;
            dir_change  =  0.0;
        }

        /** information amount */
        int how_many;
        /** distance to the player*/
        double dist;
        /** direction from the player*/
        double dir;
        /** change of distance.*/
        double dist_change;
        /** change of direction.*/
        double dir_change;
    };

    Msg_see()
    {
        reset();
    }

    void reset()
    {
        time        = -1;
        markers_num =  0;
        players_num =  0;
        line_upd    =  false;
        ball_upd    =  false;
    }

    int time;
    static const int markers_MAX = 60;
    static const int players_MAX = 22;
    int markers_num;  //how many markers were updated
    int players_num;  //how many markers were updated
    bool line_upd;  //was there an update ot the line field
    bool ball_upd;  //was there an update ot the line field

    _see_marker markers[ markers_MAX ];
    _see_player players[ players_MAX ];
    _see_line   line;
    _see_ball   ball;
};

struct NOT_NEEDED{};

struct Msg_server_param
{
    double     goal_width;
    double     player_size;
    double     player_decay;
    double     player_rand;
    double     player_weight;
    double     player_speed_max;
    double     player_accel_max;
    double     stamina_max;
    double     stamina_inc_max;
    double     recover_init;
    double     recover_dec_thr;
    double     recover_min;
    double     recover_dec;
    double     effort_init;
    double     effort_dec_thr;
    double     effort_min;
    double     effort_dec;
    double     effort_inc_thr;
    double     effort_inc;
    double     kick_rand;
    int        team_actuator_noise;
    double     prand_factor_l;
    double     prand_factor_r;
    double     kick_rand_factor_l;
    double     kick_rand_factor_r;
    double     ball_size;
    double     ball_decay;
    double     ball_rand;
    double     ball_weight;
    double     ball_speed_max;
    double     ball_accel_max;
    double     dash_power_rate;
    double     kick_power_rate;
    double     kickable_margin;
    double     control_radius;
    double     control_radius_width;
    double     catch_probability;
    double     catchable_area_l;
    double     catchable_area_w;
    int        goalie_max_moves;
    double     maxpower;
    double     minpower;
    double     maxmoment;
    double     minmoment;
    double     maxneckmoment;
    double     minneckmoment;
    double     maxneckang;
    double     minneckang;
    double     visible_angle;
    double     visible_distance;
    double     audio_cut_dist;
    double     quantize_step;
    double     quantize_step_l;
    double     quantize_step_dir;
    double     quantize_step_dist_team_l;
    double     quantize_step_dist_team_r;
    double     quantize_step_dist_l_team_l;
    double     quantize_step_dist_l_team_r;
    double     quantize_step_dir_team_l;
    double     quantize_step_dir_team_r;
    double     ckick_margin;
    double     wind_dir;
    double     wind_force;
    double     wind_ang;
    double     wind_rand;
    double     kickable_area;
    double     inertia_moment;
    int        wind_none;
    int        wind_random;
    int        half_time;
    int        drop_ball_time;
    int        port;
    int        coach_port;
    int        olcoach_port;
    int        say_coach_cnt_max;
    int        say_coach_msg_size;
    int        simulator_step;
    int        send_step;
    int        recv_step;
    int        sense_body_step;
    int        lcm_step;
    int        say_msg_size;
    int        clang_win_size;
    int        clang_define_win;
    int        clang_meta_win;
    int        clang_advice_win;
    int        clang_info_win;
    int        clang_mess_delay;
    int        clang_mess_per_cycle;
    int        hear_max;
    int        hear_inc;
    int        hear_decay;
    int        catch_ban_cycle;
    int        coach;
    int        coach_w_referee;
    int        old_coach_hear;
    int        send_vi_step;
    int        use_offside;
    double     offside_active_area_size;
    int        forbid_kick_off_offside;
    int        verbose;
    NOT_NEEDED replay;
    double     offside_kick_margin;
    int        slow_down_factor;
    int        synch_mode;
    int        synch_offset;
    int        synch_micro_sleep;
    int        start_goal_l;
    int        start_goal_r;
    bool       fullstate_l;
    bool       fullstate_r;
    double     slowness_on_top_for_left_team;
    double     slowness_on_top_for_right_team;
    int        send_comms;
    int        text_logging;
    int        game_logging;
    int        game_log_version;
    NOT_NEEDED text_log_dir;
    NOT_NEEDED game_log_dir;
    NOT_NEEDED text_log_fixed_name;
    NOT_NEEDED game_log_fixed_name;
    int        text_log_fixed;
    int        game_log_fixed;
    int        text_log_dated;
    int        game_log_dated;
    NOT_NEEDED log_date_format;
    int        log_times;
    int        record_messages;
    int        text_log_compression;
    int        game_log_compression;
    int        profile;
    int        point_to_ban;
    int        point_to_duration;

    //new since version 8.05
    NOT_NEEDED freeform_send_period;
    NOT_NEEDED freeform_wait_period;
    NOT_NEEDED max_goal_kicks;
    int        tackle_cycles;
    NOT_NEEDED landmark_file;
    NOT_NEEDED back_passes;
    NOT_NEEDED free_kick_faults;
    NOT_NEEDED proper_goal_kicks;
    NOT_NEEDED stopped_ball_vel;
    double     tackle_back_dist;
    double     tackle_dist;
    int        tackle_exponent;
    double     tackle_power_rate;
    double     tackle_width;
    NOT_NEEDED clang_del_win;
    NOT_NEEDED clang_rule_win;

    // new in order to comply with v12 server
    NOT_NEEDED kick_off_wait;
    NOT_NEEDED keepaway;
    NOT_NEEDED keepaway_width;
    NOT_NEEDED keepaway_start;
    NOT_NEEDED keepaway_length;
    NOT_NEEDED keepaway_logging;
    NOT_NEEDED keepaway_log_fixed;
    NOT_NEEDED keepaway_log_dated;
    NOT_NEEDED keepaway_log_fixed_name;
    NOT_NEEDED keepaway_log_dir;
    NOT_NEEDED game_over_wait;
    NOT_NEEDED connect_wait;
    NOT_NEEDED pen_coach_moves_players;
    NOT_NEEDED pen_before_setup_wait;
    NOT_NEEDED penalty_shoot_outs;
    NOT_NEEDED pen_setup_wait;
    NOT_NEEDED pen_taken_wait;
    NOT_NEEDED pen_ready_wait;
    NOT_NEEDED pen_random_winner;
    NOT_NEEDED team_r_start;
    NOT_NEEDED team_l_start;
    NOT_NEEDED auto_mode;
    NOT_NEEDED nr_extra_halfs;
    NOT_NEEDED nr_normal_halfs;
    int        pen_allow_mult_kicks;
    int        pen_nr_kicks;
    int        pen_max_extra_kicks;
    double     pen_max_goalie_dist_x;
    double     pen_dist_x;
    int        synch_see_offset;
    double     player_speed_max_min;
    double     max_tackle_power;
    double     max_back_tackle_power;
    int        extra_stamina;
    int        ball_stuck_area;

    // JTS: new in order to comply with v13 server
    double    stamina_capacity;
    double    max_dash_angle;
    double    min_dash_angle;
    double    dash_angle_step;
    double    side_dash_rate;
    double    back_dash_rate;
    double    max_dash_power;
    double    min_dash_power;
    int       extra_half_time;

    // JTS 10: new for server v14
    double    tackle_rand_factor;
    double    foul_detect_probability;
    double    foul_exponent;
    int       foul_cycles;
    int       golden_goal;
};

struct Msg_player_param
{
    int        player_types;
    int        subs_max;
    int        pt_max;
    double     player_speed_max_delta_min;
    double     player_speed_max_delta_max;
    double     stamina_inc_max_delta_factor;
    double     player_decay_delta_min;
    double     player_decay_delta_max;
    double     inertia_moment_delta_factor;
    double     dash_power_rate_delta_min;
    double     dash_power_rate_delta_max;
    double     player_size_delta_factor;
    double     kickable_margin_delta_min;
    double     kickable_margin_delta_max ;
    double     kick_rand_delta_factor;
    double     extra_stamina_delta_min;
    double     extra_stamina_delta_max;
    double     effort_max_delta_factor;
    double     effort_min_delta_factor;
    long       random_seed;

    NOT_NEEDED new_dash_power_rate_delta_min;
    NOT_NEEDED new_dash_power_rate_delta_max;
    NOT_NEEDED new_stamina_inc_max_delta_factor;

    // JTS new in order to comply with v12 server protocol
    int        allow_mult_default_type;

    // JTS 10: new for server v14
    double     kick_power_rate_delta_min;
    double     kick_power_rate_delta_max;
    double     foul_detect_probability_delta_factor;
    double     catchable_area_l_stretch_min;
    double     catchable_area_l_stretch_max;
};

struct Msg_player_type
{
    int    id;
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
    // JTS 10: heterogenous goalie and foul
    double catchable_area_l_stretch;
    double foul_detect_probability;
    double kick_power_rate;

    double real_player_speed_max; //this value is not parsed, but computed!!!
    double stamina_demand_per_meter; //this value is valid if the player runs with his max speed (and dash is reduced if possible)

    void adapt_better_entries( Msg_player_type const &t )
    {
        if( player_speed_max         < t.player_speed_max         ) player_speed_max         = t.player_speed_max;
        if( stamina_inc_max          < t.stamina_inc_max          ) stamina_inc_max          = t.stamina_inc_max;
        if( player_decay             < t.player_decay             ) player_decay             = t.player_decay;
        if( inertia_moment           > t.inertia_moment           ) inertia_moment           = t.inertia_moment;
        if( dash_power_rate          < t.dash_power_rate          ) dash_power_rate          = t.dash_power_rate;
        if( kickable_margin          < t.kickable_margin          ) kickable_margin          = t.kickable_margin;
        if( kick_rand                > t.kick_rand                ) kick_rand                = t.kick_rand;
        if( extra_stamina            < t.extra_stamina            ) extra_stamina            = t.extra_stamina;
        if( effort_max               < t.effort_max               ) effort_max               = t.effort_max;
        if( effort_min               > t.effort_min               ) effort_min               = t.effort_min;
        if( real_player_speed_max    < t.real_player_speed_max    ) real_player_speed_max    = t.real_player_speed_max;
        if( stamina_demand_per_meter > t.stamina_demand_per_meter ) stamina_demand_per_meter = t.stamina_demand_per_meter;
    }

    struct StaminaPos
    {
        StaminaPos()
        {
            dist    = 0;
            stamina = 0;
        }

        double dist;
        double stamina;
    };

    void show( std::ostream &outStream )
    {
        double speed_progress[ 5 ];
        speed_progress[ 0 ] = 100.0 * ( dash_power_rate * effort_max );
        for( int i = 1; i < 5; i++ )
        {
            speed_progress[ i ]  = speed_progress[ i - 1 ] * player_decay;
            speed_progress[ i ] += 100.0 * ( dash_power_rate * effort_max );
            if( speed_progress[ i ] > player_speed_max ) speed_progress[ i ] = player_speed_max;
        }

        const int max = 40;

        StaminaPos demand[ max ];
        double player_speed = 0;
        for( int i = 1; i < max; i++ )
        {
            player_speed *= player_decay;
            double dash_power = 100.0;
            double dash_to_keep_max_speed= ( player_speed_max - player_speed ) / (dash_power_rate*effort_max);

            if ( dash_to_keep_max_speed < dash_power ) dash_power= dash_to_keep_max_speed;

            player_speed += dash_power * ( dash_power_rate * effort_max );

            demand[ i].dist    += demand[ i - 1 ].dist    + player_speed;
            demand[ i].stamina += demand[ i - 1 ].stamina + dash_power - stamina_inc_max;
        }

        double dash_to_keep_max_speed = ( real_player_speed_max - real_player_speed_max * player_decay ) / ( dash_power_rate * effort_max );

        outStream << "\n--------------------------------------------------------------"
                << "\ntype                       " << id
                << "\n dash_power_rate=          " << dash_power_rate
                << "\n player_decay=             " << player_decay
                << "\n kick_rand=                " << kick_rand
                << "\n kickable_area             " << player_size+kickable_margin << " (player_size= " << player_size << ")"
                << "\n--"
                << "\n real_player_speed_max=    " << real_player_speed_max << " (player_speed_max= " << player_speed_max << ")"
                << "\n dash_to_keep_max_speed=   " << dash_to_keep_max_speed
                << "\n stamina_inc_max=          " << stamina_inc_max
                << "\n effort_min=               " << effort_min
                << "\n effort_max=               " << effort_max
                //<< "\n stamina_demand_per_cycle= " << stamina_demand_per_meter * real_player_speed_max
                << "\n stamina_demand_per_meter= " << stamina_demand_per_meter
                << "\n--"
                << "\n speed_progression=        " << speed_progress[ 0 ] << " , " << speed_progress[ 1 ] << " , " << speed_progress[ 2 ] << " , " << speed_progress[ 3 ] << " , " << speed_progress[ 4 ]
                << "\n";

//        for( int i = 1; i < max; i++ )
//        {
//            outStream << "(" << i << ": " << demand[ i ].dist << ", " << demand[ i ].stamina / demand[ i ].dist << ")";
//            if( i % 4 == 0) out << "\n";
//        }

        for( int i = 1; i < max; i++ )
        {
            if( ( demand[ i - 1 ].dist < 3.0 && demand[ i ].dist >= 3.0 ) || ( int( demand[ i - 1 ].dist / 5.0 ) < int( demand[ i ].dist / 5.0 ) ) )
                outStream << "\n(" << i << ": " << demand[ i ].dist << ", " << demand[ i ].stamina / demand[ i ].dist << ")";
        }
    }
};

struct Msg_change_player_type
{
    Msg_change_player_type()
    {
        reset();
    }

    void reset()
    {
        number =  0;
        type   = -1;
    }

    int number;
    int type; //-1 if no type was specified
};

//////////////////////////////////////////////////

/** this struct provides fields needed in communication between
    team members (server version 8.04 allowes 10 character of communication)

    up to 3 objects are possible (object in {player,ball})
 */   
struct Msg_teamcomm2
{
    Msg_teamcomm2()
    {
        reset();
    }

    void reset()
    {
        players_num                           =  0;
        msg.valid                             =  false;
        ball.valid                            =  false;
        pass_info.valid                       =  false;
        ball_info.valid                       =  false;
        ball_holder_info.valid                =  false;
        direct_opponent_assignment_info.valid =  false; //TGdoa
        pass_request_info.valid               =  false; //TGpr
        from                                  = -1;
    }

    int get_num_objects() const
    {
        int num = 0;

        if( msg.valid                             ) num++;
        if( ball.valid                            ) num++;
        if( pass_info.valid                       ) num++;
        if( ball_info.valid                       ) num++;
        if( ball_holder_info.valid                ) num++;
        if( direct_opponent_assignment_info.valid ) num++; //TGdoa
        if( pass_request_info.valid               ) num++; //TGpr

        return players_num + num;
    }

    SayMsg msg;

    struct _player
    {
        int number;
        int team;
        Vector pos;
    };

    struct
    {
        bool valid;
        Vector ball_pos;
        Vector ball_vel;
        int time; //must be in [0,...,127], it is the relative time
    } pass_info;

    struct
    {
        bool valid;
        Vector ball_pos;
        Vector ball_vel;
        int age_pos; //must be in [0,...,3], it is the relative age
        int age_vel; //must be in [0,...,3], it is the relative age
    } ball_info;

    struct
    {
        bool valid;
        Vector pos;
    } ball_holder_info;

    //TGdoa: begin
    struct
    {
        bool valid;
        int  assignment;
    } direct_opponent_assignment_info;
    //TGdoa: end

    //TGpr: begin
    struct
    {
        bool valid;
        int  pass_in_n_steps; //must be in [0,...,15], it is a relative time
        int  pass_param;      //must be in [0,...,15] (4bits)
    } pass_request_info;
    //TGpr: end

    static const int players_MAX = 5;

    int players_num;
    _player players[ players_MAX ];

    struct
    {
        bool valid;
        Vector pos;
    } ball;

    int from; // -1 indicates that the number is not known
};

/** this struct provides all the fields needed in communication between
    team members.
 */
struct Msg_teamcomm
{
    Msg_teamcomm()
    {
        reset();
    }

    void reset()
    {
        side                  =  unknown_SIDE;
        time                  = -1;
        time_cycle            =  0;
        from                  = -1;
        players_num           =  0;
        ball_upd              =  false;
        his_goalie_number_upd =  false;
    }

    //side of sender/receiver (you need it to play against a team using the same communication)
    int side;

    int time;
    int time_cycle;

    //number of sender/receiver range [0,...,11]
    int from;

    struct _tc_player
    {
        int how_old;
        int team;
        int number;
        double x;
        double y;
    };

    struct _tc_ball
    {
        int how_old;
        double x;
        double y;
        double vel_x;
        double vel_y;
    };

    static const int players_MAX = 25;

    ///if his_goalie_number_upd==false, then his_goalie_number is undefined
    bool his_goalie_number_upd;
    int his_goalie_number;

    ///if ball_upd==false, then the ball is undefined
    bool ball_upd;
    _tc_ball ball;

    int players_num;
    _tc_player players[ players_MAX ];
};

struct Msg_my_online_coachcomm
{
    Msg_my_online_coachcomm()
    {
        reset();
    }

    void reset()
    {
        time                           = -1;
        his_player_types_upd           =  false;
        his_goalie_number_upd          =  false;
        direct_opponent_assignment_upd =  false;
        stamin_capacity_info_upd       =  false;
    }

    int time;
    int his_player_types[ NUM_PLAYERS ];
    bool his_player_types_upd;

    bool his_goalie_number_upd;
    int his_goalie_number;

    bool direct_opponent_assignment_upd;
    int  direct_opponent_assignment[ NUM_PLAYERS ];
    int  direct_opponent_conflict[ NUM_PLAYERS ];

    bool stamin_capacity_info_upd;
    int  stamin_capacity_info[ NUM_PLAYERS ];
};

struct Msg_card
{
    Card type;
    int card_player;
    int side;
    int time;

    Msg_card()
    {
        reset();
    }

    void reset()
    {
        type        =  NO_CARD;
        side        = -1;
        card_player = -1;
        time        = -1;
    }
};

/** This class can contain am hear message       */
struct Msg_hear
{
    Msg_hear()
    {
        reset();
    }

    void reset()
    {
        time                    = -1;
        play_mode_upd           =  false;
        my_score_upd            =  false;
        his_score_upd           =  false;
        teamcomm_upd            =  false;
        teamcomm_partial_upd    =  false;
        my_online_coachcomm_upd =  false;
        card_update             =  false;
    }

    int time;
    PlayMode play_mode;
    bool     play_mode_upd;
    int      my_score;
    bool     my_score_upd;
    int      his_score;
    bool     his_score_upd;
    //Comm   teamcomm;
    Msg_teamcomm2 teamcomm;  // 2 -> new teamcomm type (10 characters since server ver. 8.04)
    bool     teamcomm_upd;
    bool     teamcomm_partial_upd;
    // JTS 10: new Msg to handle yellow / red cards
    bool     card_update;
    Msg_card card;

    Msg_my_online_coachcomm my_online_coachcomm;
    bool my_online_coachcomm_upd;
};

//////////////////////////////////////////////////
/** This struct can contain an init message       */
struct Msg_init
{
    Msg_init()
    {
        reset();
    }

    void reset()
    {
        side      = -9;
        number    = -1;
        play_mode =  PM_Null;
    }

    int side;
    int number;
    PlayMode play_mode;
};

struct Msg_fullstate
{
    Msg_fullstate()
    {
        reset();
    }

    void reset()
    {
        time = -1;
        players_num = 0;
    }

    static const int players_MAX = 22;
    int players_num;

    struct _fs_player
    {
        int    team;
        int    number;
        double x;
        double y;
        double vel_x;
        double vel_y;
        double angle;
        double neck_angle;
        double stamina;
        double effort;
        double recovery;
    };

    struct _fs_ball
    {
        double x;
        double y;
        double vel_x;
        double vel_y;
    };

    int time;
    _fs_player players[ players_MAX ];
    _fs_ball ball;
    PlayMode play_mode;
    int view_quality;
    int view_width;
    int my_score;
    int his_score;
};

struct Msg_fullstate_v8
{
    Msg_fullstate_v8()
    {
        reset();
    }

    void reset()
    {
        time = -1;
        players_num = 0;
    }

    static const int players_MAX = 22;
    int players_num;

    struct _fs_player
    {
        int team;
        int number;
        int type;
        bool goalie;

        double x;
        double y;
        double vel_x;
        double vel_y;
        double angle;
        double neck_angle;
        double stamina;
        double effort;
        double recovery;
        double stamina_capacity;
        double point_dist;//TG14
        double point_dir;//TG14
        bool   kick_flag;//TG14
        bool   tackle_flag;//TG14
        //JTS10
        Card card;
        bool fouled;
        int foul_cycles;
    };

    struct _fs_ball
    {
        double x;
        double y;
        double vel_x;
        double vel_y;
    };

    int time;
    _fs_player players[ players_MAX ];
    _fs_ball ball;
    PlayMode play_mode;
    int view_quality;
    int view_width;
    int my_score;
    int his_score;

    int count_kick;
    int count_dash;
    int count_turn;
    int count_catch;
    int count_move;
    int count_turn_neck;
    int count_change_view;
    int count_say;
};

//////////////////////////////////////////////////
/** Sense-Body information. */
class Msg_sense_body
{
public:
    Msg_sense_body()
    {
        reset();
    }

    void reset()
    {

    }

    /** timestep of last update. */
    int time;
    int view_quality;
    int view_width;
    double stamina;
    double stamina_capacity;
    double effort;
    double speed_value;
    double speed_angle;
    double neck_angle;
    int kick_count;
    int dash_count;
    int turn_count;
    int say_count;
    int turn_neck_count;
    int catch_count;
    int move_count;
    int change_view_count;

    int arm_movable;
    int arm_expires;
    double arm_target_x;
    double arm_target_y;
    int arm_count;

    int focus_target; // -1 == none
    int focus_count;

    int tackle_expires;
    int tackle_count;
    /** collision information **/
    int coll_ball;
    int coll_player;
    int coll_post;
    //JTS10 foul model
    bool fouled;
    int foul_cycles;
    Card card;
};



std::ostream& operator<<( std::ostream &outStream, const Msg_sense_body          &sb        );
std::ostream& operator<<( std::ostream &outStream, const Msg_see                 &see       );
std::ostream& operator<<( std::ostream &outStream, const Msg_hear                &hear      );
std::ostream& operator<<( std::ostream &outStream, const Msg_init                &init      );
std::ostream& operator<<( std::ostream &outStream, const Msg_fullstate           &fullstate );
std::ostream& operator<<( std::ostream &outStream, const Msg_teamcomm            &teamcomm  );
std::ostream& operator<<( std::ostream &outStream, const Msg_teamcomm2           &teamcomm  );
std::ostream& operator<<( std::ostream &outStream, const Msg_my_online_coachcomm &moc       );
#endif
