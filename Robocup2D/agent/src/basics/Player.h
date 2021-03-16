#ifndef BS2K_BASICS_PLAYER_H_
#define BS2K_BASICS_PLAYER_H_

#include <limits>

#include "globaldef.h"
#include "Vector.h"
#include "angle.h"

struct Player
{
    bool   alive;    // if alive == false all other members take deliberate values

    int    number;   // can be <= 0 if the number is unknown
    int    team;     // takes values in MY_TEAM, HIS_TEAM, UNKNOWN_TEAM
    int    type;     // heterogen player type

    int    time;     // time of last update of the position
    int    age;      // current time minus own time, just for user convenience!

    Vector pos;

    Vector vel;
    int    age_vel;  // time of last update of the velocity

    ANGLE  ang;
    int    age_ang;  // time of last update of the angle

    ANGLE  neck_ang; // neck_ang is  the absolute neck angle, NOT relative to player angle
    ANGLE  neck_ang_rel;


    double radius;   // radius of the player's body

    double speed_max;
    double dash_power_rate;
    double inertia_moment;
    double decay;
    double stamina;
    double effort;
    double recovery;
    double stamina_capacity;
    double stamina_capacity_bound; // this is the min remaining stamina as provided from the coach
    double stamina_inc_max;
    double stamina_demand_per_meter;

    double kick_radius; // kick radius is the max distance from the middle of the player, where the ball is still kickable
    bool   kick_flag;
    double kick_power_rate;
    double kick_rand_factor;

    double catchable_area_l_stretch;

    bool   pointto_flag;
    int    pointto_age; // pointto_age is only valid, if pointto_flag == true
    ANGLE  pointto_dir; // pointto_dir is only valid, if pointto_flag == true


    bool   tackle_flag; // true if the oppoent is tackling (it's a snapshop from time 'time')
    int    tackle_time; // recent time a tackling was perceived
    int    action_time; // recent time an action (non-tackling) was perceived

    bool   yellow_card;
    bool   red_card;

    bool   fouled;
    int    foul_cycles;

    double foul_detect_probability;


    int direct_opponent_conflict_number; // JTS10 conflict information as provided by the coach
    int direct_opponent_number; //number of direct opponent as assigned by coach


    struct PassInfo
    {
        bool   valid;
        int    age;      // indicates how old this message is!
        Vector ball_pos;
        Vector ball_vel;
        int    abs_time; // this is the absolute time when ball_pos and ball_vel will be valid!
        int    rel_time;
    } pass_info;

    //TGpr: begin
    struct PassRequest
    {
        bool valid;
        int  pass_in_n_steps;
        int  pass_param;
        int  pass_param_as_angle; //angle in degree (between -80 and 70)
        int  received_at;
    } pass_request;
    //TGpr: end


    Player();

    void set_direct_opponent_number( int nr );
};

std::ostream& operator<<( std::ostream &outStream, const Player &player );

#endif /* BS2K_BASICS_PLAYER_H_ */
