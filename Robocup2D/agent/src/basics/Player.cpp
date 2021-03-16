#include "Player.h"

Player::Player()
{
    alive            = false;

    number           = -1;
    team             = UNKNOWN_TEAM;
    type             = -1;

    time             = -1;
    age              = std::numeric_limits<int>::max();

    age_vel          = std::numeric_limits<int>::max();

    age_ang          = std::numeric_limits<int>::max();


    radius           = -1;

    speed_max        = -1;
    dash_power_rate  = -1;
    inertia_moment   = -1;
    decay            = -1;
    stamina          = -1;
    effort           = -1;
    recovery         = -1;
    stamina_capacity = -1;
    stamina_capacity_bound = -1;
    stamina_inc_max  = -1;
    stamina_demand_per_meter = -1;

    kick_radius      = -1;
    kick_flag        = false;
    kick_power_rate  = -1;
    kick_rand_factor = -1;

    catchable_area_l_stretch = -1;

    pointto_flag     = false;
    pointto_age      = std::numeric_limits<int>::min();


    tackle_flag      = false;
    tackle_time      = std::numeric_limits<int>::min();
    action_time      = std::numeric_limits<int>::min();

    yellow_card      = false;
    red_card         = false;

    fouled           = false;
    foul_cycles      = -1;

    foul_detect_probability =  0;


    direct_opponent_conflict_number = -1;
    direct_opponent_number          = -1;


    pass_info.valid    = false;
    pass_info.age      = std::numeric_limits<int>::max();
    pass_info.abs_time = -1;
    pass_info.rel_time = -1;

    pass_request.valid               = false;
    pass_request.pass_in_n_steps     = -1;
    pass_request.pass_param          = -1;
    pass_request.pass_param_as_angle =  0;
    pass_request.received_at         = std::numeric_limits<int>::min();
}

void Player::set_direct_opponent_number( int nr )
{
    direct_opponent_number = nr;
}

std::ostream& operator<<( std::ostream &outStream, const Player &player )
{
    outStream << "\nPlayer= " << player.number;

    if( !player.alive )
    {
        outStream << " NOT alive";
    }
    else
    {
        outStream
        << " age= "         << player.age
        << " time= "        << player.time
        << ", Pos= "        << player.pos
        << ", angle= "      << player.ang
        << ", neck_angle= " << player.neck_ang
        << ", vel= "        << player.vel
        << ", stamina= "    << player.stamina;
    }
    return outStream;
}
