#include "serverparam.h"

Msg_server_param  *ServerParam::server_param = 0;
Msg_player_param  *ServerParam::player_param = 0;
Msg_player_type  **ServerParam::player_types = 0;
Msg_player_type    ServerParam::worst_case_opponent_type;

bool ServerParam::incorporate_server_param_string( const char *buf )
{
    if( server_param )
    {
        ERROR_OUT << ID << "\nstrange, server_param should be == 0";
        return false;
    }

    server_param = new Msg_server_param();

    return SensorParser::manual_parse_server_param( buf, *server_param );
}

bool ServerParam::incorporate_player_param_string( const char *buf )
{
    if( player_param )
    {
        ERROR_OUT << ID << "\nstrange, player_param should be == 0";

        return false;
    }

    player_param = new Msg_player_param();

    if( !SensorParser::manual_parse_player_param( buf, *player_param ) )
        return false;

    if( player_types )
    {
        ERROR_OUT << ID << "\nstrange, player_types should be == 0";

        player_types = 0;

        return false;
    }

    player_types = new Msg_player_type*[ player_param->player_types ];

    for( int i = 0; i < player_param->player_types; i++ )
        player_types[ i ] = 0;

    return true;
}

bool ServerParam::incorporate_player_type_string( const char *buf )
{
    if( !player_types )
    {
        ERROR_OUT << ID
                << "\nstrange, player_types should be != 0, ignoring player types";

        return false;
    }

    Msg_player_type *pt = new Msg_player_type();

    if( !SensorParser::manual_parse_player_type( buf, *pt ) )
        return false;

    if( player_types[ pt->id ] )
    {
        ERROR_OUT << ID << "\nstrange player type " << pt->id
                << " already exists";

        return false;
    }

    double speed = 0;
    double old_speed = 0;

    do
    {
        old_speed = speed;

        speed *= pt->player_decay;
        speed += 100 * ( pt->dash_power_rate * pt->effort_max );

        if( speed > pt->player_speed_max )
        {
            speed = pt->player_speed_max;

            break;
        }
    } while( speed - old_speed >= 0.001 );

    pt->real_player_speed_max = speed * 1.01; // 1 percent tolerance!

    if( pt->real_player_speed_max > pt->player_speed_max )
        pt->real_player_speed_max = pt->player_speed_max;

    /* experimental
     if ( ! player_param )
     ERROR_OUT << ID << " player param should be initialized at this moment!";
     else
     pt->real_player_speed_max += player_param->player_speed_max_delta_max;
     */

    double dash_to_keep_max_speed = ( pt->real_player_speed_max - pt->real_player_speed_max * pt->player_decay ) / ( pt->dash_power_rate * pt->effort_max );

    if( dash_to_keep_max_speed > 100.0 )
    {
        dash_to_keep_max_speed = 100.0;
    }

    pt->stamina_demand_per_meter = ( dash_to_keep_max_speed - pt->stamina_inc_max ) / pt->real_player_speed_max;

    player_types[ pt->id ] = pt;

    if( pt->id == 0 )
    {
        worst_case_opponent_type    = *pt;
        worst_case_opponent_type.id = -1;
    }
    else
    {
        worst_case_opponent_type.adapt_better_entries( *pt );
    }

    if( pt->id + 1 == player_param->player_types )
    {
        INFO_OUT << ID << "Player types: ";
        for( int i = 0; i < player_param->player_types; i++ )
            player_types[ i ]->show(   INFO_STREAM );
        worst_case_opponent_type.show( INFO_STREAM );
    }

    return true;
}

bool ServerParam::all_params_ok()
{
    if( !server_param )
        return false;

    if( !player_param )
        return false;

    for( int i = 0; i < player_param->player_types; i++ )
        if( !player_types[ i ] )
            return false;

    return true;
}

bool ServerParam::export_server_options()
{
    if( !server_param )
        return false;

    ServerOptions::ball_accel_max    = server_param->ball_accel_max;
    ServerOptions::ball_decay        = server_param->ball_decay;
    ServerOptions::ball_rand         = server_param->ball_rand;
    ServerOptions::ball_size         = server_param->ball_size;
    ServerOptions::ball_speed_max    = server_param->ball_speed_max;
    //ServerOptions::ball_weight       = server_param->ball_weight;
    ServerOptions::catchable_area_l  = server_param->catchable_area_l;
    ServerOptions::catchable_area_w  = server_param->catchable_area_w;
    ServerOptions::catch_ban_cycle   = server_param->catch_ban_cycle;
    //ServerOptions::catch_probability = server_param->catch_probability;
    ServerOptions::dash_power_rate   = server_param->dash_power_rate;
    ServerOptions::drop_ball_time    = server_param->drop_ball_time;
    ServerOptions::effort_dec        = server_param->effort_dec;
    ServerOptions::effort_dec_thr    = server_param->effort_dec_thr;
    ServerOptions::effort_inc        = server_param->effort_inc;
    ServerOptions::effort_inc_thr    = server_param->effort_inc_thr;
    ServerOptions::effort_min        = server_param->effort_min;
    ServerOptions::fullstate_l       = server_param->fullstate_l;
    ServerOptions::fullstate_r       = server_param->fullstate_r;
    //ServerOptions::goalie_max_moves  = server_param->goalie_max_moves;
    ServerOptions::goal_width        = server_param->goal_width;
    ServerOptions::half_time         = server_param->half_time;
    ServerOptions::inertia_moment    = server_param->inertia_moment;
    //kickable_area is defined at the end
    ServerOptions::kickable_margin   = server_param->kickable_margin;
    ServerOptions::kick_power_rate   = server_param->kick_power_rate;
    ServerOptions::maxneckang        = ANGLE( DEG2RAD( server_param->maxneckang ) );
    ServerOptions::maxneckmoment     = ANGLE( DEG2RAD( server_param->maxneckmoment ) );
    ServerOptions::maxpower          = server_param->maxpower;
    ServerOptions::minneckang        = ANGLE( DEG2RAD( server_param->minneckang ) );
    ServerOptions::minneckmoment     = ANGLE( DEG2RAD( server_param->minneckmoment ) );
    ServerOptions::minpower          = server_param->minpower;
    //ServerOptions::player_accel_max  = server_param->player_accel_max;
    ServerOptions::player_decay      = server_param->player_decay;
    ServerOptions::player_rand       = server_param->player_rand;
    ServerOptions::player_size       = server_param->player_size;
    ServerOptions::player_speed_max  = server_param->player_speed_max;
    //ServerOptions::player_weight     = server_param->player_weight;
    ServerOptions::recover_dec       = server_param->recover_dec;
    ServerOptions::recover_dec_thr   = server_param->recover_dec_thr;
    ServerOptions::recover_min       = server_param->recover_min;
    ServerOptions::simulator_step    = server_param->simulator_step;
    ServerOptions::send_step         = server_param->send_step;
    ServerOptions::slow_down_factor  = server_param->slow_down_factor;
    ServerOptions::stamina_max       = server_param->stamina_max;
    ServerOptions::stamina_inc_max   = server_param->stamina_inc_max;
    ServerOptions::use_offside       = server_param->use_offside;
    ServerOptions::visible_angle     = server_param->visible_angle;
    ServerOptions::visible_distance  = server_param->visible_distance;
    ServerOptions::tackle_dist       = server_param->tackle_dist;
    ServerOptions::tackle_back_dist  = server_param->tackle_back_dist;
    ServerOptions::tackle_width      = server_param->tackle_width;
    ServerOptions::tackle_exponent   = server_param->tackle_exponent;
    ServerOptions::tackle_cycles     = server_param->tackle_cycles;

    // new Options for v12 protocol
    ServerOptions::player_speed_max_min  = server_param->player_speed_max_min;
    ServerOptions::max_tackle_power      = server_param->max_tackle_power;
    ServerOptions::max_back_tackle_power = server_param->max_back_tackle_power;
    ServerOptions::extra_stamina         = server_param->extra_stamina;
    ServerOptions::ball_stuck_area       = server_param->ball_stuck_area;
    ServerOptions::synch_see_offset      = server_param->synch_see_offset;
    ServerOptions::pen_allow_mult_kicks  = server_param->pen_allow_mult_kicks;
    ServerOptions::pen_nr_kicks          = server_param->pen_nr_kicks;
    ServerOptions::pen_max_extra_kicks   = server_param->pen_max_extra_kicks;
    ServerOptions::pen_max_goalie_dist_x = server_param->pen_max_goalie_dist_x;
    ServerOptions::pen_dist_x            = server_param->pen_dist_x;

    // JTS: new Options for v13 protocol
    ServerOptions::stamina_capacity = server_param->stamina_capacity;
    ServerOptions::max_dash_angle   = server_param->max_dash_angle;
    ServerOptions::min_dash_angle   = server_param->min_dash_angle;
    ServerOptions::dash_angle_step  = server_param->dash_angle_step;
    ServerOptions::side_dash_rate   = server_param->side_dash_rate;
    ServerOptions::back_dash_rate   = server_param->back_dash_rate;
    ServerOptions::max_dash_power   = server_param->max_dash_power;
    ServerOptions::min_dash_power   = server_param->min_dash_power;
    ServerOptions::extra_half_time  = server_param->extra_half_time;

    // JTS10: new  v14 options
    ServerOptions::tackle_rand_factor      = server_param->tackle_rand_factor;
    ServerOptions::foul_detect_probability = server_param->foul_detect_probability;
    ServerOptions::foul_exponent           = server_param->foul_exponent;
    ServerOptions::foul_cycles             = server_param->foul_cycles;
    ServerOptions::golden_goal             = server_param->golden_goal;

    ServerOptions::kickable_area = ServerOptions::kickable_margin + ServerOptions::ball_size + ServerOptions::player_size;

    return true;
}

int ServerParam::number_of_player_types()
{
    if( !player_param )
        return -1;

    return player_param->player_types;
}

Msg_player_type const* ServerParam::get_player_type( int type )
{
    if( player_types == 0 )
    {
        ERROR_OUT << ID << "\nplayer types do not exist";

        return 0;
    }

    if( type >= player_param->player_types || type < 0 )
    {
        ERROR_OUT << ID << "\nwrong type request " << type;

        return 0;
    }

    if( player_types[ type ]->id != type )
        ERROR_OUT << ID << "\nwrong type info";

    return player_types[ type ];
}

Msg_player_type const* ServerParam::get_worst_case_opponent_type()
{
    return &worst_case_opponent_type;
}
