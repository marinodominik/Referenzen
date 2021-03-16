#include "goalie_neck_bmn.h"

#define GOALIE_MAX_ANGLE_TO_BALL 1/180.*PI

bool GoalieNeck::initialized = false;

GoalieNeck::GoalieNeck()
{
    neck_cmd = new NeckCmd();

    go2ball_list = 0;

    time_of_turn = 0;
    time_attacker_seen = 0;
    last_time_look_to_ball = 0;
    turnback2_scandir = 0;
}

GoalieNeck::~GoalieNeck()
{
    delete neck_cmd;
}

bool GoalieNeck::get_cmd( Cmd &cmd )
{
    return goalie_neck( cmd );
}

bool GoalieNeck::init( char const * conf_file, int argc, char const* const* argv )
{
    if( initialized ) return initialized;

    initialized = NeckCmd::init( conf_file, argc, argv );

    if( !initialized ) { std::cout << "GoalieNeck behavior NOT initialized!!!"; }

    return initialized;
}

bool GoalieNeck::neck_lock( Cmd &cmd )
{
    LOG_POL( 3, <<"Goalie_Neck - neck_lock: Locking neck angle to body angle" );

    neck_cmd->set_turn_neck_rel( 0.0 );
    return neck_cmd->get_cmd( cmd );
}

void GoalieNeck::compute_steps2go()
{
    /* calculate number of steps to intercept ball, for every player on the pitch, return sorted list */
    Policy_Tools::go2ball_steps_update();

    go2ball_list = Policy_Tools::go2ball_steps_list();

    steps2go.me              = 1000;
    steps2go.opponent        = 1000;
    steps2go.opponent_number =    0;
    steps2go.teammate        = 1000;
    steps2go.teammate_number =    0;
    steps2go.my_goalie       = 1000;

    /* get predicted steps to intercept ball for me, my goalie , fastest teammate and fastest opponent */
    for( int i = 0; i < 22; i++ )
    {
        if( Go2Ball_Steps::MY_TEAM == go2ball_list[ i ].side )
        {
            if( WSinfo::me->number == go2ball_list[ i ].number )
            {
                steps2go.me        =  go2ball_list[ i ].steps;
            }
            else if( WSinfo::ws->my_goalie_number == go2ball_list[ i ].number )
            {
                steps2go.my_goalie =  go2ball_list[ i ].steps;

            }
            else if( steps2go.teammate == 1000 )
            {
                steps2go.teammate        = go2ball_list[ i ].steps;
                steps2go.teammate_number = go2ball_list[ i ].number;
            }
        }
        if( go2ball_list[ i ].side == Go2Ball_Steps::THEIR_TEAM )
        {
            if( steps2go.opponent  == 1000 )
            {
                steps2go.opponent        = go2ball_list[ i ].steps;
                steps2go.opponent_number = go2ball_list[ i ].number;
            }
        }
    }

    PPlayer p_tmp = WSinfo::get_opponent_by_number( steps2go.opponent_number );
    if( p_tmp != NULL )
    {
        steps2go.opponent_pos               = p_tmp->pos;
        steps2go.ball_kickable_for_opponent = WSinfo::is_ball_kickable_for( p_tmp );
    }
    else
    {
        steps2go.opponent_pos               = Vector( 0.0, 0.0 );
        steps2go.ball_kickable_for_opponent = false;
    }

    p_tmp = WSinfo::get_teammate_by_number( steps2go.teammate_number );
    if( p_tmp != NULL )
    {
        steps2go.teammate_pos               = p_tmp->pos;
        steps2go.ball_kickable_for_teammate = WSinfo::is_ball_kickable_for( p_tmp );
    }
    else
    {
        steps2go.teammate_pos               = Vector( 0.0, 0.0 );
        steps2go.ball_kickable_for_teammate = false;
    }
}

bool GoalieNeck::goalie_neck( Cmd &cmd )
{
    double VZ;
    IntentionType it;
    double neck2ball_dir;
    mdpInfo::get_my_neck_intention( it );

    if( it.type == NECK_INTENTION_SCANFORBALL )
    {
        LOG_POL( 3, <<"Goalie_Neck: Got neck intention type SCANFORBALL!" );
        return neck_lock( cmd );
    }

    compute_steps2go();

    if( ( ( WSinfo::ws->play_mode == PM_my_GoalKick ) || ( WSinfo::ws->play_mode == PM_my_GoalieFreeKick ) ) && ( it.type != NECK_INTENTION_LOOKINDIRECTION ) )
        return neck_standard( cmd );

    if( it.type == NECK_INTENTION_LOOKINDIRECTION )
    {
        LOG_POL( 0, << "Neck_Intention ist gesetzt, Drehung nach" << it.p1 );
        time_of_turn  = WSinfo::ws->time;
        neck2ball_dir = it.p1;
    }
    else
    {
        neck2ball_dir = mdpInfo::my_neck_angle_to( WSinfo::ball->pos );

        if( ( !steps2go.ball_kickable_for_opponent ) && ( !steps2go.ball_kickable_for_teammate ) )
        {
            neck2ball_dir = mdpInfo::my_neck_angle_to( WSinfo::ball->pos + WSinfo::ball->vel );
        }
        else if( ( WSinfo::me->pos - WSinfo::ball->pos ).sqr_norm() < 36.0 )
        {
            if( steps2go.ball_kickable_for_opponent )
            {
                LOG_DEB( 0, << "ball kickable for opponent!" );
                neck2ball_dir = mdpInfo::my_neck_angle_to( steps2go.opponent_pos );
            }
            else
            {
                LOG_DEB( 0, << "ball kickable for teammate!" );
                neck2ball_dir = mdpInfo::my_neck_angle_to( steps2go.teammate_pos );
            }
        }
    }

    /*
     if (((mdpInfo::my_distance_to_ball() <= 24.0) ||
     (mdpInfo::is_ball_infeelrange_next_time())) &&
     (mdpInfo::time_current() == last_time_look_to_ball + 1) &&
     (!mdpInfo::is_ball_kickable_for_teammate(mdpInfo::teammate_closest_to_ball())) &&
     (!mdpInfo::is_ball_kickable_for_opponent(mdpInfo::opponent_closest_to(WSinfo::ball->pos)))) {
     //neck2ball_dir = mdpInfo::my_neck_angle_to();
     double rel_neck = mdpInfo::mdp->me->neck_angle_rel();
     if (mdpInfo::mdp->me->ang.v < M_PI) {
     neck2ball_dir = 1.5 * M_PI - rel_neck;
     } else {
     neck2ball_dir = M_PI/2.0 - rel_neck;
     }
     } else {
     last_time_look_to_ball = mdpInfo::time_current();
     }*/

    if( cmd.cmd_body.get_type() == cmd.cmd_body.TYPE_TURN )
    {
        double turnangle = 0;
        cmd.cmd_body.get_turn( turnangle );
        if( turnangle > M_PI )
        {
            turnangle = 2 * M_PI - turnangle;
            VZ = 1.0;
        }
        else
        {
            VZ = -1.0;
        }
        turnangle = turnangle / ( 1.0 + ServerOptions::inertia_moment * WSinfo::me->vel.norm() );
        neck2ball_dir = neck2ball_dir + VZ * turnangle;
    }

    if( Tools::get_abs_angle( neck2ball_dir ) > GOALIE_MAX_ANGLE_TO_BALL )
    {
        LOG_MOV( 1, <<"Turn Neck in Goalie-mode:  try to turn to ball "<<RAD2DEG(neck2ball_dir) );

        time_of_turn = WSinfo::ws->time;

        neck_cmd->set_turn_neck( neck2ball_dir );
        return neck_cmd->get_cmd( cmd );
    }
    else
    {
        LOG_MOV( 1, <<"Turn Neck in Goalie-mode:  Already looking in ball direction " );

        time_of_turn = WSinfo::ws->time;

        neck_cmd->set_turn_neck( 0.0 );
        return neck_cmd->get_cmd( cmd );
    }
}

bool GoalieNeck::neck_standard( Cmd &cmd )
{
    double desired_turn;

    if( cmd.cmd_body.get_type() == cmd.cmd_body.TYPE_TURN )
    {
        LOG_MOV( 1, <<"Turn Neck: body turn command -> do not turn neck " );

        time_of_turn = WSinfo::ws->time;

        neck_cmd->set_turn_neck( 0.0 );
        return neck_cmd->get_cmd( cmd );
    }

    if( mdpInfo::mdp->time_of_last_update > time_of_turn )
    {
        if( ( WSinfo::me->neck_ang_rel.get_value() < PI ) && ( WSinfo::me->neck_ang_rel.get_value() > fabs( ServerOptions::maxneckang.get_value() ) - 10. / 180. * PI ) )
            desired_turn = -2 * fabs( ServerOptions::minneckang.get_value() );
        else
            desired_turn = 0.9 * mdpInfo::view_angle_width_rad();

        desired_turn = Tools::get_angle_between_null_2PI( desired_turn );

        time_of_turn = WSinfo::ws->time;

        neck_cmd->set_turn_neck( desired_turn );
        return neck_cmd->get_cmd( cmd );
    }
    else
    {
        neck_cmd->set_turn_neck( 0.0 );
        return neck_cmd->get_cmd( cmd );
    }
}
