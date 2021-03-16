#include <iomanip>
#include <string>

#define DEBUG 1   /* 1: debugging; 0: no debugging */

#define DEBUG_METHOD_NAME_LENGTH  18
#define DEBUG_VAR_NAME_LENGTH     22
#define DEBUG_VAR_VAL_LENGTH       3

#define DEBUG_VAR(XXX)                     std::setw(DEBUG_VAR_NAME_LENGTH) << #XXX << ": " << std::setw(DEBUG_VAR_VAL_LENGTH) << XXX
#define DEBUG_VAR_WITH_NAME(XXX, YYY)      std::setw(DEBUG_VAR_NAME_LENGTH) <<  XXX << ": " << std::setw(DEBUG_VAR_VAL_LENGTH) << YYY
#define DEBUG_BOOL_VAR_WITH_NAME(XXX, YYY) std::setw(DEBUG_VAR_NAME_LENGTH) <<  XXX << ": " << std::setw(DEBUG_VAR_VAL_LENGTH) << boolalpha << YYY

#define DEBUG_VECTOR(XXX)                  std::setw(DEBUG_VAR_NAME_LENGTH-2) << #XXX << ".x: " << std::setw(DEBUG_VAR_VAL_LENGTH) << XXX.x << std::setw(DEBUG_VAR_NAME_LENGTH-2) << #XXX << ".y: " << std::setw(DEBUG_VAR_VAL_LENGTH) << XXX.y

#if DEBUG
#define LOG(XXX) LOG_POL( 0, "GoToBallAndAlign::" << std::setw(DEBUG_METHOD_NAME_LENGTH) << __FUNCTION__ << "() " << XXX )
#define DRAW_CIRCLE(CENTER_VEC, RADIUS, COLOR) LOG_POL( 0, << _2D << VC2D( CENTER_VEC, RADIUS, COLOR ) )
#define DRAW_LINE(START_VEC, END_VEC, COLOR) LOG_POL( 0, << _2D << VL2D( START_VEC, END_VEC, COLOR ) )
#else
#define LOG(XXX)
#define DRAW_CIRCLE(CENTER, RADIUS, COLOR)
#define DRAW_LINE(START_VEC, END_VEC, COLOR)
#endif

#include "GoToBallAndAlign_bms.h"

bool GoToBallAndAlign::initialized = false;

const double GoToBallAndAlign::max_go_directly_to_ball_dist = 3.;
const double GoToBallAndAlign::go_to_dropped_perpendicular_foot_tolerance = 0.1;
const double GoToBallAndAlign::min_angle = 0.09;
const double GoToBallAndAlign::min_nearing_dist = 0.2;

Vector GoToBallAndAlign::HIS_GOAL = HIS_GOAL_CENTER;

GoToBallAndAlign::GoToBallAndAlign()
{
    ivpGoToPos    = new GoToPos2016();
    ivpFaceBall   = new FaceBall();
    ivpSearchBall = new SearchBall();
    ivpBasicCmd   = new BasicCmd();

    ivpGoToPos->setAngleDiff(min_angle);

    firstActTimeThisStdSituation = -ServerOptions::drop_ball_time;

    pos_state = go_to_ball_directly;
    done_pos = false;
    already_nearing_on_line = false;

    additionalMinDistFactor = ( ServerOptions::ball_size + ServerOptions::player_size ) + 0.1;
}

GoToBallAndAlign::~GoToBallAndAlign()
{
    if(ivpGoToPos)    { delete ivpGoToPos;    }
    if(ivpFaceBall)   { delete ivpFaceBall;   }
    if(ivpSearchBall) { delete ivpSearchBall; }
    if(ivpBasicCmd)   { delete ivpBasicCmd;   }
}

bool GoToBallAndAlign::init(char const * conf_file, int argc, char const* const* argv)
{
    if( !initialized )
    {
        bool result = true;

        result &= GoToPos2016::init( conf_file, argc, argv );
        result &= FaceBall::init(    conf_file, argc, argv );
        result &= SearchBall::init(  conf_file, argc, argv );
        result &= BasicCmd::init(    conf_file, argc, argv );

        initialized = result;
    }

    return initialized;
}

bool GoToBallAndAlign::get_cmd(Cmd& cmd)
{
    return get_cmd(cmd, HIS_GOAL, 1);
}

bool GoToBallAndAlign::get_cmd( Cmd& cmd, Vector& target )
{
    return get_cmd(cmd, target, 1);
}

bool GoToBallAndAlign::get_cmd_to_align_to_his_goal(Cmd& cmd)
{
    return get_cmd(cmd, HIS_GOAL, 1);
}

bool GoToBallAndAlign::get_cmd(Cmd& cmd, Vector& target, int depthFactor)
{
    if(depthFactor == 1){
        LOG( "-------------------------------------------------" );
        LOG( "GoToBallAndAlign::get_cmd()" );
    }
    bool cmd_set = false;

    if( WSinfo::ws->time - firstActTimeThisStdSituation > ServerOptions::drop_ball_time )
    {
        firstActTimeThisStdSituation = WSinfo::ws->time;
        reset_state();
    }

    if( !WSinfo::is_ball_pos_valid() )
    {
        if( !ivpSearchBall->is_searching() )
        {
            ivpSearchBall->start_search();
        }
        cmd_set = ivpSearchBall->get_cmd(cmd);
        LOG( "Search Ball" );
    }
    else
    {
        bool i_am_near = ( WSinfo::me->pos - WSinfo::ball->pos ).sqr_norm() < pow( GoToBallAndAlign::max_go_directly_to_ball_dist, 2 ) ;
        DRAW_CIRCLE( WSinfo::ball->pos, GoToBallAndAlign::max_go_directly_to_ball_dist, "000000");

//        if( !i_am_near && pos_state != go_to_ball_directly )
//        {
//            reset_state();
//        }

            Vector nearingPos = Tools::get_Lotfuss( WSinfo::ball->pos, target, WSinfo::me->pos );
            Vector additionalMinDistVec = ( WSinfo::ball->pos - target ).normalize() * additionalMinDistFactor;

            if(nearingPos.distance(target) < (WSinfo::ball->pos + ( already_nearing_on_line ? additionalMinDistVec : additionalMinDistVec * 3 ) ).distance(target))
            {
                nearingPos = (WSinfo::ball->pos + ( already_nearing_on_line ? additionalMinDistVec : additionalMinDistVec * 3 ) );
            }

            /* Visualisierung der obrigen Berechnung */
            DRAW_LINE( target, WSinfo::ball->pos, "FF0000" );
            DRAW_LINE( nearingPos, WSinfo::me->pos, "FF0000" );
            DRAW_LINE( WSinfo::ball->pos, nearingPos, "0000FF" );
            DRAW_CIRCLE( nearingPos, 0.1, "000000");
            DRAW_CIRCLE( WSinfo::ball->pos, 0.1, "000000");


            bool ang_to_target_is_good = fabs( Tools::my_angle_to( target            ).get_value_mPI_pPI() ) < GoToBallAndAlign::min_angle;
            bool ang_to_ball_is_good   = fabs( Tools::my_angle_to( WSinfo::ball->pos ).get_value_mPI_pPI() ) < GoToBallAndAlign::min_angle;
            LOG( DEBUG_VAR( ang_to_target_is_good ) );
            LOG( DEBUG_VAR( ang_to_ball_is_good ) );
//            LOG( DEBUG_VAR( Tools::my_abs_angle_to( WSinfo::ball->pos ).get_value_mPI_pPI() ) );
//            LOG( DEBUG_VAR( ( target - WSinfo::ball->pos ).sqr_norm() ) );
//            LOG( DEBUG_VAR( ( target - WSinfo::me->pos   ).sqr_norm() ) );

            if( !cmd_set && pos_state == go_to_ball_directly )
            {
                LOG( "IN: Go directly!" );
                if( !i_am_near )
                {
                    cmd_set = ivpGoToPos->get_cmd_go_to( cmd, WSinfo::ball->pos );
                    LOG( "Go directly!" );
                    LOG( DEBUG_VAR( cmd_set ) );
                }
                else
                {
                    pos_state = go_to_nearing_pos;
                }
            }

            if( !cmd_set && pos_state == go_to_nearing_pos )
            {
                LOG( "IN: Go to nearing pos!" );

                if( nearingPos.distance(WSinfo::me->pos) > GoToBallAndAlign::go_to_dropped_perpendicular_foot_tolerance * 2. / depthFactor )
                {
                    LOG( "Go to nearing Pos!" );
                    LOG( DEBUG_VAR( nearingPos.distance(WSinfo::me->pos) ) );
                    cmd_set = ivpGoToPos->get_cmd_go_to( cmd, nearingPos, GoToBallAndAlign::go_to_dropped_perpendicular_foot_tolerance / depthFactor );
                    LOG( DEBUG_VAR( cmd_set ) );
                }
                else
                {
                    already_nearing_on_line = true;
                    pos_state = stop_on_nearing_pos;
                }
            }

            if( !cmd_set && pos_state == stop_on_nearing_pos )
            {
                LOG( "IN: Stop on dropped perpendicular foot!" );
                LOG( DEBUG_VAR( WSinfo::me->vel.norm() ) );
                if( WSinfo::me->vel.norm() > 0.01 )
                {
                    LOG( "Stop on dropped perpendicular foot!" );
                    LOG( DEBUG_VAR( WSinfo::me->vel.norm() ) );
                    ivpGoToPos->setPermissionToGoToAlreadyReachedTarget();
                    cmd_set = ivpGoToPos->get_cmd_go_to( cmd, nearingPos, GoToBallAndAlign::go_to_dropped_perpendicular_foot_tolerance );
                    LOG( DEBUG_VAR( cmd_set ) );
                }
                else
                {
                    pos_state = turn_to_target;
                }
            }

            if( !cmd_set && pos_state == turn_to_target )
            {
                LOG( "IN: Turn to target!" );
                if( ! (ang_to_ball_is_good && ang_to_target_is_good ) )
                {
                    if( ang_to_target_is_good )
                    {
                        LOG( "RE: NEARING!" );
                        pos_state = go_to_nearing_pos;
                        cmd_set = get_cmd( cmd, target, depthFactor * 2 );
                    }
                    else
                    {
                        cmd_set = ivpFaceBall->get_cmd_turn_to_point( cmd, target );
                        LOG( "Turn to target!" );
                        LOG( DEBUG_VAR( cmd_set ) );
                    }
                }
                else
                {
                    pos_state = approach_the_ball;
                }
            }

            if( !cmd_set && pos_state == approach_the_ball )
            {
                LOG( "IN: Approach the ball!" );
                if( !ang_to_ball_is_good || !ang_to_target_is_good )
                {
                    LOG( "RE: Turn to Target!" );
                    pos_state = turn_to_target;
                    cmd_set = get_cmd( cmd, target, depthFactor * 2 );
                }
                else if( ( WSinfo::me->pos.distance( WSinfo::ball->pos ) - ( ServerOptions::ball_size + ServerOptions::player_size ) ) > min_nearing_dist)
                {
                    LOG( "Approaching ...!" );
                    LOG( DEBUG_VAR( WSinfo::me->pos.distance( WSinfo::ball->pos ) ) );
                    ivpGoToPos->get_cmd_go_to( cmd, WSinfo::ball->pos + ( ( WSinfo::ball->pos - target ).normalize() * ( ServerOptions::ball_size + ServerOptions::player_size ) ), 0.01 );
                }
                else
                {
                    pos_state = positioning_done;
                }
            }

            if( !cmd_set && pos_state == positioning_done )
            {
            	if( !i_am_near )
            	{
                    LOG( "RE: Go directly!" );
                    pos_state = go_to_ball_directly;
                    cmd_set = get_cmd( cmd, target, depthFactor );
            	}
            	else if( nearingPos.distance(WSinfo::me->pos) > GoToBallAndAlign::go_to_dropped_perpendicular_foot_tolerance * 2. / depthFactor)
            	{
                    LOG( "RE: Go to nearing pos!" );
                    pos_state = go_to_nearing_pos;
                    cmd_set = get_cmd( cmd, target, depthFactor );
            	}
                else if( !ang_to_ball_is_good || !ang_to_target_is_good )
                {
                    LOG( "RE: Turn to Target!" );
                    pos_state = turn_to_target;
                    cmd_set = get_cmd( cmd, target, depthFactor );
                }
                else if( ( WSinfo::me->pos.distance( WSinfo::ball->pos ) - ( ServerOptions::ball_size + ServerOptions::player_size ) ) > min_nearing_dist)
                {
                    LOG( "RE: Approaching ...!" );
                    LOG( DEBUG_VAR( WSinfo::me->pos.distance( WSinfo::ball->pos ) ) );
                    pos_state = approach_the_ball;
                    cmd_set = get_cmd( cmd, target, depthFactor );
                }
                else
                {
					done_pos = true;
					LOG( "Positioning Done! dist="<<WSinfo::ball->pos.distance(WSinfo::me->pos) );
					ivpBasicCmd->set_turn(0.0);
					ivpBasicCmd->get_cmd(cmd);
					cmd_set = true;
                }
            }

    }

    return cmd_set;
}

bool GoToBallAndAlign::isPositioningFinished()
{
  return done_pos;
}

void GoToBallAndAlign::reset_state()
{
    LOG( "" );
    done_pos = false;
    pos_state = go_to_ball_directly;
    already_nearing_on_line = false;
}
