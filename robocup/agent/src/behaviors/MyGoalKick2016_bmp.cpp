#include <iomanip>
#include <string>

#define DEBUG 1   /* 1: debugging; 0: no debugging */

#define DEBUG_METHOD_NAME_LENGTH  18
#define DEBUG_VAR_NAME_LENGTH     22
#define DEBUG_VAR_VAL_LENGTH       3

#define DEBUG_VAR(XXX)                     std::setw(DEBUG_VAR_NAME_LENGTH) << #XXX << ": " << std::setw(DEBUG_VAR_VAL_LENGTH) << XXX
#define DEBUG_VAR_WITH_NAME(XXX, YYY)      std::setw(DEBUG_VAR_NAME_LENGTH) <<  XXX << ": " << std::setw(DEBUG_VAR_VAL_LENGTH) << YYY
#define DEBUG_BOOL_VAR_WITH_NAME(XXX, YYY) std::setw(DEBUG_VAR_NAME_LENGTH) <<  XXX << ": " << std::setw(DEBUG_VAR_VAL_LENGTH) << boolalpha << YYY

#if DEBUG
#define LOG(XXX) LOG_POL( 0, "MyGoalKick2016::" << std::setw(DEBUG_METHOD_NAME_LENGTH) << __FUNCTION__ << "() " << XXX )
#else
#define LOG(XXX)
#endif


#include "MyGoalKick2016_bmp.h"

bool   MyGoalKick2016::cvInitialized          =  false;
bool   MyGoalKick2016::cvDebug                =  true;

Vector MyGoalKick2016::cvZeroVector;

int    MyGoalKick2016::cvDefaultWaitTime      = (ServerOptions::drop_ball_time / 20) * 13; //TG17: 9->13
int    MyGoalKick2016::cvMinWaitTime          =  0;
int    MyGoalKick2016::cvMaxWaitTime          = (ServerOptions::drop_ball_time / 20) * 19; //TG17: 16->19
int    MyGoalKick2016::cvWaitOnLowStaminaTime =  ServerOptions::drop_ball_time - 5;
int    MyGoalKick2016::cvHomeposStaminaMin    =  ServerOptions::stamina_max * 0.5;

const double MyGoalKick2016::cvHOMEPOS_TOLERANCE = 0.8;

/**
 * Initialization of static dummy variables *
 */
double MyGoalKick2016::cvDummyVal;
int    MyGoalKick2016::cvDummyInt;
Vector MyGoalKick2016::cvDummyVec;


MyGoalKick2016::MyGoalKick2016()
{
    ivpBasicCmd                    = new BasicCmd();
    ivpWBall                       = new Wball06();
    ivpFaceBall                    = new FaceBall();
    ivpNeuroKick                   = new NeuroKick05();
    ivpIntercept                   = new NeuroIntercept();
    ivpGoToPos                     = new GoToPos2016();
    ivpGoToBallAndAlign            = new GoToBallAndAlign();
    ivpOneStepKick                 = new OneStepKick();

    ivMyPanickickLeftCornerTarget  = Vector( (-FIELD_BORDER_X + -2.5), (  ( ServerOptions::goal_width / 2 ) +  ( ( FIELD_BORDER_Y - ( ServerOptions::goal_width / 2 ) ) / 2 ) ) );
    ivMyPanickickRightCornerTarget = Vector( (-FIELD_BORDER_X + -2.5), ( -( ServerOptions::goal_width / 2 ) + -( ( FIELD_BORDER_Y - ( ServerOptions::goal_width / 2 ) ) / 2 ) ) );

    goalKickAlreadyDone = false;

    ivLastActTime                  = -1;
    ivActGoalKickStartTime         = -1;
    ivActGoalKickDuration          = -1;
    ivTimeLeft                     = -1;
    ivLastTimeOfRequestToOfferMySelf = -1000;

    reset_state();
}

MyGoalKick2016::~MyGoalKick2016()
{
    if( ivpBasicCmd         ) { delete ivpBasicCmd;         }
    if( ivpWBall            ) { delete ivpWBall;            }
    if( ivpFaceBall         ) { delete ivpFaceBall;         }
    if( ivpNeuroKick        ) { delete ivpNeuroKick;        }
    if( ivpIntercept        ) { delete ivpIntercept;        }
    if( ivpGoToPos          ) { delete ivpGoToPos;          }
    if( ivpGoToBallAndAlign ) { delete ivpGoToBallAndAlign; }
    if( ivpOneStepKick      ) { delete ivpOneStepKick;      }
}

bool MyGoalKick2016::init( char const * conf_file, int argc, char const* const* argv )
{
    if( !cvInitialized )
    {
        bool result = true;

        result &= Wball06::init(          conf_file, argc, argv );
        result &= FaceBall::init(         conf_file, argc, argv );
        result &= NeuroKick05::init(      conf_file, argc, argv );
        result &= NeuroIntercept::init(   conf_file, argc, argv );
        result &= GoToPos2016::init(      conf_file, argc, argv );
        result &= GoToBallAndAlign::init( conf_file, argc, argv );
        result &= OneStepKick::init(      conf_file, argc, argv );

        cvInitialized = result;
    }

    return cvInitialized;
}

bool MyGoalKick2016::get_cmd( Cmd &cmd )
{
    if (   WSinfo::ws //TG17 FINAL HELIOS HACK
        && WSinfo::ws->his_team_score <= WSinfo::ws->my_team_score
        && WSinfo::get_current_opponent_identifier() == TEAM_IDENTIFIER_HELIOS )
    {
      cvDefaultWaitTime = (ServerOptions::drop_ball_time / 20) * 16; //TG17: 9->13
    }
    else
    {
      cvDefaultWaitTime = (ServerOptions::drop_ball_time / 20) * 13; //TG17: 9->13
    }



    bool cmd_set = false;

    int now = WSinfo::ws->time;

    if( now - ivLastActTime > 1 )
    {
        ivActGoalKickStartTime = now;
        reset_state();
        LOG( " --NEW-GOALKICK!!!------------------ " );
        LOG( " " );
    }

    ivLastActTime = now;
    ivActGoalKickDuration = now - ivActGoalKickStartTime;
    ivTimeLeft = this->getTimeLeft();

    if( !goalKickAlreadyDone && !( cmd_set = get_cmd_continued( cmd ) ) )
    {
        if( ClientOptions::consider_goalie ) { cmd_set = get_goalie_cmd( cmd ); }
        else                                 { cmd_set = get_player_cmd( cmd ); }
    }

    if( ivLastIntention.valid_at() == now ) {
        Blackboard::pass_intention = ivLastIntention;
    }

    LOG( "- return = " << cmd_set << " (time left: "<<ivTimeLeft<<")" );
    LOG( " ----------------------------------- " );

    return cmd_set;
}

void MyGoalKick2016::reset_state()
{
    goalKickAlreadyDone = false;
    ivLastKickSpeed = 0;
    ivContinueWBallBehavior = false;
    ivConsecKicks = 0;
}

bool MyGoalKick2016::get_cmd_continued( Cmd &cmd )
{
    bool cmd_set = false;

    if( ivConsecKicks >= cvMAX_CONSEC_KICKS )
    {
        LOG( "- Case 1: Deadlock prevention" );
        reset_state();
    }
    else if( ivLastKickSpeed > 0 )
    {
        if( WSinfo::is_ball_pos_valid() && WSinfo::is_ball_kickable() )
        {
            ivpNeuroKick->reset_state();
            ivpNeuroKick->kick_to_pos_with_initial_vel( ivLastKickSpeed, ivLastKickTarget );
            if( ( cmd_set = ivpNeuroKick->get_cmd( cmd ) ) )
            {
                LOG( "- Case 2: Continue NeuroKick" );
                ivLastIntention.set_valid_at( WSinfo::ws->time );
                ivConsecKicks++;
            }
            else
            {
                LOG( "- Case 2: Continue NeuroKick - NeuroKick didn't set a cmd!" );
            }
        }
        else{
            LOG( "- Case 2: Continue NeuroKick - NOT! => reset" );
            LOG( "- GOALKICK ALREADY DONE!!!" );
            goalKickAlreadyDone = true;
        }
    }
    else if( ivContinueWBallBehavior )
    {
        ivContinueWBallBehavior = false;
        Intention intention;
        if( getPassIntention( intention ) )
        {
            if( ( cmd_set = ivpWBall->intention2cmd( intention, cmd ) ) )
            {
                LOG( "- Case 3: Continue WBall" );
                intention.get_kick_info( ivLastKickSpeed, ivLastKickTarget );
                ivLastIntention = intention;
                ivContinueWBallBehavior = true;
                ivConsecKicks++;
            }
            else
            {
                LOG( "- Case 3: Continue WBall - WBall didn't set a cmd!" );
            }
        }
        else
        {
            LOG( "- Case 3: Continue WBall - NOT! => NO Intention" );
        }
    }
    else
    {
//        LOG( "- NO CMD continued!" );
    }
    if (   OpponentAwarePositioning::getRole(WSinfo::me->number != PT_DEFENDER)
        && WSinfo::ws->my_goalie_number > 0
        && ivTimeLeft < 30)
      Tools::set_attention_to_request(WSinfo::ws->my_goalie_number);
    return cmd_set;
}

bool MyGoalKick2016::get_goalie_cmd( Cmd &cmd )
{
    bool cmd_set = false;

    if( !goalKickAlreadyDone )
    {
        int remainingGoalKickTime = ServerOptions::drop_ball_time - ivActGoalKickDuration;
        bool ballIsKickable = WSinfo::is_ball_pos_valid() && WSinfo::is_ball_kickable();

        if( remainingGoalKickTime < cvPANICKICK_TRESHOLD )
        {
            LOG( "- DO PANICKICK" );
            if( ballIsKickable )
            {
                if( WSinfo::me->pos.getY() < 0 )
                {
                    ivpOneStepKick->kick_to_pos_with_initial_vel(2.5, ivMyPanickickRightCornerTarget);
                    //ivpNeuroKick->kick_to_pos_with_initial_vel( 2.5, ivMyPanickickRightCornerTarget );
                }
                else
                {
                    ivpOneStepKick->kick_to_pos_with_initial_vel( 2.5, ivMyPanickickLeftCornerTarget );
                    //ivpNeuroKick->kick_to_pos_with_initial_vel( 2.5, ivMyPanickickLeftCornerTarget );
                }
                LOG( "- DO PANICKICK - DO IT!" );
                ivpOneStepKick->get_cmd( cmd );
                cmd_set = true;
                //cmd_set = ivpNeuroKick->get_cmd( cmd );

                goalKickAlreadyDone = true;
            }
            else
            {
                LOG( "- DO PANICKICK - RUN TO BALL!" );
                cmd_set = ivpGoToPos->get_cmd_go_to( cmd, WSinfo::ball->pos );
            }
        }
        else // Normal GoalKick Behavior
        {
            LOG( "- Do normal GoalKick behavior" );
            Intention intention;

            //if( !ivpGoToBallAndAlign->get_cmd_to_align_to_his_goal( cmd )/*ballIsKickable*/ )
            ivpGoToBallAndAlign->get_cmd_to_align_to_his_goal( cmd );/*ballIsKickable*/
            if( ivpGoToBallAndAlign->isPositioningFinished() )
            {
                if( ivTimeLeft > 10 && ivTimeLeft < 16 )
                {
                    LOG( "- Do normal GoalKick behavior - Phase 2 - start" );
                    cmd.cmd_say.set_pass( WSinfo::me->pos, cvZeroVector, WSinfo::ws->time ); // I give THE SIGNAL to my teammates to RUN FREE.
                    LOG( "- Do normal GoalKick behavior - Phase 2 - end ( SIGNAL TO RUN FREE GIVEN )" );
                }
                else if( ivTimeLeft <= cvPASS_ANNOUNCE_TIME && WSinfo::ws->time % 20 >= ( 7 - cvPASS_ANNOUNCE_TIME ) ) // last part of condition:
                {                                                                                                      // don't kick while player scan field
                    LOG( "- Do normal GoalKick behavior - Phase 3 - start" );
                    if( getPassIntention( intention ) )
                    {
                        Blackboard::pass_intention = intention;

                        double new_kick_speed,  old_kick_speed;
                        Vector new_kick_target, old_kick_target;

                        intention.get_kick_info(       new_kick_speed, new_kick_target );
                        ivLastIntention.get_kick_info( old_kick_speed, old_kick_target );

                        if( new_kick_speed != old_kick_speed || new_kick_target.sqr_distance( old_kick_target ) > 1.5 )
                        {
                            ivContinueWBallBehavior = true;
                            cmd_set = ivpFaceBall->get_cmd_turn_to_point( cmd, new_kick_target );
                            LOG( "- Do normal GoalKick behavior - Phase 3 - Contintue WBall in next timestep!" );
                        }
                        else
                        {
                            LOG( "- Do normal GoalKick behavior - Phase 3 - DON'T Contintue WBall in next timestep!" );
                        }
                        ivLastIntention = intention;
                    }
                    else{
                        LOG( "- Do normal GoalKick behavior - Phase 3 - NO Intention!" );
                    }
                    LOG( "- Do normal GoalKick behavior - Phase 3 - end" );
                }
            }

            if( ivTimeLeft == 0 && ivLastIntention.valid_at() == WSinfo::ws->time && ( cmd_set = ivpWBall->intention2cmd( intention, cmd ) ) && WSinfo::ws->time % 20 >= 7 ) // last part of condition:
            {                                                                                                                                                                // don't kick while player scan field
                LOG( "- Do normal GoalKick behavior - Phase 4 - Contintue WBall in next timestep (FORCED)!" );
                ivContinueWBallBehavior = true;
                intention.get_kick_info( ivLastKickSpeed, ivLastKickTarget );
                ivLastIntention = intention;
            }

            if( cmd_set && cmd.cmd_body.is_type( cmd.cmd_body.TYPE_DASH ) && ivTimeLeft > cvTIME_LOW_THRESHOLD )
            {
                double allowedDashPower, intendedDashPower, absoluteMaxDash = 100;

                if     ( WSinfo::me->stamina < 0.45 * ServerOptions::stamina_max ) { absoluteMaxDash =       cvLIMITED_DASH_POWER; }
                else if( WSinfo::me->stamina < 0.60 * ServerOptions::stamina_max ) { absoluteMaxDash = 1.5 * cvLIMITED_DASH_POWER; }

                cmd.cmd_body.get_dash( intendedDashPower );

                if( intendedDashPower < 0 ) { intendedDashPower *= 2; }

                if( fabs( intendedDashPower ) > absoluteMaxDash ) { allowedDashPower = absoluteMaxDash;   }
                else                                              { allowedDashPower = intendedDashPower; }

                if( intendedDashPower < 0 ) { allowedDashPower *= -0.5; }

                cmd.cmd_body.unset_lock();
                cmd.cmd_body.unset_cmd();
                cmd.cmd_body.set_dash( allowedDashPower );
                LOG( "- Dash-Cmd modified!" );
            }

            cmd_set = true;
        }
    }

    return cmd_set;
}

bool MyGoalKick2016::get_player_cmd(Cmd &cmd)
{
#if LOGGING && BASIC_LOGGING
    Vector target = getTargetNonStartPlayer(); // get Cover Position
#endif
    LOG_POL( 0, << _2D << VC2D( target, ( 2 + ( ( WSinfo::ws->time % 5 ) * 0.2 ) ), "000000" ) );
    LOG_POL( 0, << _2D << VL2D( target, WSinfo::me->pos, "000000" ) );
    LOG_POL( 0, << _2D << L2D( -FIELD_BORDER_X + 7, -FIELD_BORDER_Y, -FIELD_BORDER_X + 7, FIELD_BORDER_Y, "000000" ) );

    bool cmd_set = false;

    if( !WSinfo::is_ball_pos_valid() || WSinfo::ball->time <= ivActGoalKickStartTime ) // must search ball?
    {
        LOG( "- Search Ball!" );
        cmd_set = ivpFaceBall->get_cmd_turn_to_ball( cmd );
    }
    else if( LEFT_PENALTY_AREA.inside( WSinfo::ball->pos ) && WSinfo::ball->pos.getX() > -FIELD_BORDER_X + 7 )
    {
        LOG( "- GOALKICK ALREADY DONE!!!");

        cmd_set = ivpWBall->get_cmd( cmd );
    }
    else
    {
        Vector target = getTargetNonStartPlayer(); // get Cover Position
        PPlayer p = WSinfo::get_teammate_with_newest_pass_info();
        LOG_POL( 0, "- Player with pass info: "<< (p==NULL?-1:p->number) );
        //TG17: Yes, eventually a null pointer p will be handed over in the
        //      subsequent call. Thus, care must be taken in that method!
        if( !( ( cmd_set = getCmdReactOnPassInfo( cmd, p, target ) ) ) ) // TODO Sollte nicht vorher oder in der Methode getestet werden ob die Infos aktuell sind?
        {
            // LOG_POL( 0, "GOT NO PASS INFO!" );

            bool iAmOnHomePos = WSinfo::me->pos.distance( target ) < cvHOMEPOS_TOLERANCE;
            bool iHaveNotEnoughStamina = WSinfo::me->stamina < cvHomeposStaminaMin;
            bool noTimeLeftToWalkToHomePos = ivTimeLeft < cvPASS_ANNOUNCE_TIME;
            bool iAmCloserToBallThanMyCoveredOpp = WSinfo::me->pos.distance( WSinfo::ball->pos ) < target.distance( WSinfo::ball->pos );
            bool iAmNotInsideOurPenaltyArea = !LEFT_PENALTY_AREA.inside( WSinfo::me->pos );

            if( iAmOnHomePos || iHaveNotEnoughStamina || ( noTimeLeftToWalkToHomePos && iAmCloserToBallThanMyCoveredOpp && iAmNotInsideOurPenaltyArea ) )
            {
                if( WSinfo::ws->time % 20 < 7 ) // Scan field for ball
                {
                    ANGLE dir;

                    if( WSinfo::ws->time_of_last_update == WSinfo::ws->time )
                    {
                        if( WSinfo::ws->view_quality == Cmd_View::VIEW_QUALITY_LOW )
                        {
                            dir = ANGLE( DEG2RAD( 15 ) );
                        }
                        else
                        {
                            dir = ANGLE( DEG2RAD( 43 ) );
                        }
                    }

                    Tools::set_neck_request( NECK_REQ_SCANFORBALL );

                    ivpBasicCmd->set_turn_inertia( dir.get_value_mPI_pPI() );
                    LOG( "- ScanField! ( time % 20 < 7 )" );
                    cmd_set = ivpBasicCmd->get_cmd( cmd );
                }
                else if( fabs( Tools::my_angle_to( WSinfo::ball->pos ).get_value_mPI_pPI() ) > DEG2RAD( 5 ) )
                {
                    LOG( "- Turn to Ball" );
                    cmd_set = ivpFaceBall->get_cmd_turn_to_ball( cmd );
                }
                else
                {
                    LOG( "- Do nothing!" );
                }
            }
            else
            {
                if( target.getX() < -33 ) { target.setX( -33 ); } //do not enter Strafraum

                ivpGoToPos->set_target( target ); // TODO: vlt höhere Tolleranz bevor PassInfo rein kommt
                LOG( "- Run to Coverposition" );
                cmd_set = ivpGoToPos->get_cmd( cmd );
            }
        }
        else
        {
            LOG( "- React on PassInfo!" );
        }
    }
    return cmd_set;
}

Vector MyGoalKick2016::getTargetNonStartPlayer() {
    Vector target = OpponentAwarePositioning::getCoverPosition();

    PlayerSet nearTeammates = WSinfo::valid_teammates_without_me.keep_players_in_circle( WSinfo::me->pos, 5.0 );
    if( nearTeammates.num > 0 && WSinfo::me->number > nearTeammates[0]->number )
    {
        target.addToX( 6.0 );
    }

    if( WSinfo::me->number == 2 || WSinfo::me->number == 5 )
    {
        PlayerSet nearOpps = WSinfo::valid_opponents.keep_players_in_circle( target, 6.0 );
        if( nearOpps.num > 0 )
        {
            if     ( target.getY() >=  0   ) { target.addToY(     6.0 ); }
            else if( target.getY() <   0   ) { target.subFromY(   6.0 ); }
            if     ( target.getY() >  30.0 ) { target.setY(      30.0 ); }
            else if( target.getY() < -30.0 ) { target.setY(     -30.0 ); }
        }
    }

    if( target.getX() < -FIELD_BORDER_X + WSinfo::me->kick_radius ) { target.setX( -FIELD_BORDER_X + WSinfo::me->kick_radius ); }

    if( WSinfo::ball->pos.getX() < 20 && OpponentAwarePositioning::getRole( WSinfo::me->number ) == PT_DEFENDER )
    {
        target += .04 * ( MY_GOAL_CENTER - target );
    }

    return target;
}

bool MyGoalKick2016::getCmdReactOnPassInfo( Cmd &cmd, const PPlayer p, Vector &origTarget )
{
    bool returnValue = false;
    LOG( "- Entered getCmdReactOnPassInfo" << flush);
    if(   (!p && (WSinfo::ws->time - ivLastTimeOfRequestToOfferMySelf < 40))
       || (p && (p->pass_info.ball_vel.sqr_norm() < 0.01)) )
    {
        ivLastTimeOfRequestToOfferMySelf = WSinfo::ws->time;
        // That was no pass announcement, but a request to OFFER MYSELF (target modified)!
        ivpGoToPos->set_target( origTarget );
        // consider both defenders
        if (   (WSinfo::me->number == 2 && WSinfo::ball->pos.getY() > 0.0 )
            || (WSinfo::me->number == 5 && WSinfo::ball->pos.getY() < 0.0) )
        {
          Vector leftOrRightDefendeTarget;
          leftOrRightDefendeTarget.setX(-40.0);
          leftOrRightDefendeTarget.setY(  WSinfo::ball->pos.getY() > 0.0
                                        ? FIELD_BORDER_Y - 3.0
                                        : -FIELD_BORDER_Y + 3.0);
          ivpGoToPos->set_target( leftOrRightDefendeTarget );
          if (leftOrRightDefendeTarget.distance(WSinfo::me->pos) < 2.0)
          {
            LOG( "- Near enough to special offer pos." <<flush);
            ivpBasicCmd->set_turn_inertia( Tools::my_angle_to(MY_GOAL_CENTER).get_value_mPI_pPI() );
            return ivpBasicCmd->get_cmd(cmd);
          }
        }
        // consider left/right midfielder
        Vector midfielderTarget;
        midfielderTarget.setX( -FIELD_BORDER_X * 0.5 );
        midfielderTarget.setY(  WSinfo::ball->pos.getY() > 0.0
                              ? FIELD_BORDER_Y - 5.0
                              : -FIELD_BORDER_Y + 5.0);
        PlayerSet tmms = WSinfo::valid_teammates;
        for (int t=1; t<NUM_PLAYERS; t++)
          if (! (   OpponentAwarePositioning::getRole( t ) == PT_MIDFIELD
                 || OpponentAwarePositioning::getRole( t ) == PT_ATTACKER) )
            tmms.remove( tmms.get_player_by_number(t) );
        tmms.keep_and_sort_closest_players_to_point(1, midfielderTarget);
        if (tmms.num > 0)
        {
          LOG( "- Nearest player to "<<midfielderTarget<<" is my "<<tmms[0]->number<<"." <<flush);
        }
        if (   tmms.num > 0
            && tmms[0]->number == WSinfo::me->number
            && OpponentAwarePositioning::getRole( WSinfo::me->number ) != PT_DEFENDER
            && Tools::willMyStaminaCapacitySufficeForThisHalftime() )
        {
          midfielderTarget.setX( Tools::max( Tools::min( WSinfo::me->pos.getX(),
                                                         midfielderTarget.getX() + 10.0),
                                             midfielderTarget.getX() - 10.0 ) );
          ivpGoToPos->set_target( midfielderTarget );
          if (midfielderTarget.distance(WSinfo::me->pos) < 2.0)
          {
            LOG( "- Near enough to special offer pos (midfielder)." <<flush);
            ivpBasicCmd->set_turn_inertia( Tools::my_angle_to(MY_GOAL_CENTER).get_value_mPI_pPI() );
            return ivpBasicCmd->get_cmd(cmd);
          }
        }
        // default
        returnValue = ivpGoToPos->get_cmd( cmd );
        LOG( "- Got a request to offer myself!" <<flush);
    }
    else
    if (p == NULL)
    {
      returnValue = false;
    }
    else
    {
        // check if i should intercept
        InterceptResult ires[3];
        PlayerSet ps = WSinfo::valid_teammates.keep_and_sort_best_interceptors_with_intercept_behavior( 3, p->pass_info.ball_pos, p->pass_info.ball_vel, ires );

        for( int p = 0; p < ps.num; p++ )
        {
            // LOG_POL(0, "Best intercept: p="<<p<<",#="<<ps[p]->number <<",pos="<<ires[p].pos);
            if( ps[p]->number == WSinfo::me->number )
            {
                if( WSinfo::me->pos.distance( WSinfo::ball->pos ) > 35 )
                {
                    // Could go for pass but i am too far away, see info might be wrong
                    continue;
                }
                Vector iPos, ballPos, ballVel; //TG17: Bug removed: Using cvDummyVec 3 times yields errors (call by reference)!
                if( Policy_Tools::check_go4pass( iPos, ballPos, ballVel ) ) // TODO: Warum Policy_Tools::check_go4pass() und nicht Policy_Tools::I_am_fastest4pass()?
                {
                    // I intercept!
                    mdpInfo::set_my_intention( DECISION_TYPE_INTERCEPTBALL );
                    ivpIntercept->set_virtual_state( ires[p].pos, cvZeroVector );
                    returnValue = ivpIntercept->get_cmd( cmd );
                }
            }
        }
        LOG( "- I intercept the ball (rv="<<returnValue<<")!" );
    }
    return returnValue;
}

int MyGoalKick2016::getTimeLeft()
{
    int additionalWait = int( 0.75 * ServerOptions::stamina_max - WSinfo::me->stamina ) / 30;
    if( additionalWait < 0 ) { additionalWait = 0; }

    int goalDiff = WSinfo::ws->my_team_score - WSinfo::ws->his_team_score;

    ivStandardWaitTime = cvDefaultWaitTime + ( ( goalDiff > 0 && goalDiff < 4 ) ? 30 * goalDiff : 0 );

    int desiredWait = Tools::max( ivStandardWaitTime, ivActGoalKickDuration + additionalWait );

    if( goalDiff >= 0 && !Tools::willStaminaCapacityOfAllPlayersSufficeForThisHalftime() )
      desiredWait = cvWaitOnLowStaminaTime;

    if     ( desiredWait < cvMinWaitTime )
      desiredWait = cvMinWaitTime;
    else
      if ( desiredWait > cvMaxWaitTime )
        desiredWait = cvMaxWaitTime;

    int kick_time = ivActGoalKickStartTime + desiredWait;

    if( kick_time < WSinfo::ws->time ) { kick_time = WSinfo::ws->time; }

//    LOG( " " );
//    LOG( "   |-----------------------------|" );
//    LOG( "   | " << DEBUG_VAR( ivActGoalKickDuration ) << " | ");
//    LOG( "   |-----------------------------|" );
//    LOG( "   | " << DEBUG_VAR( additionalWait ) << " | " );
//    LOG( "   | " << DEBUG_VAR_WITH_NAME( "actGoKickDur+AddWait", ivActGoalKickDuration + additionalWait ) << " | " );
////    LOG( "   | " << DEBUG_VAR( cvDefaultWaitTime ) << " | " );
////    LOG( "   | " << DEBUG_VAR( goalDiff ) << " | " );
////    LOG( "   | " << DEBUG_VAR( ivStandardWaitTime ) << " | " );
////    LOG( "   | " << DEBUG_VAR( cvWaitOnLowStaminaTime ) << " | " );
//    LOG( "   | " << DEBUG_VAR( desiredWait ) << " | " );
//    LOG( "   | " << DEBUG_VAR( kick_time ) << " | " );
//    LOG( "   |-----------------------------|" );
//    LOG( "   | " << DEBUG_VAR_WITH_NAME( "ivTimeLeft", kick_time - WSinfo::ws->time )            << " | ");
//    LOG( "   |-----------------------------|" );
//    LOG( " " );

    return kick_time - WSinfo::ws->time;
}

bool MyGoalKick2016::getPassIntention( Intention &intention ) {
    bool returnValue = false;

    if( WSinfo::is_ball_kickable() )
    {
        ivpWBall->determine_pass_option( intention, WSinfo::me->pos, WSinfo::me->vel, WSinfo::me->ang, WSinfo::ball->pos, WSinfo::ball->vel );

        Vector target;
        intention.get_kick_info( cvDummyVal, target, cvDummyInt );

        bool intentionIsValid = intention.valid_at() == WSinfo::ws->time;
        bool targetIsNotInMyPenArea = !LEFT_PENALTY_AREA.inside( target );
        bool noOppsNearTarget = WSinfo::valid_opponents.keep_players_in_circle( target, 6.0 ).num == 0;

        if( intentionIsValid && targetIsNotInMyPenArea && noOppsNearTarget )
        {
            returnValue = true;
        }
    }
    return returnValue;
}
