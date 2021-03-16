#include "GoToPos2016_bms.h"

bool   GoToPos2016::cvInitialized = false;

double GoToPos2016::cvMaxDiffDefaultVal = 0.5; // Entspricht ungefähr 30 Grad
int    GoToPos2016::cvSpeedFindIteartionCount = 8;

#define G2P16LL 6

GoToPos2016::GoToPos2016()
{
    ivpBasicCmdBehavior = new BasicCmd();

    ivTargetSetAt       = -1;
    ivTargetFirstUse    = false;
    ivEstimatedAt       = -1;
    ivEstimatedSteps    = -1;
    ivTolerance         = 1.0;
    ivUseBackDashes     = false;

    ivTurnRads         = 0.0;
    ivTurnBackRads     = 0.0;
    ivMomentToTurnReal = 0.0;

    ivDummyAng = 0.0;

    ivMinSpeed = -100;
    ivMidSpeed =    0;
    ivMaxSpeed =  100;

    ivFirstRun = true;
    ivFound    = false;
    ivOptFound = -1;

    resetAngleDiffToDefault();
}

GoToPos2016::~GoToPos2016()
{
    if (ivpBasicCmdBehavior) delete ivpBasicCmdBehavior;
}

bool GoToPos2016::init(char const * conf_file, int argc, char const* const* argv)
{
    if( !cvInitialized )
    {
        bool result = true;

        result &= BasicCmd::init( conf_file, argc, argv );

        cvInitialized = result;
    }

    return cvInitialized;
}

void GoToPos2016::set_target( const Vector &target )
{
    set_target(target, 1.0, false);
}

void GoToPos2016::set_target( const Vector &target, bool useBackDashes )
{
    set_target(target, 1.0, useBackDashes);
}

void GoToPos2016::set_target( const Vector &target, double tolerance, bool useBackDashes )
{
    ivTargetSetAt    = WSinfo::ws->time;
    ivTargetFirstUse = true;
    ivTarget         = target;
    ivTolerance      = tolerance;
    ivUseBackDashes  = useBackDashes;
}

Vector* GoToPos2016::get_target( bool useTargetMoreThanOnceInSameCycle )
{
    Vector *retVal = 0;
    if( is_target_valid( useTargetMoreThanOnceInSameCycle ) )
    {
        retVal = &ivTarget;
    }
    return retVal;
}

bool GoToPos2016::is_target_valid( bool useTargetMoreThanOnceInSameCycle )
{
    bool retVal = false;
    if( ivTargetSetAt != WSinfo::ws->time )
    {
        LOG_POL( 0, << "GoToPos2016: Target has not been set!" );
    }
    else
    {
        if( !ivTargetFirstUse && !useTargetMoreThanOnceInSameCycle )
        {
            LOG_POL( 0, << "GoToPos2016: Target has already been used! Please set a new target or set the useTargetMoreThanOnceInSameCycle-Parameter true" );
        }
        else{
            if( !getPermissionToGoToAlreadyReachedTarget() && WSinfo::me->pos.distance( ivTarget ) < ivTolerance )
            {
                LOG_POL( 0, << "GoToPos2016: Target already reached!" );
            }
            else
            {
                retVal = true;
            }
        }
    }
    return retVal;
}

bool GoToPos2016::get_cmd( Cmd &cmd )
{
    return get_cmd( cmd, false );
}

bool GoToPos2016::get_cmd( Cmd &cmd, bool useTargetMoreThanOnceInSameCycle )
{
    bool retVal = false;

    if( estimate_duration( useTargetMoreThanOnceInSameCycle ) != -1 )
    {
        cmd.cmd_body = ivCmdToUse;

        retVal = true;
    }

    return retVal;
}

bool GoToPos2016::get_cmd_go_to( Cmd &cmd, const Vector &target, double tolerance, bool useBackDashes, bool useTargetMoreThanOnceInSameCycle )
{
    set_target( target, tolerance, useBackDashes );
    return get_cmd( cmd, useTargetMoreThanOnceInSameCycle );
}

int GoToPos2016::estimate_duration( bool useTargetMoreThanOnceInSameCycle )
{
    return estimate_duration_from_player_to_target( WSinfo::me, ivTarget, ivTolerance, ivUseBackDashes, useTargetMoreThanOnceInSameCycle );
}

int GoToPos2016::estimate_duration_to_target( const Vector &target, double tolerance, bool useBackDashes, bool useTargetMoreThanOnceInSameCycle )
{
    return estimate_duration_from_player_to_target( WSinfo::me, target, tolerance, useBackDashes, useTargetMoreThanOnceInSameCycle );
}

int GoToPos2016::estimate_duration_from_player_to_target( PPlayer &player, const Vector &target, double tolerance, bool useBackDashes, bool useTargetMoreThanOnceInSameCycle )
{
    int retVal = -1;

    if( ivEstimatedAt != WSinfo::ws->time )
    {
        // reset Cache
        ivEstimatedAt = WSinfo::ws->time;
    }

    set_target( target, tolerance, useBackDashes );
    if( is_target_valid( useTargetMoreThanOnceInSameCycle ) )
    {
        //if(cached)
        //then return cached Vals
        //else clac

        for(int i = 0; i < (ivUseBackDashes ? 3 : 2); i++)
        {
            ivOptUseful[i] = true;

            ivPlayerPosOpt[i] = player->pos;
            ivPlayerVelOpt[i] = player->vel;
            ivPlayerAngOpt[i] = player->ang.get_value();
        }

        ivLastOpt0PlayerPos = player->pos;

        ivFirstRun = true;
        ivFound = false;
        ivOptFound = -1;

        for( ivEstimatedSteps = 1; !ivFound; ivEstimatedSteps++ )
        {
            LOG_POL(G2P16LL, "GoToPos2016:ivEstimatedSteps="<< ivEstimatedSteps);
            if( ivOptUseful[0] )
            {
                //-----------------------------------------------------
                //------- Option No. 0 --- Turn and Dash! -------------
                //-----------------------------------------------------
                ivCmdOpt[0].cmd_body.unset_lock();
                ivCmdOpt[0].cmd_body.unset_cmd();

                ivAngToTargetOpt[0] = ( target - ivPlayerPosOpt[0]).ARG() - ANGLE(ivPlayerAngOpt[0] );
                ivTurnRads = ivAngToTargetOpt[0].get_value_mPI_pPI();

                if(fabs(ivTurnRads) > ivMaxDiff)
                {
                    ivMomentToTurnReal = ivTurnRads * ( 1.0 + ( player->inertia_moment * ivPlayerVelOpt[0].norm() ) );
                    if(ivMomentToTurnReal >  3.14) { ivMomentToTurnReal =  3.14; }
                    if(ivMomentToTurnReal < -3.14) { ivMomentToTurnReal = -3.14; }
                    ivpBasicCmdBehavior->set_turn( Tools::get_angle_between_null_2PI(ivMomentToTurnReal) );
                }
                else
                {
                    if( ivPlayerPosOpt[0].distance(target) > player->speed_max )
                    {
                        ivpBasicCmdBehavior->set_dash(100);
                    }
                    else{
                        ivSpeedCmd.cmd_body.unset_lock();
                        ivSpeedCmd.cmd_body.unset_cmd();
                        ivpBasicCmdBehavior->set_dash(98);
                        ivpBasicCmdBehavior->get_cmd(ivSpeedCmd);
                        Tools::model_player_movement(
                                ivPlayerPosOpt[0],
                                ivPlayerVelOpt[0],
                                ivPlayerAngOpt[0],
                                ivSpeedCmd.cmd_body,
                                ivSpeedPos,
                                ivDummyVec,
                                ivDummyAng);
                        ivSpeedDist[0] = ivSpeedPos.distance(target);

                        ivSpeedCmd.cmd_body.unset_lock();
                        ivSpeedCmd.cmd_body.unset_cmd();
                        ivpBasicCmdBehavior->set_dash(100);
                        ivpBasicCmdBehavior->get_cmd(ivSpeedCmd);
                        Tools::model_player_movement(
                                ivPlayerPosOpt[0],
                                ivPlayerVelOpt[0],
                                ivPlayerAngOpt[0],
                                ivSpeedCmd.cmd_body,
                                ivSpeedPos,
                                ivDummyVec,
                                ivDummyAng);
                        ivSpeedDist[1] = ivSpeedPos.distance(target);

                        if(ivSpeedDist[0] > ivSpeedDist[1]
                                                    && ((ivPlayerPosOpt[0].getX() < target.getX() ? ivSpeedPos.getX() < target.getX() : ivSpeedPos.getX() > target.getX())
                                                            && (ivPlayerPosOpt[0].getY() < target.getY() ? ivSpeedPos.getY() < target.getY() : ivSpeedPos.getY() > target.getY())))
                        {
                            ivpBasicCmdBehavior->set_dash(100);
                        }
                        else
                        {
                            ivMinSpeed = -100;
                            ivMidSpeed =    0;
                            ivMaxSpeed =  100;

                            for(int i = 0; i < cvSpeedFindIteartionCount; i++)
                            {
                                ivSpeedCmd.cmd_body.unset_lock();
                                ivSpeedCmd.cmd_body.unset_cmd();
                                ivpBasicCmdBehavior->set_dash(ivMinSpeed);
                                ivpBasicCmdBehavior->get_cmd(ivSpeedCmd);
                                Tools::model_player_movement(
                                        ivPlayerPosOpt[0],
                                        ivPlayerVelOpt[0],
                                        ivPlayerAngOpt[0],
                                        ivSpeedCmd.cmd_body,
                                        ivSpeedFindPos[0],
                                        ivDummyVec,
                                        ivDummyAng);

                                ivSpeedCmd.cmd_body.unset_lock();
                                ivSpeedCmd.cmd_body.unset_cmd();
                                ivpBasicCmdBehavior->set_dash(ivMidSpeed);
                                ivpBasicCmdBehavior->get_cmd(ivSpeedCmd);
                                Tools::model_player_movement(
                                        ivPlayerPosOpt[0],
                                        ivPlayerVelOpt[0],
                                        ivPlayerAngOpt[0],
                                        ivSpeedCmd.cmd_body,
                                        ivSpeedFindPos[1],
                                        ivDummyVec,
                                        ivDummyAng);

                                ivSpeedCmd.cmd_body.unset_lock();
                                ivSpeedCmd.cmd_body.unset_cmd();
                                ivpBasicCmdBehavior->set_dash(ivMaxSpeed);
                                ivpBasicCmdBehavior->get_cmd(ivSpeedCmd);
                                Tools::model_player_movement(
                                        ivPlayerPosOpt[0],
                                        ivPlayerVelOpt[0],
                                        ivPlayerAngOpt[0],
                                        ivSpeedCmd.cmd_body,
                                        ivSpeedFindPos[2],
                                        ivDummyVec,
                                        ivDummyAng);

                                if(
                                        (ivSpeedFindPos[0].distance(target) >= ivSpeedFindPos[1].distance(target)) &&
                                        (ivSpeedFindPos[0].distance(target) >= ivSpeedFindPos[2].distance(target)))
                                {
                                    ivMinSpeed = ivMidSpeed;
                                    ivMidSpeed = ivMinSpeed + ((ivMaxSpeed - ivMinSpeed) / 2);
                                }
                                else if(
                                        (ivSpeedFindPos[2].distance(target) >= ivSpeedFindPos[1].distance(target)) &&
                                        (ivSpeedFindPos[2].distance(target) >= ivSpeedFindPos[0].distance(target)))
                                {
                                    ivMaxSpeed = ivMidSpeed;
                                    ivMidSpeed = ivMinSpeed + ((ivMaxSpeed - ivMinSpeed) / 2);
                                }
                                else
                                {
                                    LOG_POL(G2P16LL, "GoToPos2016:ERROR! DARF NICHT AUFTRETEN 0 " << i );
                                    LOG_POL(G2P16LL, "GoToPos2016:speedFindPos[0].distance(target)=" << ivSpeedFindPos[0].distance(target) << " minSpeed = " << ivMinSpeed);
                                    LOG_POL(G2P16LL, "GoToPos2016:speedFindPos[1].distance(target)=" << ivSpeedFindPos[1].distance(target) << " midSpeed = " << ivMidSpeed);
                                    LOG_POL(G2P16LL, "GoToPos2016:speedFindPos[2].distance(target)=" << ivSpeedFindPos[2].distance(target) << " maxSpeed = " << ivMaxSpeed);
                                }
                            }
                            LOG_POL(G2P16LL, "GoToPos2016:Calced Dash-Power = " << ivMidSpeed);
                            ivpBasicCmdBehavior->set_dash(ivMidSpeed);
                        }
                    }
                    if( ivOptUseful[2] )
                    {
                        ivOptUseful[2] = false;
                        LOG_POL(G2P16LL, "GoToPos2016:Opt2 isn't useful anymore!");
                    }
                }
                ivpBasicCmdBehavior->get_cmd(ivCmdOpt[0]);

                if(ivFirstRun){
                    ivCmdOnFirstRun[0] = ivCmdOpt[0];
                }

                ivLastOpt0PlayerPos = ivPlayerPosOpt[0];

                Tools::model_player_movement(
                        ivPlayerPosOpt[0],
                        ivPlayerVelOpt[0],
                        ivPlayerAngOpt[0],
                        ivCmdOpt[0].cmd_body,
                        ivPlayerPosOpt[0],
                        ivPlayerVelOpt[0],
                        ivPlayerAngOpt[0]);

                if(ivPlayerPosOpt[0].distance(target) < tolerance)
                {
                    ivFound = true;
                    ivOptFound = 0;
                    LOG_POL(G2P16LL, "GoToPos2016:Use=TURN_DASH - "<<((ivCmdOnFirstRun[0].cmd_body.get_type() == ivCmdOnFirstRun[0].cmd_body.TYPE_TURN) ? "TURN" : "DASH"));
                    break;
                }
            }
            else
            {
                LOG_POL(6, "GoToPos2016:Opt0 isn't calced!");
            }

            if( ivOptUseful[1] )
            {
                //-----------------------------------------------------
                //------- Option No. 1 --- Side-Dash! -----------------
                //-----------------------------------------------------
                ivCmdOpt[1].cmd_body.unset_lock();
                ivCmdOpt[1].cmd_body.unset_cmd();

                ivAngToTargetOpt[1] = ( target - ivPlayerPosOpt[1]).ARG() - ANGLE(ivPlayerAngOpt[1] );

                if( ivPlayerPosOpt[1].distance(target) > player->speed_max )
                {
                    ivpBasicCmdBehavior->set_dash(100, ivAngToTargetOpt[1]);
                }
                else{
                    ivSpeedCmd.cmd_body.unset_lock();
                    ivSpeedCmd.cmd_body.unset_cmd();
                    ivpBasicCmdBehavior->set_dash(98, ivAngToTargetOpt[1]);
                    ivpBasicCmdBehavior->get_cmd(ivSpeedCmd);
                    Tools::model_player_movement(
                            ivPlayerPosOpt[1],
                            ivPlayerVelOpt[1],
                            ivPlayerAngOpt[1],
                            ivSpeedCmd.cmd_body,
                            ivSpeedPos,
                            ivDummyVec,
                            ivDummyAng);
                    ivSpeedDist[0] = ivSpeedPos.distance(target);

                    ivSpeedCmd.cmd_body.unset_lock();
                    ivSpeedCmd.cmd_body.unset_cmd();
                    ivpBasicCmdBehavior->set_dash(100, ivAngToTargetOpt[1]);
                    ivpBasicCmdBehavior->get_cmd(ivSpeedCmd);
                    Tools::model_player_movement(
                            ivPlayerPosOpt[1],
                            ivPlayerVelOpt[1],
                            ivPlayerAngOpt[1],
                            ivSpeedCmd.cmd_body,
                            ivSpeedPos,
                            ivDummyVec,
                            ivDummyAng);
                    ivSpeedDist[1] = ivSpeedPos.distance(target);

                    if(ivSpeedDist[0] > ivSpeedDist[1]
                                                && ((ivPlayerPosOpt[1].getX() < target.getX() ? ivSpeedPos.getX() < target.getX() : ivSpeedPos.getX() > target.getX())
                                                        && (ivPlayerPosOpt[1].getY() < target.getY() ? ivSpeedPos.getY() < target.getY() : ivSpeedPos.getY() > target.getY())))
                    {
                        ivpBasicCmdBehavior->set_dash(100, ivAngToTargetOpt[1]);
                    }
                    else
                    {
                        ivMinSpeed = -100;
                        ivMidSpeed =    0;
                        ivMaxSpeed =  100;

                        for(int i = 0; i < cvSpeedFindIteartionCount; i++)
                        {
                            ivSpeedCmd.cmd_body.unset_lock();
                            ivSpeedCmd.cmd_body.unset_cmd();
                            ivpBasicCmdBehavior->set_dash(ivMinSpeed, ivAngToTargetOpt[1]);
                            ivpBasicCmdBehavior->get_cmd(ivSpeedCmd);
                            Tools::model_player_movement(
                                    ivPlayerPosOpt[1],
                                    ivPlayerVelOpt[1],
                                    ivPlayerAngOpt[1],
                                    ivSpeedCmd.cmd_body,
                                    ivSpeedFindPos[0],
                                    ivDummyVec,
                                    ivDummyAng);

                            ivSpeedCmd.cmd_body.unset_lock();
                            ivSpeedCmd.cmd_body.unset_cmd();
                            ivpBasicCmdBehavior->set_dash(ivMidSpeed, ivAngToTargetOpt[1]);
                            ivpBasicCmdBehavior->get_cmd(ivSpeedCmd);
                            Tools::model_player_movement(
                                    ivPlayerPosOpt[1],
                                    ivPlayerVelOpt[1],
                                    ivPlayerAngOpt[1],
                                    ivSpeedCmd.cmd_body,
                                    ivSpeedFindPos[1],
                                    ivDummyVec,
                                    ivDummyAng);

                            ivSpeedCmd.cmd_body.unset_lock();
                            ivSpeedCmd.cmd_body.unset_cmd();
                            ivpBasicCmdBehavior->set_dash(ivMaxSpeed, ivAngToTargetOpt[1]);
                            ivpBasicCmdBehavior->get_cmd(ivSpeedCmd);
                            Tools::model_player_movement(
                                    ivPlayerPosOpt[1],
                                    ivPlayerVelOpt[1],
                                    ivPlayerAngOpt[1],
                                    ivSpeedCmd.cmd_body,
                                    ivSpeedFindPos[2],
                                    ivDummyVec,
                                    ivDummyAng);

                            if(
                                    (ivSpeedFindPos[0].distance(target) >= ivSpeedFindPos[1].distance(target)) &&
                                    (ivSpeedFindPos[0].distance(target) >= ivSpeedFindPos[2].distance(target)))
                            {
                                ivMinSpeed = ivMidSpeed;
                                ivMidSpeed = ivMinSpeed + ((ivMaxSpeed - ivMinSpeed) / 2);
                            }
                            else if(
                                    (ivSpeedFindPos[2].distance(target) >= ivSpeedFindPos[1].distance(target)) &&
                                    (ivSpeedFindPos[2].distance(target) >= ivSpeedFindPos[0].distance(target)))
                            {
                                ivMaxSpeed = ivMidSpeed;
                                ivMidSpeed = ivMinSpeed + ((ivMaxSpeed - ivMinSpeed) / 2);
                            }
                            else
                            {
                                LOG_POL(G2P16LL, "GoToPos2016:ERROR! DARF NICHT AUFTRETEN 1 " << i );
                                LOG_POL(G2P16LL, "GoToPos2016:speedFindPos[0].distance(target)=" << ivSpeedFindPos[0].distance(target) << " minSpeed = " << ivMinSpeed);
                                LOG_POL(G2P16LL, "GoToPos2016:speedFindPos[1].distance(target)=" << ivSpeedFindPos[1].distance(target) << " midSpeed = " << ivMidSpeed);
                                LOG_POL(G2P16LL, "GoToPos2016:speedFindPos[2].distance(target)=" << ivSpeedFindPos[2].distance(target) << " maxSpeed = " << ivMaxSpeed);
                            }
                        }
                        LOG_POL(G2P16LL, "GoToPos2016:Calced Dash-Power = " << ivMidSpeed);
                        ivpBasicCmdBehavior->set_dash(ivMidSpeed, ivAngToTargetOpt[1]);
                    }
                }
                ivpBasicCmdBehavior->get_cmd(ivCmdOpt[1]);

                if(ivFirstRun){
                    ivCmdOnFirstRun[1] = ivCmdOpt[1];
                }

                Tools::model_player_movement(
                        ivPlayerPosOpt[1],
                        ivPlayerVelOpt[1],
                        ivPlayerAngOpt[1],
                        ivCmdOpt[1].cmd_body,
                        ivPlayerPosOpt[1],
                        ivPlayerVelOpt[1],
                        ivPlayerAngOpt[1]);

                if(ivPlayerPosOpt[1].distance(target) < tolerance)
                {
                    ivFound = true;
                    ivOptFound = 1;
                    LOG_POL(G2P16LL, "GoToPos2016:Use=SIDE_DASH");
                    break;
                }

                if( ivLastOpt0PlayerPos.distance( target ) < ivPlayerPosOpt[1].distance( target ) )
                {
                    ivOptUseful[1] = false;
                    LOG_POL(G2P16LL, "GoToPos2016:Opt1 isn't useful anymore!");
                }
            }
            else{
                LOG_POL(G2P16LL, "GoToPos2016:Opt1 isn't calced!");
            }

            if( ivUseBackDashes )
            {
                if( ivOptUseful[2] )
                {
                    //-------------------------------------------------
                    //--- Option No. 2 --- Turn and Dash backwards! ---
                    //-------------------------------------------------
                    ivCmdOpt[2].cmd_body.unset_lock();
                    ivCmdOpt[2].cmd_body.unset_cmd();

                    ivAngToTargetOpt[2] = ( target - ivPlayerPosOpt[2]).ARG() - ANGLE(ivPlayerAngOpt[2] );
                    ivTurnBackRads = Vector(ivAngToTargetOpt[2]).rotate(M_PI).ARG().get_value_mPI_pPI();

                    if(fabs(ivTurnBackRads) > ivMaxDiff)
                    {
                        ivMomentToTurnReal = ivTurnBackRads * ( 1.0 + ( player->inertia_moment * ivPlayerVelOpt[2].norm() ) );
                        if(ivMomentToTurnReal >  3.14) { ivMomentToTurnReal =  3.14; }
                        if(ivMomentToTurnReal < -3.14) { ivMomentToTurnReal = -3.14; }
                        ivpBasicCmdBehavior->set_turn( Tools::get_angle_between_null_2PI(ivMomentToTurnReal) );
                    }
                    else
                    {
                        if( ivPlayerPosOpt[2].distance(target) > player->speed_max )
                        {
                            ivpBasicCmdBehavior->set_dash(-100);
                        }
                        else{
                            ivSpeedCmd.cmd_body.unset_lock();
                            ivSpeedCmd.cmd_body.unset_cmd();
                            ivpBasicCmdBehavior->set_dash(-98);
                            ivpBasicCmdBehavior->get_cmd(ivSpeedCmd);
                            Tools::model_player_movement(
                                    ivPlayerPosOpt[2],
                                    ivPlayerVelOpt[2],
                                    ivPlayerAngOpt[2],
                                    ivSpeedCmd.cmd_body,
                                    ivSpeedPos,
                                    ivDummyVec,
                                    ivDummyAng);
                            ivSpeedDist[0] = ivSpeedPos.distance(target);

                            ivSpeedCmd.cmd_body.unset_lock();
                            ivSpeedCmd.cmd_body.unset_cmd();
                            ivpBasicCmdBehavior->set_dash(-100);
                            ivpBasicCmdBehavior->get_cmd(ivSpeedCmd);
                            Tools::model_player_movement(
                                    ivPlayerPosOpt[2],
                                    ivPlayerVelOpt[2],
                                    ivPlayerAngOpt[2],
                                    ivSpeedCmd.cmd_body,
                                    ivSpeedPos,
                                    ivDummyVec,
                                    ivDummyAng);
                            ivSpeedDist[1] = ivSpeedPos.distance(target);

                            if(ivSpeedDist[0] > ivSpeedDist[1]
                                                        && ((ivPlayerPosOpt[2].getX() < target.getX() ? ivSpeedPos.getX() < target.getX() : ivSpeedPos.getX() > target.getX())
                                                                && (ivPlayerPosOpt[2].getY() < target.getY() ? ivSpeedPos.getY() < target.getY() : ivSpeedPos.getY() > target.getY())))
                            {
                                ivpBasicCmdBehavior->set_dash(-100);
                            }
                            else
                            {
                                ivMinSpeed = -100;
                                ivMidSpeed =    0;
                                ivMaxSpeed =  100;

                                for(int i = 0; i < cvSpeedFindIteartionCount; i++)
                                {
                                    ivSpeedCmd.cmd_body.unset_lock();
                                    ivSpeedCmd.cmd_body.unset_cmd();
                                    ivpBasicCmdBehavior->set_dash(ivMinSpeed);
                                    ivpBasicCmdBehavior->get_cmd(ivSpeedCmd);
                                    Tools::model_player_movement(
                                            ivPlayerPosOpt[2],
                                            ivPlayerVelOpt[2],
                                            ivPlayerAngOpt[2],
                                            ivSpeedCmd.cmd_body,
                                            ivSpeedFindPos[0],
                                            ivDummyVec,
                                            ivDummyAng);

                                    ivSpeedCmd.cmd_body.unset_lock();
                                    ivSpeedCmd.cmd_body.unset_cmd();
                                    ivpBasicCmdBehavior->set_dash(ivMidSpeed);
                                    ivpBasicCmdBehavior->get_cmd(ivSpeedCmd);
                                    Tools::model_player_movement(
                                            ivPlayerPosOpt[2],
                                            ivPlayerVelOpt[2],
                                            ivPlayerAngOpt[2],
                                            ivSpeedCmd.cmd_body,
                                            ivSpeedFindPos[1],
                                            ivDummyVec,
                                            ivDummyAng);

                                    ivSpeedCmd.cmd_body.unset_lock();
                                    ivSpeedCmd.cmd_body.unset_cmd();
                                    ivpBasicCmdBehavior->set_dash(ivMaxSpeed);
                                    ivpBasicCmdBehavior->get_cmd(ivSpeedCmd);
                                    Tools::model_player_movement(
                                            ivPlayerPosOpt[2],
                                            ivPlayerVelOpt[2],
                                            ivPlayerAngOpt[2],
                                            ivSpeedCmd.cmd_body,
                                            ivSpeedFindPos[2],
                                            ivDummyVec,
                                            ivDummyAng);

                                    if(
                                            (ivSpeedFindPos[0].distance(target) >= ivSpeedFindPos[1].distance(target)) &&
                                            (ivSpeedFindPos[0].distance(target) >= ivSpeedFindPos[2].distance(target)))
                                    {
                                        ivMinSpeed = ivMidSpeed;
                                        ivMidSpeed = ivMinSpeed + ((ivMaxSpeed - ivMinSpeed) / 2);
                                    }
                                    else if(
                                            (ivSpeedFindPos[2].distance(target) > ivSpeedFindPos[1].distance(target)) &&
                                            (ivSpeedFindPos[2].distance(target) > ivSpeedFindPos[0].distance(target)))
                                    {
                                        ivMaxSpeed = ivMidSpeed;
                                        ivMidSpeed = ivMinSpeed + ((ivMaxSpeed - ivMinSpeed) / 2);
                                    }
                                    else
                                    {
                                        LOG_POL(G2P16LL, "GoToPos2016:ERROR! DARF NICHT AUFTRETEN 2 " << i );
                                        LOG_POL(G2P16LL, "GoToPos2016:speedFindPos[0].distance(target)=" << ivSpeedFindPos[0].distance(target) << " minSpeed = " << ivMinSpeed);
                                        LOG_POL(G2P16LL, "GoToPos2016:speedFindPos[1].distance(target)=" << ivSpeedFindPos[1].distance(target) << " midSpeed = " << ivMidSpeed);
                                        LOG_POL(G2P16LL, "GoToPos2016:speedFindPos[2].distance(target)=" << ivSpeedFindPos[2].distance(target) << " maxSpeed = " << ivMaxSpeed);
                                    }
                                }
                                LOG_POL(G2P16LL, "GoToPos2016:Calced Dash-Power = " << ivMidSpeed);
                                ivpBasicCmdBehavior->set_dash(ivMidSpeed);
                            }
                        }
                        if( ivOptUseful[0] )
                        {
                            ivOptUseful[0] = false;
                            LOG_POL(G2P16LL, "GoToPos2016:Opt0 isn't useful anymore!");
                        }
                    }
                    ivpBasicCmdBehavior->get_cmd(ivCmdOpt[2]);

                    if(ivFirstRun){
                        ivCmdOnFirstRun[2] = ivCmdOpt[2];
                    }

                    Tools::model_player_movement(
                            ivPlayerPosOpt[2],
                            ivPlayerVelOpt[2],
                            ivPlayerAngOpt[2],
                            ivCmdOpt[2].cmd_body,
                            ivPlayerPosOpt[2],
                            ivPlayerVelOpt[2],
                            ivPlayerAngOpt[2]);

                    if(ivPlayerPosOpt[2].distance(target) < tolerance)
                    {
                        ivFound = true;
                        ivOptFound = 2;
                        LOG_POL(G2P16LL, "GoToPos2016:Use=TURN_BACKDASH - "<<((ivCmdOnFirstRun[2].cmd_body.get_type() == ivCmdOnFirstRun[2].cmd_body.TYPE_TURN) ? "TURN" : "DASH"));
                        break;
                    }
                }
                else
                {
                    LOG_POL(G2P16LL, "GoToPos2016:Opt2 isn't calced!");
                }
            }

            ivFirstRun = false;
        }
        ivEstimatedSteps--;

        ivCmdToUse = ivCmdOnFirstRun[ivOptFound].cmd_body;

        retVal = ivEstimatedSteps;
    }
    return retVal;
}

void GoToPos2016::setAngleDiff( double newAngleDiff)
{
    this->ivMaxDiff = newAngleDiff;
}

double GoToPos2016::getAngleDiff()
{
    return ivMaxDiff;
}

void GoToPos2016::resetAngleDiffToDefault()
{
    this->ivMaxDiff = this->cvMaxDiffDefaultVal;
}

void GoToPos2016::setPermissionToGoToAlreadyReachedTarget( bool allow )
{
    if( allow )
    {
        this->ivAllowAlreadyReachedTarget = allow;
        this->ivAllowAlreadyReachedTargetTimeStep = WSinfo::ws->time;
    }
    else
    {
        resetPermissionToGoToAlreadyReachedTarget();
    }
}

bool GoToPos2016::getPermissionToGoToAlreadyReachedTarget()
{
    if( this->ivAllowAlreadyReachedTargetTimeStep == WSinfo::ws->time )
    {
        return this->ivAllowAlreadyReachedTarget;
    }
    else
    {
        return false;
    }
}

void GoToPos2016::resetPermissionToGoToAlreadyReachedTarget()
{
    this->ivAllowAlreadyReachedTarget = false;
}

void GoToPos2016::resetAllCustomizations()
{
    resetAngleDiffToDefault();
    resetPermissionToGoToAlreadyReachedTarget();
}
