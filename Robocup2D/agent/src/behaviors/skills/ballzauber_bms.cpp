#include "ballzauber_bms.h"
#include "dribble_around06.h"

#if 0
#define   TGbzLOGPOL(YYY,XXX)        LOG_POL(YYY,XXX)
#else
#define   TGbzLOGPOL(YYY,XXX)
#endif

bool Ballzauber::cvInitialized = false;

Ballzauber::Ballzauber()
{
  ivpBasicCmdBehavior        = new BasicCmd();
  ivpOneStepKickBehavior     = new OneStepKick();
  ivLastResult = false;
  ivLastTimeCalled = -1;
  ivNumberOfStepsToTurnToTargetDirection = UNABLE_TO_HOLD_BALL;
  ivSafetyWithRespectToNearbyOpponents = false;
}
Ballzauber::~Ballzauber()
{
  if (ivpBasicCmdBehavior) delete ivpBasicCmdBehavior;
  if (ivpOneStepKickBehavior) delete ivpOneStepKickBehavior;
}

bool 
Ballzauber::init(char const * confFile, int argc, char const* const* argv) 
{
  if ( cvInitialized )
    return true;
  cvInitialized = 
       BasicCmd::init(confFile, argc, argv)
    && OneStepKick::init(confFile, argc, argv);
  return cvInitialized;
}

bool
Ballzauber::checkForTermination()
{
  if (     fabs((ivTargetDirection - WSinfo::me->ang).get_value_mPI_pPI()) 
         < MAX_DEVIATION
      && ivBallInRelaxedTargetAreaInTSteps[0] == true )
  {
    TGbzLOGPOL(0,"Ballzauber: FINISHED, Ball erfolgreich vorgelegt! "
      <<"Bei Weiterverwendung von BALL~ZAUBER~ agiert es wie ^HOLD^TURN!");
    return true;
  }
  return false;
}

bool
Ballzauber
  ::createActionToKeepBallSomehowInKickrange()
{
  Vector basePoint = ivMyPositionInTSteps[ 1 ];
  double bestKickVelocityValue = 0;
  Vector bestKickVelocity;
  for ( int i=0; i<360; i+=20)
  {
    for ( int j=60; j<=(int)((99.0/100.0)*(100.0*WSinfo::me->kick_radius)); j+=5 )
    {
      Vector targetDirectionVector
             ( ivTargetDirection + ANGLE( 2.0*PI*((double)i/360.0) ) );
      targetDirectionVector.normalize( (double)j/100.0 );
      Vector kickTarget = basePoint + targetDirectionVector;

      Vector kickVelocity = kickTarget - ivBallPositionInTSteps[0];
      double maxVel = getMaximalVelocityWhenKicking( ivMyPositionInTSteps[0],
                                          ivMyVelocityInTSteps[0],
                                          WSinfo::me->ang,
                                          ivBallPositionInTSteps[0],
                                          ivBallVelocityInTSteps[0],
                                          kickTarget ),
            kickVelocityValue;
      if (    maxVel >= kickVelocity.norm()
           && isBallPositionSafe( kickTarget ) )
      {
        kickVelocityValue 
          =   0.6 * (WSinfo::me->kick_radius - kickTarget.distance( basePoint ))
            + 0.4 * (ServerOptions::ball_speed_max - kickVelocity.norm() );
        if ( kickVelocityValue > bestKickVelocityValue )
        {
          bestKickVelocityValue = kickVelocityValue;
          bestKickVelocity = kickVelocity;
          TGbzLOGPOL(2,<<"Ballzauber: i="<<i<<" j="<<j<<": kickTgt="<<kickTarget
            <<" kickVel="<<kickVelocity<<" maxVel="<<maxVel<<" -> v="
            <<kickVelocityValue<<" (new best!)");
        }
        else
        {
          TGbzLOGPOL(3,<<"Ballzauber: i="<<i<<" j="<<j<<": kickTgt="<<kickTarget
            <<" kickVel="<<kickVelocity<<" maxVel="<<maxVel<<" -> v="
            <<kickVelocityValue<<" (bestV="<<bestKickVelocityValue<<")");
        }
      }
      else
      {
        TGbzLOGPOL(3,<<"Ballzauber: i="<<i<<" j="<<j<<": kickTgt="<<kickTarget
          <<" kickVel="<<kickVelocity<<" maxVel="<<maxVel<<" -> maxVel not enough");
      }
    }
  }
  if ( bestKickVelocityValue > 0.0 )
  {
    TGbzLOGPOL(0,<<"Ballzauber: createAct keepBallInKickRng: YEP - "
      <<" target for kick is "<<(ivBallPositionInTSteps[0]+bestKickVelocity)
      <<" and v="<<bestKickVelocity<<" dist from player in t+1 = "
      <<((ivBallPositionInTSteps[0]+bestKickVelocity).distance( basePoint ))<<std::flush);
    TGbzLOGPOL(0,<<_2D<<C2D((ivBallPositionInTSteps[0]+bestKickVelocity).x,
      (ivBallPositionInTSteps[0]+bestKickVelocity).y,0.085,"A52A2A"));
    ivAction = BALLZAUBER_KICK;
    ivActionKickVelocity = bestKickVelocity;
    ivPlanStringStream << "KICK[keepBallInKRange] ";
    ivPlanStringStream << "KICK[final] ";
    for (int i=1; i<=ivRequiredTurnCommandsWhenStartingTurningInTSteps[2]; i++)
      ivPlanStringStream << "TURN["<<i<<"] ";
    ivPlanStringStream << "=> "
      <<(2+ivRequiredTurnCommandsWhenStartingTurningInTSteps[2])<<" STEPS ";
    ivNumberOfStepsToTurnToTargetDirection
      = 2+ivRequiredTurnCommandsWhenStartingTurningInTSteps[2];
    return true;
  }
  TGbzLOGPOL(0,<<"Ballzauber: createAct keepBallInKickRng: NOPE"<<std::flush);
  return false;
}

bool
Ballzauber
  ::createActionToKickBallToTargetAreaInAsManyStepsINeedThenForTurning()
{
  //try it with a one step kick
  int necessaryTurnCmdsWhenStartingTurningInT1
    = ivRequiredTurnCommandsWhenStartingTurningInTSteps[1];
  int stepsForBallToTargetArea 
    = necessaryTurnCmdsWhenStartingTurningInT1 + 1;
  //iterate over possible target points
  Vector basePoint = ivMyPositionInTSteps[ stepsForBallToTargetArea ];
  Quadrangle2d finalTargetArea = ivTargetAreaInTSteps[ stepsForBallToTargetArea ];
  double bestKickVelocityValue = 10.0;
  Vector bestKickVelocity, bestKickTarget;
  for ( int i=0; i<360; i+=20)
  {
    for ( int j=45; j<(int)((90.0/100.0)*(100.0*WSinfo::me->kick_radius)); j+=2 )
    {
      bool discardThisTargetPoint = false;
      Vector targetDirectionVector
             ( ivTargetDirection + ANGLE( 2.0*PI*((double)i/360.0) ) );
      targetDirectionVector.normalize( (double)j/100.0 );
      Vector kickTarget = basePoint + targetDirectionVector;
      if (    finalTargetArea.inside( kickTarget )
           && isBallPositionSafe(kickTarget) )
      {
        double relativeBallVelocity  = 1.0,
              relativeWayGoneByBall = 0.0;
        int cnt = stepsForBallToTargetArea;
        while (cnt > 0)
        {
          relativeWayGoneByBall += relativeBallVelocity;
          relativeBallVelocity  *= ServerOptions::ball_decay;
          cnt -- ;
        }
        double realWayForBallToMove
          = ivBallPositionInTSteps[0].distance( kickTarget );
        double requiredInitialBallVelocity
          = (1.0 / relativeWayGoneByBall) * realWayForBallToMove;
        Vector initialBallVelocity
          = kickTarget - ivBallPositionInTSteps[0];
        initialBallVelocity.normalize( requiredInitialBallVelocity );
        Vector kickTargetForInitialKick
          = ivBallPositionInTSteps[0] + initialBallVelocity;
        
        if (    getMaximalVelocityWhenKicking( ivMyPositionInTSteps[0],
                                               ivMyVelocityInTSteps[0],
                                               WSinfo::me->ang,
                                               ivBallPositionInTSteps[0],
                                               ivBallVelocityInTSteps[0],
                                               kickTargetForInitialKick )
                >= initialBallVelocity.norm()
             && isBallPositionSafe(kickTargetForInitialKick) )
        {
          //simulate next ball positions
          Vector currentBallPos = ivBallPositionInTSteps[0],
                 currentBallVel = initialBallVelocity;
          for (int cnt=0; cnt<=stepsForBallToTargetArea; cnt++)
          {
            if ( currentBallPos.distance( ivMyPositionInTSteps[cnt] )
                 < 0.4 )
            {
              discardThisTargetPoint = true;
            }
            currentBallPos += currentBallVel;
            currentBallVel *= ServerOptions::ball_decay;
          }
          if ( discardThisTargetPoint == false )
          {
            double kickVelocityValue
              = kickTarget.distance
                ( ivTargetAreaCentralPointInTSteps[stepsForBallToTargetArea] );
            if (kickVelocityValue < bestKickVelocityValue)
            {
              bestKickVelocityValue = kickVelocityValue;
              bestKickVelocity = initialBallVelocity;
              bestKickTarget = kickTarget;
              TGbzLOGPOL(3,"Ballzauber: found new best at i="<<i<<", j="<<j
                <<" V="<<kickVelocityValue);
            }
          }
        }
        else
          discardThisTargetPoint = true;
      }
      else
        discardThisTargetPoint = true;
    }
  }
  if ( bestKickVelocityValue < 2.0 )
  {
    TGbzLOGPOL(0,"Ballzauber: createAct kickBall2TargetInNSteps: YEP"
      <<" - I found a kick by which the ball will be in the target"
      <<" area in "<<stepsForBallToTargetArea<<" steps.");
    Vector ballPos = ivBallPositionInTSteps[0],
           ballVel = bestKickVelocity;
    TGbzLOGPOL(0,<<_2D<<L2D(ballPos.x,ballPos.y,bestKickTarget.x,bestKickTarget.y,"ff0000"));
    for (int b=0; b<=stepsForBallToTargetArea; b++)
    {
      TGbzLOGPOL(0,<<_2D<<C2D(ballPos.x,ballPos.y,0.085,"ff0000"));
      TGbzLOGPOL(0,<<_2D<<C2D(ivMyPositionInTSteps[b].x,ivMyPositionInTSteps[b].y,0.3,"55ff55"));
      ballPos += ballVel;
      ballVel *= ServerOptions::ball_decay;
    }
    ivAction = BALLZAUBER_KICK;
    ivActionKickVelocity = bestKickVelocity;
    ivPlanStringStream << "KICK[final] ";
    for (int i=1; i<=ivRequiredTurnCommandsWhenStartingTurningInTSteps[1]; i++)
      ivPlanStringStream << "TURN["<<i<<"] ";
    ivPlanStringStream << "=> "
      <<(1+ivRequiredTurnCommandsWhenStartingTurningInTSteps[1])<<" STEPS ";
    ivNumberOfStepsToTurnToTargetDirection
      = 1+ivRequiredTurnCommandsWhenStartingTurningInTSteps[1];
    return true;
  }
  TGbzLOGPOL(0,"Ballzauber: createAct kickBall2TargetInNSteps: NOPE");
  return false;
}

bool
Ballzauber
  ::createActionToStopBallOrMoveItALittleTowardsTargetArea()
{
  Vector basePoint = ivMyPositionInTSteps[ 1 ];
  double bestKickVelocityValue = 0.0;
  Vector bestKickVelocity;
  for ( int i=0; i<360; i+=30)
  {
    if ( i > 130 && i < 230 ) continue;
    for ( int j=50; j<=(int)((75.0/100.0)*(100.0*WSinfo::me->kick_radius)); j+=5 )
    {
      Vector targetDirectionVector
             ( ivTargetDirection + ANGLE( 2.0*PI*((double)i/360.0) ) );
      targetDirectionVector.normalize( (double)j/100.0 );
      Vector kickTarget = basePoint + targetDirectionVector;

      Vector kickVelocity = kickTarget - ivBallPositionInTSteps[0];
      if (    getMaximalVelocityWhenKicking( ivMyPositionInTSteps[0],
                                             ivMyVelocityInTSteps[0],
                                             WSinfo::me->ang,
                                             ivBallPositionInTSteps[0],
                                             ivBallVelocityInTSteps[0],
                                             kickTarget )
              >= kickVelocity.norm() 
           && kickTarget.distance( ivMyPositionInTSteps[1] ) 
              > MIN_DISTANCE_TO_AVOID_COLLSIONS 
           && isBallPositionSafe(kickTarget) )
      {
        Vector goodPoint_1( ivTargetDirection + ANGLE(PI) ),
               goodPoint_2( ivTargetDirection + ANGLE(-PI) );
        goodPoint_1.normalize(0.6); goodPoint_2.normalize(0.6);
        goodPoint_1 += basePoint;    goodPoint_2 += basePoint;
        double dist2GoodPoint_1 = 2.0-goodPoint_1.distance(kickTarget),
              dist2GoodPoint_2 = 2.0-goodPoint_2.distance(kickTarget);
        
        double kickVelocityValue
          =   0.6 * (ServerOptions::ball_speed_max - kickVelocity.norm())
            + 0.4 * Tools::max(dist2GoodPoint_1,dist2GoodPoint_2);
        if ( kickVelocityValue > bestKickVelocityValue )
        {
          bestKickVelocityValue = kickVelocityValue;
          bestKickVelocity = kickVelocity;
        }
      }
    }
  }
  if ( bestKickVelocityValue > 0.0 )
  {
    TGbzLOGPOL(0,<<"Ballzauber: createAct stopBallOrMoveLitte: YEP - "
      <<" target for intermediate kick is "<<(ivBallPositionInTSteps[0]+bestKickVelocity)
      <<" and v="<<bestKickVelocity<<std::flush);
    TGbzLOGPOL(0,<<_2D<<C2D((ivBallPositionInTSteps[0]+bestKickVelocity).x,
      (ivBallPositionInTSteps[0]+bestKickVelocity).y,0.085,"A52A2A"));
    TGbzLOGPOL(0,<<_2D<<C2D(ivMyPositionInTSteps[1].x,ivMyPositionInTSteps[1].y,
      0.3,"55ff55"));
    ivAction = BALLZAUBER_KICK;
    ivActionKickVelocity = bestKickVelocity;
    ivPlanStringStream << "KICK[initial] ";
    ivPlanStringStream << "KICK[final] ";
    for (int i=1; i<=ivRequiredTurnCommandsWhenStartingTurningInTSteps[2]; i++)
      ivPlanStringStream << "TURN["<<i<<"] ";
    ivPlanStringStream << "=> "
      <<(2+ivRequiredTurnCommandsWhenStartingTurningInTSteps[2])<<" STEPS ";
    ivNumberOfStepsToTurnToTargetDirection
      = 2+ivRequiredTurnCommandsWhenStartingTurningInTSteps[2];
    return true;
  }
  TGbzLOGPOL(0,<<"Ballzauber: createAct stopBallOrMoveLitte: NOPE"<<std::flush);
  return false;
}

bool
Ballzauber
  ::createActionToTurnToTargetDirection()
{
  //check if turning to target direction now is possible!
  int necessaryTurnCmdsWhenStartingTurningNow
    = ivRequiredTurnCommandsWhenStartingTurningInTSteps[0];
  if (    ivBallInRelaxedTargetAreaInTSteps[ necessaryTurnCmdsWhenStartingTurningNow ]
       || (    necessaryTurnCmdsWhenStartingTurningNow==0 
            && ivBallInRelaxedTargetAreaInTSteps[ 1 ]  )  
     )
  {
    ivAction = BALLZAUBER_TURN;
    ivActionTurnAngle = ivTargetDirection - WSinfo::me->ang;
    TGbzLOGPOL(0,<<"Ballzauber: createAct turn2TargetDir: YEP - BasicCmd "
      <<"(turn inertia "<<ivActionTurnAngle<<") is used. I will need "
      <<necessaryTurnCmdsWhenStartingTurningNow<<" turn commands (myAng="
      <<RAD2DEG(WSinfo::me->ang.get_value_mPI_pPI())<<",tgtAng="
      <<RAD2DEG(ivTargetDirection.get_value_mPI_pPI())<<",|v|="
      <<ivMyVelocityInTSteps[0].norm()<<"). Ball "
      <<"should then be in target area."<<std::flush);
    for (int i=0; i<=necessaryTurnCmdsWhenStartingTurningNow; i++)
      TGbzLOGPOL(0,<<_2D<<C2D(ivBallPositionInTSteps[i].x,
        ivBallPositionInTSteps[i].y,0.085,"5555ff"));
    for (int i=1; i<=ivRequiredTurnCommandsWhenStartingTurningInTSteps[0]; i++)
      ivPlanStringStream << "TURN["<<i<<"] ";
    ivPlanStringStream << "=> "
      <<(ivRequiredTurnCommandsWhenStartingTurningInTSteps[0])<<" STEP(S) ";
    ivNumberOfStepsToTurnToTargetDirection
      = ivRequiredTurnCommandsWhenStartingTurningInTSteps[0];
    return true;
  }
  TGbzLOGPOL(0,<<"Ballzauber: createAct turn2TargetDir: NOPE - I need "
    <<necessaryTurnCmdsWhenStartingTurningNow<<" turn commands, but ball"
    <<" is then not in the target area."<<std::flush);
  TGbzLOGPOL(0,<<_2D<<ivTargetAreaInTSteps[necessaryTurnCmdsWhenStartingTurningNow]);
  TGbzLOGPOL(0,<<_2D<<ivRelaxedTargetAreaInTSteps[necessaryTurnCmdsWhenStartingTurningNow]);
  TGbzLOGPOL(0,<<_2D<<C2D(ivBallPositionInTSteps[necessaryTurnCmdsWhenStartingTurningNow].x,
                          ivBallPositionInTSteps[necessaryTurnCmdsWhenStartingTurningNow].y,
                          0.085,"000000"));
  return false;
}

bool 
Ballzauber::doAHoldTurn()
{
  Vector holdTurnTarget = ivTargetAreaCentralPointInTSteps[1];
  if (       getMaximalVelocityWhenKicking( ivMyPositionInTSteps[0],
                                            ivMyVelocityInTSteps[0],
                                            WSinfo::me->ang,
                                            ivBallPositionInTSteps[0],
                                            ivBallVelocityInTSteps[0],
                                            holdTurnTarget )
          >= holdTurnTarget.distance(ivBallPositionInTSteps[0])
       && isBallPositionSafe(holdTurnTarget) )
  {
    TGbzLOGPOL(0,"Ballzauber: I can do a hold turn now!");
    ivAction = BALLZAUBER_KICK;
    ivActionKickVelocity = holdTurnTarget - ivBallPositionInTSteps[0];
    return true;
  }
  TGbzLOGPOL(0,"Ballzauber: Sorry, but hold turn is impossible now.");
  return false;
}

bool
Ballzauber::get_cmd( Cmd & cmd )
{
  if (WSinfo::ws->time == ivLastTimeCalled)
  {
    if (ivLastResult) this->setCommand( cmd );
    return ivLastResult;
  }
  ivLastTimeCalled = WSinfo::ws->time;
  
  this->initCycle();
  
  if (checkForTermination())
  {
    if ( doAHoldTurn() )
    {
      TGbzLOGPOL(0,"Ballzauber: get_cmd: Acts similar to ^HOLD^TURN now.");
      this->setCommand( cmd );
      TGbzLOGPOL(0,"Ballzauber: get_cmd: SUCCESS");
      ivLastResult = true;
      return true;
    }
    else
    {
      TGbzLOGPOL(0,"Ballzauber: get_cmd: I have finished: Ball has been successfully"
        <<" brought to the target area. Unfortunately, I CANNOT DO A ^HOLD^TURN at this point.");
      return false;
    }
  }  
  
  if ( ivSafetyWithRespectToNearbyOpponents == false )
  {
    TGbzLOGPOL(0,"Ballzauber: get_cmd: FAILURE (there is a dangerous opponent "
      <<"nearby, i must give up)");
    ivLastResult = false;
    return false;
  }
  
  if (    createActionToTurnToTargetDirection() 
       || createActionToKickBallToTargetAreaInAsManyStepsINeedThenForTurning() 
       || createActionToStopBallOrMoveItALittleTowardsTargetArea()
       || createActionToKeepBallSomehowInKickrange()
      )
  {
    this->setCommand( cmd );
    TGbzLOGPOL(0,"Ballzauber: get_cmd: SUCCESS");
    ivLastResult = true;
    return true;
  }
  else
  {
    TGbzLOGPOL(0,"Ballzauber: get_cmd: FAILURE (found no action, cannot keep ball in kickrange)");
    ivLastResult = false;
    return false;
  }
}

bool
Ballzauber::get_cmd( Cmd & cmd, ANGLE targetDir, TargetArea ta)
{
  setTargetDirection( targetDir );
	setTargetArea(ta);
  return get_cmd( cmd );
}

double
Ballzauber::getMaximalVelocityWhenKicking( Vector myPos,
                                           Vector myVel,
                                           ANGLE  myAng,
                                           Vector fromPos,
                                           Vector ballVel,
                                           Vector toPos )
{
  MyState virtualState;
  virtualState.my_pos   = myPos;
  virtualState.my_vel   = myVel;
  virtualState.my_angle = myAng;
  virtualState.ball_pos = fromPos;
  virtualState.ball_vel = ballVel;
  virtualState.op       = NULL;
  ANGLE targetAngle = (toPos-fromPos).ARG();
  return ivpOneStepKickBehavior->get_max_vel_in_dir( virtualState,
                                                     targetAngle );
}

int 
Ballzauber::getNumberOfStepsToTurnToTargetDirection()
{
  if ( WSinfo::ws->time == ivLastTimeCalled )
    return ivNumberOfStepsToTurnToTargetDirection;
  Cmd dummyCmd;
  this->get_cmd( dummyCmd );
  return ivNumberOfStepsToTurnToTargetDirection;
}

int 
Ballzauber::getNumberOfStepsToTurnToTargetDirection( ANGLE dir, TargetArea ta)
{ 
  this->setTargetDirection( dir );
  this->setTargetArea( ta );
  return getNumberOfStepsToTurnToTargetDirection();
}

void
Ballzauber::initCycle()
{
  //init
  ivPlanStringStream.str("");
  ivNumberOfStepsToTurnToTargetDirection = UNABLE_TO_HOLD_BALL;
  //show target dir
  Vector targetDirectionVector( ivTargetDirection );
  targetDirectionVector.normalize(5.0);
  TGbzLOGPOL(0,<<_2D<<L2D(WSinfo::me->pos.x,WSinfo::me->pos.y,
    WSinfo::me->pos.x+targetDirectionVector.x,WSinfo::me->pos.y
    +targetDirectionVector.y,"000099"));
  //calculate my/ball future positions
  ivMyPositionInTSteps[ 0 ] = WSinfo::me->pos;
  ivMyVelocityInTSteps[ 0 ] = WSinfo::me->vel;
  ivBallPositionInTSteps[ 0 ] = WSinfo::ball->pos;
  ivBallVelocityInTSteps[ 0 ] = WSinfo::ball->vel;
  for (int i=1; i<MAX_NUMBER_OF_PLANNING_STEPS; i++)
  {
    ivMyVelocityInTSteps[ i ] 
      = WSinfo::me->decay * ivMyVelocityInTSteps[ i-1 ];
    ivMyPositionInTSteps[ i ] 
      = ivMyPositionInTSteps[ i-1 ] + ivMyVelocityInTSteps[ i-1 ];
    ivBallVelocityInTSteps[ i ] 
      = ServerOptions::ball_decay * ivBallVelocityInTSteps[ i-1 ];
    ivBallPositionInTSteps[ i ]
      = ivBallPositionInTSteps[ i-1 ] + ivBallVelocityInTSteps[ i-1 ];
  }
  for (int i=0; i<4; i++)
    TGbzLOGPOL(0,<<_2D<<C2D(ivBallPositionInTSteps[i].x,
      ivBallPositionInTSteps[i].y,0.085,"ffbbbb"));
  //calculate required turn commands to turn to target direction depending
  //on the number of steps we wait until we start turning
  for (int i=0; i<MAX_NUMBER_OF_PLANNING_STEPS; i++)
  {
    ANGLE remainingTurnAngle = ivTargetDirection - WSinfo::me->ang;
    TGbzLOGPOL(3,"Ballzauber: initCycle: CONSIDER t+"<<i<<" remAng="
      <<remainingTurnAngle.get_value_0_p2PI());
    if (remainingTurnAngle.get_value_0_p2PI() > PI)
      remainingTurnAngle.set_value( 2.0*PI - remainingTurnAngle.get_value_0_p2PI() );
    TGbzLOGPOL(3,"Ballzauber: initCycle: CONSIDER t+"<<i<<" remAng="
      <<remainingTurnAngle.get_value_0_p2PI()<<" IM="<<WSinfo::me->inertia_moment);
    int necessaryTurnCommands = 0;
    while ( fabs(remainingTurnAngle.get_value_0_p2PI()) > MAX_DEVIATION )
    {
      necessaryTurnCommands ++ ;
      if ( i+necessaryTurnCommands >= MAX_NUMBER_OF_PLANNING_STEPS )
      {
        remainingTurnAngle.set_value( 0.0 );
        continue;
      }
      
      double turnInertiaFactor
        = 1.0 + (  WSinfo::me->inertia_moment 
                 * ivMyVelocityInTSteps[i+necessaryTurnCommands-1].norm() );
      ANGLE maximalTurnAngle( PI / turnInertiaFactor );
      if (   maximalTurnAngle.get_value_0_p2PI() 
           > remainingTurnAngle.get_value_0_p2PI() )
        remainingTurnAngle.set_value( 0.0 );
      else
        remainingTurnAngle.set_value( remainingTurnAngle.get_value_0_p2PI()
                                      - maximalTurnAngle.get_value_0_p2PI() );
      TGbzLOGPOL(3,"Ballzauber: initCycle: In t+"<<i+necessaryTurnCommands<<" steps:"
        <<" tgtDir="<<ivTargetDirection.get_value_0_p2PI()
        <<" myAng="<<WSinfo::me->ang
        <<" remAng="<<remainingTurnAngle.get_value_0_p2PI()
        <<" || TIF="<<turnInertiaFactor<<" maxTurn="<<maximalTurnAngle.get_value_0_p2PI());
    }
    ivRequiredTurnCommandsWhenStartingTurningInTSteps[i] = necessaryTurnCommands;
  }
  //calculate target area depending on the time step in future (thus also
  //considering that the player is moving)
  for (int i=0; i<MAX_NUMBER_OF_PLANNING_STEPS; i++)
  {
		switch(ivTargetArea) {
			case TA_IN_FRONT:
			{
				// Target area in front of me
				Vector targetDirectionVector( ivTargetDirection );
				targetDirectionVector.normalize( 0.75*WSinfo::me->kick_radius );
				Vector distantBasePoint = ivMyPositionInTSteps[i] + targetDirectionVector;
				targetDirectionVector.normalize( MIN_DISTANCE_TO_AVOID_COLLSIONS + 0.05 );
				Vector nearBasePoint    = ivMyPositionInTSteps[i] + targetDirectionVector;
				double distantWidth = 2.0 * (0.35*WSinfo::me->kick_radius),
							nearWidth    = 2.0 * (0.65*WSinfo::me->kick_radius);
				ivTargetAreaInTSteps[i] = Quadrangle2d( distantBasePoint,
																								nearBasePoint,
																								distantWidth,
																								nearWidth );
				if (i==0)
					ivSafetyWithRespectToNearbyOpponents
						=    isBallPositionSafe( nearBasePoint ) 
							|| isBallPositionSafe( distantBasePoint );
				//relaxed target areas
				targetDirectionVector.normalize( 0.9*WSinfo::me->kick_radius );                  
				distantBasePoint = ivMyPositionInTSteps[i] + targetDirectionVector;
				targetDirectionVector.normalize( WSinfo::me->radius + ServerOptions::ball_size ); //0.3+0.085=0.385
				nearBasePoint    = ivMyPositionInTSteps[i] + targetDirectionVector;
				distantWidth = 2.0 * (0.25*WSinfo::me->kick_radius);
				nearWidth    = 2.0 * (0.9*WSinfo::me->kick_radius);
				ivRelaxedTargetAreaInTSteps[i] = Quadrangle2d( distantBasePoint,
																											 nearBasePoint,
																											 distantWidth,
																											 nearWidth );
				break;
			}
			case TA_RIGHT_BEHIND:
			case TA_LEFT_BEHIND:
			{
				// Target area behind me, left side
				double distNear = WSinfo::me->radius + ServerOptions::ball_size + .05;
				double distFar  = .8*WSinfo::me->kick_radius;
				double minDistToKickRad = .2*WSinfo::me->kick_radius;

				double addAng = (ivTargetArea == TA_LEFT_BEHIND)?-90:90;
				Vector nearBasePoint;
				nearBasePoint.init_polar(distNear+.05,
						                     ANGLE(DEG2RAD(addAng)) + ivTargetDirection);
				Vector distantBasePoint(nearBasePoint);
				distantBasePoint.normalize(distFar);
				double alpha     = acos(distFar/WSinfo::me->kick_radius);
				double farWidth  = 2.*WSinfo::me->kick_radius * sin(alpha) - 2.0*minDistToKickRad;
				      alpha     = acos(distNear/WSinfo::me->kick_radius);
			  double nearWidth = 2.*WSinfo::me->kick_radius * sin(alpha) - 2.0*minDistToKickRad;
				nearBasePoint    += ivMyPositionInTSteps[i];
				distantBasePoint += ivMyPositionInTSteps[i];
				ivTargetAreaInTSteps[i] = Quadrangle2d( distantBasePoint,
																								nearBasePoint,
																								farWidth,
																								nearWidth );
				if (i==0)
					ivSafetyWithRespectToNearbyOpponents
						=    isBallPositionSafe( nearBasePoint ) 
							|| isBallPositionSafe( distantBasePoint );

				distNear = WSinfo::me->radius+ServerOptions::ball_size;
				distFar  = .85*WSinfo::me->kick_radius;
				nearBasePoint.init_polar(distNear,
						                     ANGLE(DEG2RAD(addAng)) + ivTargetDirection);
				alpha     = acos(distFar/WSinfo::me->kick_radius);
				farWidth  = 2.*WSinfo::me->kick_radius * sin(alpha) - 2.0*minDistToKickRad;
				alpha     = acos(distNear/WSinfo::me->kick_radius);
			  nearWidth = 2.*WSinfo::me->kick_radius * sin(alpha) - 2.0*minDistToKickRad;

				distantBasePoint = nearBasePoint;
				distantBasePoint.normalize(distFar);
				nearBasePoint    += ivMyPositionInTSteps[i];
				distantBasePoint += ivMyPositionInTSteps[i];
				ivRelaxedTargetAreaInTSteps[i] = Quadrangle2d( distantBasePoint,
																											 nearBasePoint,
																											 farWidth,
																											 nearWidth );

				break;
			}
		}
  }    
  //calculate central points of target areas
  for (int i=0; i<MAX_NUMBER_OF_PLANNING_STEPS; i++)
  {
		switch(ivTargetArea){
			case TA_IN_FRONT:
				{
					Vector targetDirectionVector( ivTargetDirection );
					targetDirectionVector.normalize( 0.6*WSinfo::me->kick_radius );
					ivTargetAreaCentralPointInTSteps[i]
						= ivMyPositionInTSteps[i] + targetDirectionVector;
					break;
				}
			case TA_LEFT_BEHIND:
			case TA_RIGHT_BEHIND:
				{
					double addAng = (ivTargetArea == TA_LEFT_BEHIND)?-90:90;
					Vector cp;
					cp.init_polar(.6*WSinfo::me->kick_radius,
							ANGLE(DEG2RAD(addAng)) + ivTargetDirection);
					ivTargetAreaCentralPointInTSteps[i]
						= ivMyPositionInTSteps[i] + cp;
				}
		}
  }    
  //calculate at which points of time the ball will be in the target area
  bool collsionDetected = false;
  for (int i=0; i<MAX_NUMBER_OF_PLANNING_STEPS; i++)
  {
    //target areas
    ivBallInTargetAreaInTSteps[ i ] = false;
    if ( collsionDetected == false )
    {
      if ( ivTargetAreaInTSteps[i].inside( ivBallPositionInTSteps[i] ) )
        ivBallInTargetAreaInTSteps[i] = true;
      if ( i>0 &&  ivBallPositionInTSteps[i].distance( ivMyPositionInTSteps[i] )
                 < MIN_DISTANCE_TO_AVOID_COLLSIONS )
        collsionDetected = true;
    }
    //relaxed target areas
    ivBallInRelaxedTargetAreaInTSteps[ i ] = false;
    if ( collsionDetected == false )
    {
      if ( ivRelaxedTargetAreaInTSteps[i].inside( ivBallPositionInTSteps[i] ) )
        ivBallInRelaxedTargetAreaInTSteps[i] = true;
      if ( i>0 &&  ivBallPositionInTSteps[i].distance( ivMyPositionInTSteps[i] )
                 < MIN_DISTANCE_TO_AVOID_COLLSIONS )
        collsionDetected = true;
    }
  }
}

bool
Ballzauber::isBallPositionSafe( Vector pos )
{
  PlayerSet relevantOpponents = WSinfo::valid_opponents;
  relevantOpponents.keep_players_in_circle( ivMyPositionInTSteps[0], 4.0 );
	if(relevantOpponents.num==1)
	{
	    int minInact, maxInact;
        WSinfo::get_player_inactivity_interval( relevantOpponents[0], minInact, maxInact );
        if (minInact > 0) return true;
        if (   WSinfo::me->pos.getX() - WSinfo::my_team_pos_of_offside_line() > 20.0
            && (minInact+maxInact)/2 > 0 )
          return true;

		DribbleAround06::PlayerState p;

		p.setAssumeNoAction(relevantOpponents[0],pos);
		if(p.reachesPos) return false;
		if(0.8<Tools::get_tackle_success_probability(p.pos,pos,p.ang.get_value()))
			return false;

		p.setAssumeToPos(relevantOpponents[0],pos);
		if(p.reachesPos) return false;
		if(0.8<Tools::get_tackle_success_probability(p.pos,pos,p.ang.get_value()))
			return false;

		p.setAssumeToPos(relevantOpponents[0],WSinfo::me->pos);
		if(p.pos.distance(pos)<p.kick_radius+.1) return false;
		if(0.8<Tools::get_tackle_success_probability(p.pos,pos,p.ang.get_value()))
			return false;
		return true;
	}
  return 
    Tools::is_ballpos_safe( relevantOpponents, pos, true ); //consider_tackles
}

bool
Ballzauber::isBallzauberToTargetDirectionPossible()
{
  if (WSinfo::ws->time == ivLastTimeCalled)
  {
    return ivLastResult;
  }
  Cmd dummyCmd;
  return get_cmd( dummyCmd );
}

bool
Ballzauber::isBallzauberToTargetDirectionPossible( ANGLE targetDir, TargetArea ta)
{
  this->setTargetDirection( targetDir );
	this->setTargetArea(ta);
  return isBallzauberToTargetDirectionPossible();
}

bool
Ballzauber::setCommand( Cmd & cmd )
{
  bool returnValue = false;
  switch (ivAction)
  {
    case BALLZAUBER_KICK:
    {
      ivpOneStepKickBehavior->kick_in_dir_with_initial_vel
                              (
                                ivActionKickVelocity.norm(),
                                ivActionKickVelocity.ARG()
                              );
      returnValue = ivpOneStepKickBehavior->get_cmd(cmd);
      double pow=0.0; Angle ang=0.0; cmd.cmd_body.get_kick(pow,ang);
      TGbzLOGPOL(0,"Ballzauber: setCommand: COMMAND: KICK ("<<
        cmd.cmd_body.get_type()<<"): "<<pow<<"/"<<RAD2DEG(ang)
        <<" | PLAN: "<<ivPlanStringStream.str()
        <<" "<<ivNumberOfStepsToTurnToTargetDirection);
      break;
    }
    case BALLZAUBER_TURN:
    {
      ivpBasicCmdBehavior->set_turn_inertia
                           ( ivActionTurnAngle.get_value_mPI_pPI() );
      returnValue = ivpBasicCmdBehavior->get_cmd(cmd);
      Angle ang=0.0; cmd.cmd_body.get_turn(ang);
      TGbzLOGPOL(0,"Ballzauber: setCommand: COMMAND: TURN ("<<
        cmd.cmd_body.get_type()<<"): "<<RAD2DEG(ang)
        <<" | PLAN: "<<ivPlanStringStream.str()
        <<" "<<ivNumberOfStepsToTurnToTargetDirection);
      break;
    }
    default:
    {
      break;
    }
  }
  return returnValue;
}

void 
Ballzauber::setTargetDirection( ANGLE ang )
{
  ivTargetDirection = ang;
  ivLastTimeCalled = -1;
}
void 
Ballzauber::setTargetArea( TargetArea ta )
{
  ivTargetArea = ta;
  ivLastTimeCalled = -1;
}
