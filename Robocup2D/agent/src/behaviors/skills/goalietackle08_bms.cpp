#include "goalietackle08_bms.h"

#include "../goalie_bs03_bmc.h"
#include "../../basics/tools.h"
#define BASELEVEL 0

#if 0
#define TGgtLOGPOL(YYY,XXX) LOG_POL(YYY,XXX)
#else
#define TGgtLOGPOL(YYY,XXX)
#endif

//#define LOG_DAN(YYY,XXX)

bool GoalieTackle08::initialized = false;
const double GoalieTackle08::cvcExpectedPerceptionNoise = 0.06;
double GoalieTackle08::cvcRealCatchRadius
  = sqrt(  ServerOptions::catchable_area_l * ServerOptions::catchable_area_l
         + 0.25 * ServerOptions::catchable_area_w * ServerOptions::catchable_area_w );


GoalieTackle08::GoalieTackle08() 
{
  ivBallCrossesMyGoalLineInNSteps      = cvcFutureBallSteps;
  ivAggressiveCatchIsAdvisable         = false;
  ivAggressiveCatchIsAdvisableNextStep = false;
  ivIsGoalieTackleSituation            = false;
  
  ivpInterceptBallBehavior = new InterceptBall();
  ivpBasicCmdBehavior      = new BasicCmd();
}


GoalieTackle08::~GoalieTackle08() 
{
  if (ivpInterceptBallBehavior) delete ivpInterceptBallBehavior;
  if (ivpBasicCmdBehavior)      delete ivpBasicCmdBehavior;
}


ANGLE
GoalieTackle08::calculateAggressiveCatchAngle()
{
  ANGLE standardCatchANGLE = Tools::my_angle_to(WSinfo::ball->pos);
  ANGLE bestCatchANGLE = standardCatchANGLE;
  Angle maxAbsDeviationFromStandardANGLE 
    = (PI*0.5) - atan(   ServerOptions::catchable_area_l 
                       / (0.5*ServerOptions::catchable_area_w) );
  const int   numberOfCatchAnglesToBeTested = 41;
  const double testAngleIncrement =   2.0*maxAbsDeviationFromStandardANGLE
                                   / (double)(numberOfCatchAnglesToBeTested-1);
  int bestCatchANGLEEvaluation = -1, bestCatchANGLEIndex = -1;
  for ( int c = 0; c < numberOfCatchAnglesToBeTested; c++ )
  {
    Angle currentCatchAngle 
      =   standardCatchANGLE.get_value_0_p2PI()
        + (double)(  testAngleIncrement
                  * (c - (numberOfCatchAnglesToBeTested/2)) );
    ANGLE currentCatchANGLE( currentCatchAngle );
    int currentCatchANGLEEvaluation 
      = evaluateCatchAngle( currentCatchANGLE, ((c%4)==0) );
    if (   currentCatchANGLEEvaluation > bestCatchANGLEEvaluation
        || (   currentCatchANGLEEvaluation == bestCatchANGLEEvaluation
            &&   abs( c - (numberOfCatchAnglesToBeTested/2) )
               < abs( bestCatchANGLEIndex - (numberOfCatchAnglesToBeTested/2) )  
           )
       )
    {
      bestCatchANGLEIndex      = c;
      bestCatchANGLE           = currentCatchANGLE;
      bestCatchANGLEEvaluation = currentCatchANGLEEvaluation;
    }
  }
  TGgtLOGPOL(1,<<"GoalieTackle08: Result of calculateAggressiveCatchAngle"
    <<" is: bestCatchANGLEIndex="<<bestCatchANGLEIndex
    <<" bestCatchANGLE="<<RAD2DEG(bestCatchANGLE.get_value_mPI_pPI())
    <<" bestCatchANGLEEvaluation="<<bestCatchANGLEEvaluation);
  Vector endPoint( bestCatchANGLE + WSinfo::me->ang );
  endPoint.normalize( ServerOptions::catchable_area_l );
  endPoint += WSinfo::me->pos;
  Quadrangle2d checkArea( WSinfo::me->pos, 
                          endPoint,
                          ServerOptions::catchable_area_w,
                          ServerOptions::catchable_area_w );
  checkArea.setColor("55ff55");
  TGgtLOGPOL(0,<<_2D<<checkArea);
  return bestCatchANGLE;  
}

bool 
GoalieTackle08::calculatePreferredTackleANGLE( ANGLE & ang )
{
  double bestEvaluation   = -INT_MAX,
        currentEvaluation;
  int   bestCheckedAngle = 0;
  for ( int checkAngle=0; checkAngle<360; checkAngle += 10 )
  {
    currentEvaluation = evaluateAngularTackleDirection( (double)checkAngle );
    if (currentEvaluation > bestEvaluation)
    {
      bestEvaluation = currentEvaluation;
      bestCheckedAngle = checkAngle;
    }                         
    TGgtLOGPOL(2,<<"GoalieTackle08: calculatePreferredTackleANGLE: "
      <<"Evaluation of "<<checkAngle<<" is "<<currentEvaluation);
  }
  if ( bestEvaluation >= -50 )
  {
    ang.set_value( DEG2RAD((double)bestCheckedAngle) );
    TGgtLOGPOL(1,"GoalieTackle08: calculatePreferredTackleANGLE: Best TACKLE"
      <<" is "<<bestCheckedAngle<<" with "<<bestEvaluation<<" points.");
    return true;
  }
  else
  {
    TGgtLOGPOL(1,<<"GoalieTackle08: calculatePreferredTackleANGLE: No suitable "
      <<"tackle angle found. GIVING UP!");
    return false;
  }
}

double
GoalieTackle08::evaluateAngularTackleDirection( double checkAngle )
{
  ANGLE myAngle = WSinfo::me->ang;
  Vector myPos   = WSinfo::me->pos,
         ballPos = WSinfo::ball->pos,
         ballVel = WSinfo::ball->vel;
  //PRECALCULATIONS  
    /*
    //tackle power depends on the tackle angle
    double effectiveTacklePower
      =   ServerOptions::max_back_tackle_power
        +   (ServerOptions::max_tackle_power - ServerOptions::max_back_tackle_power)
          * (fabs(180.0 - checkAngle)/180.0);
    Vector player_2_ball = ballPos - myPos;
    player_2_ball.rotate( - myAngle.get_value_mPI_pPI() );
    ANGLE player_2_ball_ANGLE = player_2_ball.ARG();
    effectiveTacklePower
      *= 1.0 - 0.5 * ( fabs( player_2_ball_ANGLE.get_value_mPI_pPI() ) / PI );
    double relativeTacklePower = effectiveTacklePower / 100.0;
    
    TGgtLOGPOL(4,<<"TACKLE-INFO: checkAngle="<<checkAngle
      <<" player2ball="<<player_2_ball.arg()
      <<" effPowDecay="
      <<(1.0 - 0.5 * ( fabs( player_2_ball_ANGLE.get_value_mPI_pPI() ) / PI ))
      <<" effPower="<<effectiveTacklePower);

    //directed tackle: resulting ball velocity
    Vector tackle_dir = Vector( myAngle + ANGLE(DEG2RAD(checkAngle)) );
    tackle_dir.normalize(   ServerOptions::tackle_power_rate 
                          * effectiveTacklePower);
    tackle_dir += ballVel;
    if ( tackle_dir.norm() > ServerOptions::ball_speed_max )
      tackle_dir.normalize( ServerOptions::ball_speed_max );
    */
    //model the tackle using tools    
    Vector ballNewPos, dummyBallVelAfterTackling;
    Tools::model_tackle_V12( myPos,
                             myAngle,
                             ballPos,
                             ballVel,
                             checkAngle,
                             ballNewPos,
                             dummyBallVelAfterTackling);
    Vector tackle_dir = ballNewPos - ballPos;
    //calculate ball pos in 10 steps
    Vector ballPosWhenIAwake = ballNewPos,
           continuedBallVel  = dummyBallVelAfterTackling;
    for (int i=0; i<3/*ServerOptions::tackle_cycles*/; i++)
    {
      ballPosWhenIAwake += continuedBallVel;
      continuedBallVel *= ServerOptions::ball_decay;
    } 
    //a 20m long vector into the tackling direction 
    Vector         look_dir       =  tackle_dir; 
    look_dir.normalize(  20.0 
                       * (tackle_dir.norm()/ServerOptions::ball_speed_max) );

    //a 1.5m long vector into the tackling direction 
    Vector         dir            =  tackle_dir;
    dir.normalize(1.5);
    
    //we define a check area in front of me, it starts 1.5 from the
    //ball and ends 20m from the ball
    Vector endPoint               = ballPos + look_dir;
    Vector midPoint               = endPoint + ballPos;
    midPoint *= 0.5; 
    Quadrangle2d check_area
      =  Quadrangle2d( ballPos + dir,  endPoint,  2.0,  12.0);

    //opponents in the check area 
    PlayerSet opps_in_check_area      =  WSinfo::valid_opponents;
    opps_in_check_area.keep_players_in( check_area );
    //teammates in the check area
    PlayerSet team_in_check_area      =  WSinfo::valid_teammates_without_me;
    team_in_check_area.keep_players_in( check_area );
    
  //CHECK FOR COMPLETELY EXCLUDING A DIRECTION
    Vector ballCrossesGoalLine
        = Tools::point_on_line( ballVel,
                                ballPos,
                                -FIELD_BORDER_X );
    Vector 
      tackleBallMovementYPointOnGoalLineStandard
        = Tools::point_on_line( tackle_dir,
                                ballPos,
                                -FIELD_BORDER_X );
    TGgtLOGPOL(4,<<"evaluateAngularTackleDirection: checkAngle="
      <<checkAngle<<" Ystd="<<tackleBallMovementYPointOnGoalLineStandard
      <<" (tackle_dir="<<tackle_dir<<", DEG2RAD(checkAngle)="<<DEG2RAD(checkAngle)
      <<", myAngle="<<myAngle<<")");
    
    //exclude tacklings that directly yield a self goal
    double stepsToGoalLine
      =   tackleBallMovementYPointOnGoalLineStandard.distance( ballPos )
        / tackle_dir.norm(); 
    double spitzwinkligkeit
      //= 0.5*PI - fabs(fabs(tackle_dir.ARG().get_value_mPI_pPI()) - 0.5*PI); //zwischen 0 und 3.14
      = fabs(sin( tackle_dir.ARG().get_value_mPI_pPI() ));
    double selfGoadAvoidThreshold =   0.1 //TG09: reduce from 1.5 to 0.1
                                   + 0.1 * stepsToGoalLine
                                   + 0.5 * spitzwinkligkeit;
    if (   fabs(ballCrossesGoalLine.getY()) < 0.5*ServerOptions::goal_width + 0.1
        && (ballPos+ballVel).getY() < -FIELD_BORDER_X )
      selfGoadAvoidThreshold = 0.0;

    if (  (      fabs(tackleBallMovementYPointOnGoalLineStandard.getY())
               < 0.5 * ServerOptions::goal_width + selfGoadAvoidThreshold 
            && tackle_dir.getX() < 0.0 )
        && ballPosWhenIAwake.getX() < -FIELD_BORDER_X
       )
    {
      TGgtLOGPOL(4,<<_2D<<VL2D(ballPos,ballPosWhenIAwake,"ffaaaa")<<std::flush);
      TGgtLOGPOL(4,<<_2D<<VSTRING2D(endPoint,(int)checkAngle,"ffaaaa")<<std::flush);
      TGgtLOGPOL(4,<<"evaluateAngularTackleDirection: Must exclude "<<checkAngle
        <<", because it would yield a direct self goal (self goal), thresh="<<selfGoadAvoidThreshold
        <<", stepsToGoalLine="<<stepsToGoalLine<<", spitzwinkligkeit="<<spitzwinkligkeit<<".");
      return -100.0;
    }
    else
    {
      TGgtLOGPOL(4,<<_2D<<VL2D(ballPos,ballPosWhenIAwake,"aaaaff")<<std::flush);
      TGgtLOGPOL(4,<<_2D<<VSTRING2D(endPoint,(int)checkAngle,"aaaaff")<<std::flush);
    }
    
  //STANDARD EVALUATION OF TACKLINGS
    //evaluation values that finally decide for a tackling direction
    double  returnValue      =  0.0;
    //it is good, if the endpoint is away from my goal
    double dangerValueEndPoint
            =   (endPoint - MY_GOAL_LEFT_CORNER).norm()
              + (endPoint - MY_GOAL_CENTER).norm()
              + (endPoint - MY_GOAL_RIGHT_CORNER).norm(),
          dangerValueMidPoint 
            =   (midPoint - MY_GOAL_LEFT_CORNER).norm()
              + (midPoint - MY_GOAL_CENTER).norm()
              + (midPoint - MY_GOAL_RIGHT_CORNER).norm(),
          currentDanger
            =   (WSinfo::ball->pos - MY_GOAL_LEFT_CORNER).norm()
              + (WSinfo::ball->pos - MY_GOAL_CENTER).norm()
              + (WSinfo::ball->pos - MY_GOAL_RIGHT_CORNER).norm();
    returnValue += dangerValueEndPoint - currentDanger; 
    returnValue += dangerValueMidPoint - currentDanger;
    //it is good, if the ball is moving into positive x direction
/*    if ( ballPos.x - (-FIELD_BORDER_X) < 3.0 )
      returnValue +=   (4.0 - (ballPos.x - (-FIELD_BORDER_X))) 
                     * (endPoint - ballPos).x;
    else
      returnValue += (endPoint - ballPos).x;*/
    //TG09: it is very good, if the ball goes out of the field quickly
    if (   fabs(tackleBallMovementYPointOnGoalLineStandard.getY())
         > 0.5*ServerOptions::goal_width )
      returnValue -= 0.3 * stepsToGoalLine;
    //TG09: it is very good, if a ball that comes near to the posts is not as spitzwinklig
    if (  fabs(tackleBallMovementYPointOnGoalLineStandard.getY())
        < 0.5 * ServerOptions::goal_width + 3.0 )
      returnValue -= spitzwinkligkeit;

    //reward tacklings that go not directly to the centre of our goal //TG09
    if (   fabs(tackleBallMovementYPointOnGoalLineStandard.getY())
         < 0.5*ServerOptions::goal_width + 1.0 )
      returnValue += 0.3*fabs(tackleBallMovementYPointOnGoalLineStandard.getY());
    if (   fabs(tackleBallMovementYPointOnGoalLineStandard.getY())
         < 0.5*ServerOptions::goal_width )
      returnValue -= 5.0;

    //impair tacklings that come too near to the post
    if ( fabs(tackle_dir.getX()) > 0.0 )
    {
      const double minimalPostDistance = 0.3;
      if ( tackle_dir.getY() > 0.0 ) //ball goes up
      {
        if ( ballPos.getY() <= MY_GOAL_LEFT_CORNER.getY() )
        {
          Vector intersectionWithYLineLeftPost
            = Tools::intersection_point( ballPos, 
                                         tackle_dir, 
                                         MY_GOAL_LEFT_CORNER, 
                                         Vector(1.0,0.0) );
          double distToPost
            = intersectionWithYLineLeftPost.getX() - ( -FIELD_BORDER_X );
          if ( distToPost <   minimalPostDistance
                            + 0.1 * ballPos.distance(intersectionWithYLineLeftPost) )
          {
            TGgtLOGPOL(4,<<"evaluateAngularTackleDirection: Impair "
              <<checkAngle
              <<", because it comes too near to my left post ("
              <<distToPost<<").");
            returnValue -= 10.0;
          }
        }
        if ( ballPos.getY() <= MY_GOAL_RIGHT_CORNER.getY() )
        {
          Vector intersectionWithYLineRightPost
            = Tools::intersection_point( ballPos, 
                                         tackle_dir, 
                                         MY_GOAL_RIGHT_CORNER, 
                                         Vector(1.0,0.0) );
          double distToPost
            = intersectionWithYLineRightPost.getX() - ( -FIELD_BORDER_X );
          if ( distToPost <   minimalPostDistance
                            + 0.1 * ballPos.distance(intersectionWithYLineRightPost) )
          {
            TGgtLOGPOL(4,<<"evaluateAngularTackleDirection: Impair "
              <<checkAngle
              <<", because it comes too near to my right post ("
              <<distToPost<<").");
            returnValue -= 10.0;
          }
        }
      }
      else if ( tackle_dir.getY() < 0.0 ) //ball goes down
      {
        if ( ballPos.getY() >= MY_GOAL_RIGHT_CORNER.getY() )
        {
          Vector intersectionWithYLineRightPost
            = Tools::intersection_point( ballPos, 
                                         tackle_dir, 
                                         MY_GOAL_RIGHT_CORNER, 
                                         Vector(1.0, 0.0) );
          double distToPost
            = intersectionWithYLineRightPost.getX() - ( -FIELD_BORDER_X );
          if ( distToPost <   minimalPostDistance
                            + 0.1 * ballPos.distance(intersectionWithYLineRightPost) )
          {
            TGgtLOGPOL(4,<<"evaluateAngularTackleDirection: Impair "
              <<checkAngle
              <<", because it comes too near to my right post ("
              <<distToPost<<").");
            returnValue -= 10.0;
          }
        }
        if ( ballPos.getY() >= MY_GOAL_LEFT_CORNER.getY() )
        {
          Vector intersectionWithYLineLeftPost
            = Tools::intersection_point( ballPos, 
                                         tackle_dir, 
                                         MY_GOAL_RIGHT_CORNER, 
                                         Vector( 1.0, 0.0) );
          double distToPost
            = intersectionWithYLineLeftPost.getX() - ( -FIELD_BORDER_X );
          if ( distToPost <   minimalPostDistance
                            + 0.1 * ballPos.distance(intersectionWithYLineLeftPost) )
          {
            TGgtLOGPOL(4,<<"evaluateAngularTackleDirection: Impair "
              <<checkAngle
              <<", because it comes too near to my left post ("
              <<distToPost<<").");
            returnValue -= 10.0;
          }
        }
      }
      else {}//div by zero avoided
    }
        
    //take opponents/teammates in check area into consideration
    if (opps_in_check_area.num==0)
    {
      if (team_in_check_area.num==0) 
      {
        //nobody is in the check area
        //will the tackling probably result in shooting 
        //the ball out of the field?
        if (    (   fabs(endPoint.getX()) >= FIELD_BORDER_X - 2.0
                 || fabs(endPoint.getY()) >= FIELD_BORDER_Y - 2.0)
             && opps_in_check_area.num==0) //the latter is always true
          returnValue += 1.0 + fabs(endPoint.getY())*0.05;
      }
      else 
        //the more teammates in front of me, the better
        returnValue += team_in_check_area.num;
    } 
    else //else means: i have opponents in front of me
      if ( team_in_check_area.num == 0 ) 
        returnValue += -opps_in_check_area.num;
      else //else means: i have teammates (and opponents) in the check area 
      {
        //distance of the closest teammate to the ball
        double dist_team
          = ( team_in_check_area.closest_player_to_point(ballPos) )
            ->pos.distance(ballPos);
        //distance of the closest opponent to the ball
        double dist_opps
          = (opps_in_check_area.closest_player_to_point(ballPos) )
            ->pos.distance(ballPos);
        returnValue += (dist_opps-dist_team) * 0.1;
      }

  return returnValue;
}

int 
GoalieTackle08::evaluateCatchAngle( ANGLE & ang, bool debug )
{
  int returnValue = 0;
  //calculate catch rectangle
  Vector endPoint( ang + WSinfo::me->ang );
  endPoint.normalize( ServerOptions::catchable_area_l );
  endPoint += WSinfo::me->pos;
  Quadrangle2d checkArea( WSinfo::me->pos, 
                          endPoint,
                          ServerOptions::catchable_area_w,
                          ServerOptions::catchable_area_w );
  if (debug) {TGgtLOGPOL(0,<<_2D<<checkArea);}
  const int numberOfCirclesAroundBall =  4;
  const int numberOfParticlesPerCircle= 20;
  Vector currentParticle;
  double currentRadius;
  for ( int c=0; c < numberOfCirclesAroundBall; c++ )
  {
    if (c==0)
      currentRadius = 0.01;
    else
      currentRadius = (((double)(c)) / (double)numberOfCirclesAroundBall)
                      * cvcExpectedPerceptionNoise*1.5;
    for ( int p=0; p < numberOfParticlesPerCircle; p++ )
    {
      currentParticle.setXY( 1.0, 0.0 );
      currentParticle.normalize( currentRadius );
      currentParticle.rotate(  ((double)p / (double)numberOfParticlesPerCircle)
                             * 2.0*PI );
      currentParticle += WSinfo::ball->pos;
      if ( checkArea.inside( currentParticle ) )
        returnValue ++ ;
    }
  }
  TGgtLOGPOL(3,<<"GoalieTackle08: evaluateCatchAngle "
    <<RAD2DEG(ang.get_value_mPI_pPI())<<" gets "<<returnValue<<" points.");
  return returnValue;
}

bool 
GoalieTackle08::get_cmd( Cmd & cmd ) 
{
  //handle special cases
  if ( ivNumberOfStepsToCatchTheBall == 1 )
  {
    TGgtLOGPOL(0,<<"GoalieTackle08: I call the intercept behavior "
      <<"to provide a command.");
    this->updateCatchingInterceptInformation( & cmd );
    return true;
  }
  //standard GoalieTackle08 behavior
  if ( ivAggressiveCatchIsAdvisable == true )
  {
    Goalie_Bs03::catch_ban = ServerOptions::catch_ban_cycle;
    ANGLE catchANGLE = calculateAggressiveCatchAngle();
    TGgtLOGPOL(0,<<"GoalieTackle08: Try to make an aggressive catch into "
      <<"direction "<<RAD2DEG(catchANGLE.get_value_0_p2PI()));
    if (ivBestCatchIndex != TackleAndCatchProbabilityInformationHolder
                            ::cvcCATCHORTACKLENOWINDEX)
    {
      TGgtLOGPOL(0,<<"GoalieTackle08: SEVERE ERROR. Index mismatch ("
        "ivBestCatchIndex="<<ivBestCatchIndex<<")");
    }
    ivpBasicCmdBehavior->set_catch( catchANGLE );
    ivpBasicCmdBehavior->get_cmd(cmd);
    return true;
  }
  else if ( ivAggressiveCatchIsAdvisableNextStep == true )
  {
    TGgtLOGPOL(0,<<"GoalieTackle08: Try to make move such that an aggressive"
      <<" catch is promising in the next step, ivBestCatchIndex="
      <<ivBestCatchIndex<<".");
    if (    ivBestCatchIndex == -1
         || ivBestCatchIndex >= TackleAndCatchProbabilityInformationHolder
                                ::cvcTURNtoBALLINDEX )
    {
      TGgtLOGPOL(0,<<"GoalieTackle08: SEVERE ERROR. Index mismatch ("
        "ivBestCatchIndex="<<ivBestCatchIndex<<")");
    }
    int dashPower, dummyAction;
    ivTackleAndCatchProbabilityInformation.indexToAction( ivBestCatchIndex,
                                                          dummyAction,
                                                          dashPower );
    ivpBasicCmdBehavior->set_dash( dashPower );
    ivpBasicCmdBehavior->get_cmd(cmd);
    return true;
  }
  else if ( ivImmediateTacklingIsAdvisable == true ) 
  {
    ANGLE tackleANGLE;
    TGgtLOGPOL(0,<<"GoalieTackle08: Try to make a tackling into "
      <<"direction "<<RAD2DEG(tackleANGLE.get_value_0_p2PI()));
    if (ivBestTackleIndex != TackleAndCatchProbabilityInformationHolder
                             ::cvcCATCHORTACKLENOWINDEX)
    {
      TGgtLOGPOL(0,<<"GoalieTackle08: SEVERE ERROR. Index mismatch ("
        "ivBestTackleIndex="<<ivBestTackleIndex<<")");
    }
    bool success = calculatePreferredTackleANGLE( tackleANGLE);
    if (success)
    {
      //TG2010
      if (WSinfo::ws->play_mode == PM_his_PenaltyKick)
        ivpBasicCmdBehavior
          ->set_tackle( - RAD2DEG( tackleANGLE.get_value_mPI_pPI() ), true );
      else
        ivpBasicCmdBehavior //IMPORTANT: Server uses left hand 
                            //coordinate system! ==> NEGATE!
          ->set_tackle( - RAD2DEG( tackleANGLE.get_value_mPI_pPI() ) );
      ivpBasicCmdBehavior->get_cmd(cmd);
      return true;
    }
    else
    {
      TGgtLOGPOL(0,<<"GoalieTackle08: I DID NOT FIND ANY VALUABLE TACKLE "
        <<"DIRECTION! GIVING UP!!!");
    }
  }
  else if ( ivTacklePreparationIsAdvisable == true )
  {
    if (   ivBestTackleIndex > TackleAndCatchProbabilityInformationHolder
                               ::cvcTURNandDASHP100andDASHP100INDEX
        || ivBestTackleIndex < 0  
       )
    {
      TGgtLOGPOL(0,<<"GoalieTackle08: SEVERE ERROR. Index mismatch ("
        "ivBestTackleIndex="<<ivBestTackleIndex<<")");
    }
    if ( ivBestTackleIndex == TackleAndCatchProbabilityInformationHolder
                              ::cvcTURNtoBALLINDEX )
    {
      TGgtLOGPOL(0,<<"GoalieTackle08: Try to TURN to the ball such that "
        <<"a tackling is promising in the next step, ivBestTackleIndex="
        <<ivBestTackleIndex);
      Vector myOffset = WSinfo::me->vel;
      Vector myPosNextCycleWhenNotDashing = WSinfo::me->pos + myOffset;
      ANGLE targetANGLE 
        = (ivpFutureBallPositions[1] - myPosNextCycleWhenNotDashing).ARG();
      Angle turnMoment = (targetANGLE - WSinfo::me->ang).get_value();
      ivpBasicCmdBehavior->set_turn_inertia( turnMoment );
      ivpBasicCmdBehavior->get_cmd( cmd );
      return true;
    }
    else 
    if ( ivBestTackleIndex == TackleAndCatchProbabilityInformationHolder
                              ::cvcTURNandDASHP100INDEX )
    {
      TGgtLOGPOL(0,<<"GoalieTackle08: Try to TURNandDASH to the ball such that "
        <<"a tackling is promising in the OVER-next step, ivBestTackleIndex="
        <<ivBestTackleIndex);
      Vector myOffset = WSinfo::me->vel;
      Vector myPosNextCycleWhenNotDashing = WSinfo::me->pos + myOffset;
      ANGLE targetANGLE 
        = (ivpFutureBallPositions[2] - myPosNextCycleWhenNotDashing).ARG();
      Angle turnMoment = (targetANGLE - WSinfo::me->ang).get_value();
      ivpBasicCmdBehavior->set_turn_inertia( turnMoment );
      ivpBasicCmdBehavior->get_cmd( cmd );
      return true;
    }
    else 
    if ( ivBestTackleIndex == TackleAndCatchProbabilityInformationHolder
                              ::cvcTURNandDASHP100andDASHP100INDEX )
    {
      TGgtLOGPOL(0,<<"GoalieTackle08: Try to TURNandDASHandDASH to the ball such that "
        <<"a tackling is promising in the OVER-OVER-next step, ivBestTackleIndex="
        <<ivBestTackleIndex);
      Vector myOffset = WSinfo::me->vel;
      Vector myPosNextCycleWhenNotDashing = WSinfo::me->pos + myOffset;
      ANGLE targetANGLE 
        = (ivpFutureBallPositions[3] - myPosNextCycleWhenNotDashing).ARG();
      Angle turnMoment = (targetANGLE - WSinfo::me->ang).get_value();
      ivpBasicCmdBehavior->set_turn_inertia( turnMoment );
      ivpBasicCmdBehavior->get_cmd( cmd );
      return true;
    }
    else //dash to prepare the tackling
    {
      TGgtLOGPOL(0,<<"GoalieTackle08: Try to DASH towards the ball such that "
        <<"a tackling is promising in the next step, ivBestTackleIndex="
        <<ivBestTackleIndex);
      int dashPower, dummyAction;
      ivTackleAndCatchProbabilityInformation.indexToAction( ivBestTackleIndex,
                                                            dummyAction,
                                                            dashPower );
      ivpBasicCmdBehavior->set_dash( dashPower );
      ivpBasicCmdBehavior->get_cmd(cmd);
      return true;
    }
  }
  //default
  TGgtLOGPOL(0,<<"GoalieTackle08: get_cmd() has been called, but does not "
    <<"know what to do. GoalieTackle08 FAILS.");
  return false;
}

bool
GoalieTackle08::init( char const * conf_file, int argc, 
                      char const* const* argv ) 
{
  if ( initialized )
    return true;
  initialized= true;
  return 
    (
         BasicCmd     ::init(conf_file, argc, argv) 
      && InterceptBall::init(conf_file, argc, argv) 
    );
}

bool 
GoalieTackle08::isBallHeadingForGoal() 
{
  double steigung;
  if (WSinfo::ball->vel.getX() >= 0.0)
    return false;
  steigung = -(WSinfo::ball->vel.getY() / WSinfo::ball->vel.getX());
  Vector id_Punkt_auf_Tor 
    =   ServerOptions::own_goal_pos 
      + Vector( 0, 
                  WSinfo::ball->pos.getY()
                + steigung * (  WSinfo::ball->pos.getX()
                              - ServerOptions::own_goal_pos.getX()) );

  if (   ( skalarprodukt(WSinfo::ball->vel, Vector(-1,0)) > 0) 
      && (fabs(id_Punkt_auf_Tor.getY()) <= ServerOptions::goal_width/2.0 + 2.0))
    return true;
  //default
  return false;
} 

bool 
GoalieTackle08::isGoalieTackleSituation()
{
  ivIsGoalieTackleSituation = false;
  this->updateForCurrentCycle();
  //check if the ball is too outdated
  if ( WSinfo::ball->age > 1 )
  {
    TGgtLOGPOL(0,<<"GoalieTackle08: No GT situation: Have not seen"
      <<" the ball recently.");
    ivIsGoalieTackleSituation = false;
    return false;
  }
  //check if ball is kickable for an opponent
  PlayerSet opps = WSinfo::valid_opponents;
  PPlayer closestOpponent
    = opps.closest_player_to_point( WSinfo::ball->pos );
  bool ballIsKickableForAnOpponent
    =     closestOpponent != NULL 
       &&   closestOpponent->pos.distance(WSinfo::ball->pos)
          < closestOpponent->kick_radius;
  if (    ballIsKickableForAnOpponent 
     )
  {
    TGgtLOGPOL(0,<<"GoalieTackle08: No GT situation: Ball is within the"
      <<" kick range of an opponent player.");
    ivIsGoalieTackleSituation = false;
    this->updateDirectActionInformation(); //update those infos anyway!
    return false;
  }
  //check if ball is not heading for goal
  if (    isBallHeadingForGoal() == false 
     )
  {
    TGgtLOGPOL(0,<<"GoalieTackle08: No GT situation: Ball is not heading"
      <<" toward our goal.");
    ivIsGoalieTackleSituation = false;
    return false;
  } 
  //check if no goal is about to being scored
  if (    ivBallCrossesMyGoalLineInNSteps >= cvcFutureBallSteps 
     )
  {
    TGgtLOGPOL(0,<<"GoalieTackle08: No GT situation: Ball will not go"
      <<" into my goal within the next "<<cvcFutureBallSteps<<" cycles.");
    ivIsGoalieTackleSituation = false;
    return false;
  }
  //check if i may reach the ball before it crosses the goal line
  if (    ivNumberOfStepsToCatchTheBall < ivBallCrossesMyGoalLineInNSteps
       && ivNumberOfStepsToCatchTheBall > 0 
     )
  {
    TGgtLOGPOL(0,<<"GoalieTackle08: No GT situation: I will be able to"
      <<" catch the ball before it crosses the goal line. I need "
      <<ivNumberOfStepsToCatchTheBall<<" steps, whereas ball goes "
      <<"into the goal in "<<ivBallCrossesMyGoalLineInNSteps<<" steps.");
    if ( ivNumberOfStepsToCatchTheBall == 1 )
    {
      TGgtLOGPOL(0,<<"GoalieTackle08: No GT situation: I will be able to"
        <<" catch after ONE STEP!");
      ivIsGoalieTackleSituation = true;
      this->updateDirectActionInformation();
      return true;
    }
    ivIsGoalieTackleSituation = false;
    return false;
  }
  //check if the standard goalie behavior might catch the ball now
  bool standardGoalieBehaviorPerformsCatch
    = (    (   mdpInfo::is_ball_catchable() 
            ||    (mdpInfo::is_ball_catchable_exact()
               && (   (WSinfo::ball->pos + WSinfo::ball->vel).getX()
                    < -ServerOptions::pitch_length/2.0
                  ))
           ) 
        && !Goalie_Bs03::catch_ban );
  if ( standardGoalieBehaviorPerformsCatch == true )
  {
    TGgtLOGPOL(0,<<"GoalieTackle08: Actually, no GT situation: The "
      <<"standard goalie behavior (Goalie03) wwould perform a CATCH now!");
    if ( mdpInfo::is_ball_catchable() == false
         || 1
       )
    {
      TGgtLOGPOL(0,<<"GoalieTackle08: However, Goalie03 would make an"
        <<" unrealiable catch - therefore, it might be better to catch"
        <<" the ball ourselves.");
    }
    else
    {
      ivIsGoalieTackleSituation = false;
      return false;
    }
  }    
  
  TGgtLOGPOL(0,"GoalieTackle08: WOW! We probably have a GT situation.");
  ivIsGoalieTackleSituation = true;
  this->updateDirectActionInformation();
  return true;
}

void 
GoalieTackle08::reset_intention() 
{
}

double
GoalieTackle08::skalarprodukt(Vector v1, Vector v2) 
{
  return v1.getX()*v2.getX() + v1.getY()*v2.getY();
} 

void 
GoalieTackle08::updateCatchAndTackleInformation()
{
  Vector myPos = WSinfo::me->pos,               myNewPos,
         myVel = WSinfo::me->vel,               myNewVel,
         ballPos = WSinfo::ball->pos,           ballNewPos,
         ballVel = WSinfo::ball->vel,           ballNewVel;
  ANGLE  myANG = WSinfo::me->ang,               myNewANG;
  Vector ballOffset = ballVel,
         nextBallVel = ballVel;
  nextBallVel *= ServerOptions::ball_decay;
  Vector ballPosNextCycle = ballPos + ballOffset;
  ivTackleAndCatchProbabilityInformation.ivNextBallPosition = ballPosNextCycle;
  Vector myOffset = myVel;
  Vector myPosNextCycleWhenNotDashing = myPos + myOffset;
  ivBestCatchIndex = -1, ivBestTackleIndex = -1;
  double bestCatchProb = 0.0, bestTackleProb = 0.0;
  //special case: i must act now, because ball is likely to cross the goal
  //              line right now
  double discountFutureActions1Step = 1.0,
         discountFutureActions2Step = 1.0,
         discountFutureActions3Step = 1.0,
         usedDiscounting;
  if ( ivpFutureBallPositions[1].getX() < -FIELD_BORDER_X )
  {
    if (   ivpFutureBallPositions[1].getX()
         < -FIELD_BORDER_X - cvcExpectedPerceptionNoise )
      discountFutureActions1Step = 0.0;
    else if (   ivpFutureBallPositions[1].getX()
              > -FIELD_BORDER_X + cvcExpectedPerceptionNoise )
      discountFutureActions1Step = 1.0;
    else
      discountFutureActions1Step
        =   (ivpFutureBallPositions[1].getX() - (-FIELD_BORDER_X-cvcExpectedPerceptionNoise))
          / (2.0*cvcExpectedPerceptionNoise);
    TGgtLOGPOL(0,<<"GoalieTackle08: I have to discount future actions, because"
      <<" ball may cross goal line: next x pos is "<<ivpFutureBallPositions[1].getX()
      <<", discount factor is "<<discountFutureActions1Step);
  }
  if ( ivpFutureBallPositions[2].getX() < -FIELD_BORDER_X )
  {
    if (   ivpFutureBallPositions[2].getX()
         < -FIELD_BORDER_X - cvcExpectedPerceptionNoise )
      discountFutureActions2Step = 0.0;
    else if (   ivpFutureBallPositions[2].getX()
              > -FIELD_BORDER_X + cvcExpectedPerceptionNoise )
      discountFutureActions2Step = 1.0;
    else
      discountFutureActions2Step
        =   (ivpFutureBallPositions[2].getX() - (-FIELD_BORDER_X-cvcExpectedPerceptionNoise))
          / (2.0*cvcExpectedPerceptionNoise);
    TGgtLOGPOL(0,<<"GoalieTackle08: I have to discount future future "
      <<"actions (OVER-NEXT step), because"
      <<" ball may cross goal line: over-next x pos is "
      <<ivpFutureBallPositions[2].getX()
      <<", discount factor is "<<discountFutureActions2Step);
  }
  if ( ivpFutureBallPositions[3].getX() < -FIELD_BORDER_X )
  {
    if (   ivpFutureBallPositions[3].getX()
         < -FIELD_BORDER_X - cvcExpectedPerceptionNoise )
      discountFutureActions3Step = 0.0;
    else if (   ivpFutureBallPositions[3].getX()
              > -FIELD_BORDER_X + cvcExpectedPerceptionNoise )
      discountFutureActions3Step = 1.0;
    else
      discountFutureActions3Step
        =   (ivpFutureBallPositions[3].getX() - (-FIELD_BORDER_X-cvcExpectedPerceptionNoise))
          / (2.0*cvcExpectedPerceptionNoise);
    TGgtLOGPOL(0,<<"GoalieTackle08: I have to discount future future "
      <<"actions (OVER-OVER-NEXT step), because"
      <<" ball may cross goal line: over-over-next x pos is "
      <<ivpFutureBallPositions[3].getX()
      <<", discount factor is "<<discountFutureActions3Step);
  }
  //main loop
  for ( int index = 0; 
        index < TackleAndCatchProbabilityInformationHolder::cvcNUMBEROFACTIONS;
        index ++ )
  {
    int currentActionType, currentActionParameter;
    ivTackleAndCatchProbabilityInformation.indexToAction( index,
                                                          currentActionType,
                                                          currentActionParameter);
    Cmd dummyCmd;
    switch (currentActionType)
    {
      case cvcDASH:
      {
        dummyCmd.cmd_body.set_dash( currentActionParameter );
        TGgtLOGPOL(2,<<"GoalieTackle08: Model a DASH("<<currentActionParameter<<")");
        Tools::model_cmd_main( myPos,
                               myVel,
                               myANG,
                               ballPos,
                               ballVel,
                               dummyCmd.cmd_body,
                               myNewPos,
                               myNewVel,
                               myNewANG,
                               ballNewPos,
                               ballNewVel);
        usedDiscounting = discountFutureActions1Step;
        break;
      }
      case cvcTURNtoBALL:
      {
        ANGLE targetAngle
          = (ballPosNextCycle - myPosNextCycleWhenNotDashing).ARG();
        Angle turnMoment = (targetAngle - WSinfo::me->ang).get_value();
        ivpBasicCmdBehavior->set_turn_inertia( turnMoment );
        ivpBasicCmdBehavior->get_cmd( dummyCmd );
        Angle performedTurn;
        dummyCmd.cmd_body.get_turn( performedTurn );
        Tools::model_cmd_main( myPos,
                               myVel,
                               myANG,
                               ballPos,
                               ballVel,
                               dummyCmd.cmd_body,
                               myNewPos,
                               myNewVel,
                               myNewANG,
                               ballNewPos,
                               ballNewVel);
        usedDiscounting = discountFutureActions1Step;
        TGgtLOGPOL(2,<<"GoalieTackle08: Model a TURN("
          <<RAD2DEG(performedTurn)<<") [turn to ball], myNewANG="
          <<RAD2DEG(myNewANG.get_value_mPI_pPI()));
        TGgtLOGPOL(0,<<_2D<<VC2D(myNewPos,0.5,"333300"));
        break;   
      }
      case cvcCATCHORTACKLENOW:
      {
        //catch/tackle concern current cycle -> nothing to be modelled
        myNewPos   = myPos;          myNewVel   = myVel;
        myNewANG   = myANG;
        ballNewPos = ballPos;        ballNewVel = ballVel;
        usedDiscounting = 1.0;
        break;
      }
      case cvcDOUBLEDASHP:
      case cvcDOUBLEDASHM:
      {
#if LOGGING && BASIC_LOGGING
        int dashPower = (currentActionType == cvcDOUBLEDASHP) ? 100 : -100;
#endif
        dummyCmd.cmd_body.set_dash( currentActionParameter );
        TGgtLOGPOL(2,<<"GoalieTackle08: Model a DOUBLE-DASH("<<dashPower<<")");
        Vector intermediateMyPos, intermediateMyVel,
               intermediateBallPos, intermediateBallVel;
        ANGLE  intermediateMyANG;
        Tools::model_cmd_main( myPos,
                               myVel,
                               myANG,
                               ballPos,
                               ballVel,
                               dummyCmd.cmd_body,
                               intermediateMyPos,
                               intermediateMyVel,
                               intermediateMyANG,
                               intermediateBallPos,
                               intermediateBallVel);
        Tools::model_cmd_main( intermediateMyPos,
                               intermediateMyVel,
                               intermediateMyANG,
                               intermediateBallPos,
                               intermediateBallVel,
                               dummyCmd.cmd_body,
                               myNewPos,
                               myNewVel,
                               myNewANG,
                               ballNewPos,
                               ballNewVel);
        usedDiscounting = discountFutureActions2Step;
        break;
      }
      case cvcTURNandDASHP100:
      {
        //first: turn
        ANGLE targetAngle
          //= (ballPosOverNextCycle - myPosNextCycleWhenNotDashing).ARG();
          = (ivpFutureBallPositions[2] - myPosNextCycleWhenNotDashing).ARG();
        Angle turnMoment = (targetAngle - WSinfo::me->ang).get_value();
        ivpBasicCmdBehavior->set_turn_inertia( turnMoment );
        ivpBasicCmdBehavior->get_cmd( dummyCmd );
        Angle performedTurn;
        dummyCmd.cmd_body.get_turn( performedTurn );
        Vector intermediateMyPos, intermediateMyVel,
               intermediateBallPos, intermediateBallVel;
        ANGLE  intermediateMyANG;
        Tools::model_cmd_main( myPos,
                               myVel,
                               myANG,
                               ballPos,
                               ballVel,
                               dummyCmd.cmd_body,
                               intermediateMyPos,
                               intermediateMyVel,
                               intermediateMyANG,
                               intermediateBallPos,
                               intermediateBallVel);
        TGgtLOGPOL(2,<<"GoalieTackle08: Model a TURN("
          <<RAD2DEG(performedTurn)<<") [TURNandDASH], myNewANG="
          <<RAD2DEG(myNewANG.get_value_mPI_pPI()));
        TGgtLOGPOL(0,<<_2D<<VC2D(myNewPos,0.5,"333300"));
        //second: dash
        dummyCmd.cmd_body.unset_lock();
        dummyCmd.cmd_body.unset_cmd();
        dummyCmd.cmd_body.set_dash( 100.0 );
        TGgtLOGPOL(2,<<"GoalieTackle08: Model a DASH(100) [TURNandDASH]");
        Tools::model_cmd_main( intermediateMyPos,
                               intermediateMyVel,
                               intermediateMyANG,
                               intermediateBallPos,
                               intermediateBallVel,
                               dummyCmd.cmd_body,
                               myNewPos,
                               myNewVel,
                               myNewANG,
                               ballNewPos,
                               ballNewVel);
        usedDiscounting = discountFutureActions2Step;
        break;
      }
      case cvcTURNandDASHP100andDASHP100:
      {
        //first: turn
        ANGLE targetAngle
          = (ivpFutureBallPositions[3] - myPosNextCycleWhenNotDashing).ARG();
        Angle turnMoment = (targetAngle - WSinfo::me->ang).get_value();
        ivpBasicCmdBehavior->set_turn_inertia( turnMoment );
        ivpBasicCmdBehavior->get_cmd( dummyCmd );
        Angle performedTurn;
        dummyCmd.cmd_body.get_turn( performedTurn );
        Vector intermediateMyPos, intermediateMyVel,
               intermediateBallPos, intermediateBallVel;
        ANGLE  intermediateMyANG;
        Tools::model_cmd_main( myPos,
                               myVel,
                               myANG,
                               ballPos,
                               ballVel,
                               dummyCmd.cmd_body,
                               intermediateMyPos,
                               intermediateMyVel,
                               intermediateMyANG,
                               intermediateBallPos,
                               intermediateBallVel);
        TGgtLOGPOL(2,<<"GoalieTackle08: Model a TURN("
          <<RAD2DEG(performedTurn)<<") [TURNandDASHandDASH], myNewANG="
          <<RAD2DEG(myNewANG.get_value_mPI_pPI()));
        TGgtLOGPOL(0,<<_2D<<VC2D(myNewPos,0.5,"333300"));
        //second: dash
        Vector intermediateMyPos2, intermediateMyVel2,
               intermediateBallPos2, intermediateBallVel2;
        ANGLE  intermediateMyANG2;
        dummyCmd.cmd_body.unset_lock();
        dummyCmd.cmd_body.unset_cmd();
        dummyCmd.cmd_body.set_dash( 100.0 );
        TGgtLOGPOL(2,<<"GoalieTackle08: Model a DASH(100) [TURNandDASHandDASH]");
        Tools::model_cmd_main( intermediateMyPos,
                               intermediateMyVel,
                               intermediateMyANG,
                               intermediateBallPos,
                               intermediateBallVel,
                               dummyCmd.cmd_body,
                               intermediateMyPos2,
                               intermediateMyVel2,
                               intermediateMyANG2,
                               intermediateBallPos2,
                               intermediateBallVel2);
        //third: another dash
        dummyCmd.cmd_body.unset_lock();
        dummyCmd.cmd_body.unset_cmd();
        dummyCmd.cmd_body.set_dash( 100.0 );
        TGgtLOGPOL(2,<<"GoalieTackle08: Model a DASH(100) [TURNandDASHandDASH]");
        Tools::model_cmd_main( intermediateMyPos2,
                               intermediateMyVel2,
                               intermediateMyANG2,
                               intermediateBallPos2,
                               intermediateBallVel2,
                               dummyCmd.cmd_body,
                               myNewPos,
                               myNewVel,
                               myNewANG,
                               ballNewPos,
                               ballNewVel);
        usedDiscounting = discountFutureActions3Step;
        break;
      }
      default: 
      {
        TGgtLOGPOL(0,<<"GoalieTackle08: ERROR: Unknown action ("
          <<currentActionType<<").");
      }
    }
    ivTackleAndCatchProbabilityInformation
      .calculateTackleAndCatchProbabilityForIndex
       (index, myNewPos, myNewANG, ballNewPos, usedDiscounting);
    TGgtLOGPOL(2,<<"GoalieTackle08: i="<<index<<": tackProb="
      <<ivTackleAndCatchProbabilityInformation.ivTackleProbabilities[index]
      <<" catchProb="
      <<ivTackleAndCatchProbabilityInformation.ivCatchProbabilities[index]
      <<" (dist="<<myNewPos.distance(ballNewPos)<<")");
    if (   ivTackleAndCatchProbabilityInformation.ivTackleProbabilities[index]
         > bestTackleProb )
    {
      bestTackleProb 
        = ivTackleAndCatchProbabilityInformation.ivTackleProbabilities[index];
      ivBestTackleIndex = index;
    }
    if (     ivTackleAndCatchProbabilityInformation.ivCatchProbabilities[index]
           > bestCatchProb
        || (    ivTackleAndCatchProbabilityInformation.ivCatchProbabilities[index] >= 1.0
             && currentActionType == cvcCATCHORTACKLENOW )
       )
    {
      bestCatchProb 
        = ivTackleAndCatchProbabilityInformation.ivCatchProbabilities[index];
      ivBestCatchIndex = index;
    }
  }
}

void 
GoalieTackle08::updateCatchingInterceptInformation( Cmd * cmd )
{
  //point where goalie can catch the ball in the best case
  Player copiedGoalie;
  copiedGoalie.alive        = true;
  copiedGoalie.number       = WSinfo::me->number;
  copiedGoalie.team         = WSinfo::me->team;
  copiedGoalie.time         = WSinfo::me->time;
  copiedGoalie.age          = WSinfo::me->age;
  copiedGoalie.age_vel      = WSinfo::me->age_vel;
  copiedGoalie.age_ang      = WSinfo::me->age_ang;
  copiedGoalie.pos          = WSinfo::me->pos;
  copiedGoalie.vel          = WSinfo::me->vel;
  copiedGoalie.ang          = WSinfo::me->ang;
  copiedGoalie.neck_ang     = WSinfo::me->neck_ang;
  copiedGoalie.neck_ang_rel = WSinfo::me->neck_ang_rel;
  copiedGoalie.stamina      = WSinfo::me->stamina;
  copiedGoalie.effort       = WSinfo::me->effort;
  copiedGoalie.recovery     = WSinfo::me->recovery;
  copiedGoalie.stamina_capacity = WSinfo::me->stamina_capacity;
  copiedGoalie.tackle_flag  = WSinfo::me->tackle_flag;
  copiedGoalie.tackle_time  = WSinfo::me->tackle_time;
  copiedGoalie.action_time  = WSinfo::me->action_time;
  copiedGoalie.pointto_flag = WSinfo::me->pointto_flag;
  copiedGoalie.pointto_dir  = WSinfo::me->pointto_dir;
  copiedGoalie.direct_opponent_number = WSinfo::me->direct_opponent_number;
  //copiedGoalie.pass_info <- no adaptation
  copiedGoalie.radius       = WSinfo::me->radius;
  copiedGoalie.speed_max    = WSinfo::me->speed_max;
  copiedGoalie.dash_power_rate = WSinfo::me->dash_power_rate;
  copiedGoalie.decay        = WSinfo::me->decay;
  copiedGoalie.stamina_inc_max = WSinfo::me->stamina_inc_max;
  copiedGoalie.inertia_moment=WSinfo::me->inertia_moment;
  copiedGoalie.stamina_demand_per_meter = WSinfo::me->stamina_demand_per_meter;
  copiedGoalie.kick_radius  = WSinfo::me->kick_radius;
  copiedGoalie.kick_rand_factor = WSinfo::me->kick_rand_factor;
  //change the value of the "kick" radius of the copied goalie
  //to the catch radius :-) ==> thus from 1.085 to 1.2 (1.3)
  copiedGoalie.kick_radius = ServerOptions::catchable_area_l;
  /*copiedGoalie.kick_radius = cvcRealCatchRadius;*/
  //estimated number of steps to catch the ball
  Cmd dummyCmd;
  if ( cmd == NULL )
    ivpInterceptBallBehavior->get_cmd_arbitraryPlayer
                              ( &copiedGoalie,
                                dummyCmd,
                                copiedGoalie.pos,
                                copiedGoalie.vel,
                                copiedGoalie.ang,
                                WSinfo::ball->pos,
                                WSinfo::ball->vel,
                                ivNumberOfStepsToCatchTheBall,
                                50 //max steps to check
                              );
  else
    ivpInterceptBallBehavior->get_cmd_arbitraryPlayer
                              ( &copiedGoalie,
                                *cmd,
                                copiedGoalie.pos,
                                copiedGoalie.vel,
                                copiedGoalie.ang,
                                WSinfo::ball->pos,
                                WSinfo::ball->vel,
                                ivNumberOfStepsToCatchTheBall,
                                50 //max steps to check
                              );
  TGgtLOGPOL(1,<<"GoalieTackle08: Goalie needs "
    <<ivNumberOfStepsToCatchTheBall<<" steps to have ball in its catch area.");
  if ( ivNumberOfStepsToCatchTheBall < cvcFutureBallSteps )
  {
    TGgtLOGPOL(1,<<_2D<<VC2D(ivpFutureBallPositions[ivNumberOfStepsToCatchTheBall],
                            0.3, "0000aa"));
  }
}

void
GoalieTackle08::updateDirectActionInformation()
{
  //summarizing variables
  double catchProbNow  = 0.0, catchProbNext  = 0.0,
         tackleProbNow = 0.0, tackleProbNext = 0.0;
  catchProbNow  
    = ivTackleAndCatchProbabilityInformation.ivCatchProbabilities
      [TackleAndCatchProbabilityInformationHolder::cvcCATCHORTACKLENOWINDEX];
  tackleProbNow 
    = ivTackleAndCatchProbabilityInformation.ivTackleProbabilities
      [TackleAndCatchProbabilityInformationHolder::cvcCATCHORTACKLENOWINDEX];
  if (   ivBestCatchIndex > -1 
      &&   ivBestCatchIndex 
         < TackleAndCatchProbabilityInformationHolder::cvcTURNtoBALLINDEX)
    catchProbNext = ivTackleAndCatchProbabilityInformation
                    .ivCatchProbabilities[ivBestCatchIndex];
  if (   ivBestTackleIndex > -1 
      &&    ivBestTackleIndex 
         <= TackleAndCatchProbabilityInformationHolder::cvcTURNandDASHP100andDASHP100INDEX)
    tackleProbNext = ivTackleAndCatchProbabilityInformation
                     .ivTackleProbabilities[ivBestTackleIndex];
  TGgtLOGPOL(1,<<"GoalieTackle08: Catching now is successful with p="
    <<catchProbNow<<", next step with p="<<catchProbNext);
  TGgtLOGPOL(1,<<"GoalieTackle08: Tackling now is successful with p="
    <<tackleProbNow<<", next step with p="<<tackleProbNext);

  //info & near opps
  TGgtLOGPOL(1,<<"GoalieTackle08: INFO: catch_ban="
    <<Goalie_Bs03::catch_ban<<" PM="<<WSinfo::ws->play_mode
    <<" hisPen="<<PM_his_PenaltyKick<<" ivIsGTSit="
    <<ivIsGoalieTackleSituation); 
  PlayerSet opps = WSinfo::valid_opponents;
  PPlayer closestOpponent
    = opps.closest_player_to_point( WSinfo::ball->pos );
  bool ballIsKickableForAnOpponent
    =     closestOpponent != NULL 
       &&   closestOpponent->pos.distance(WSinfo::ball->pos)
          < closestOpponent->kick_radius;
    
  //update aggressive catching decision
  double catchSuccProbThrs4Penalty = 0.5;
  ivAggressiveCatchIsAdvisable = false;
  if (    (   (   catchProbNow >  0.0
               && catchProbNow >= catchProbNext
               && catchProbNow >= tackleProbNow
               && catchProbNow >= tackleProbNext 
               && (   WSinfo::ws->play_mode != PM_his_PenaltyKick
                   || //in penalty mode:
                      catchProbNow > catchSuccProbThrs4Penalty
                   || ivIsGoalieTackleSituation
                  )
              )
            || catchProbNow >= 1.0
          )
       && Goalie_Bs03::catch_ban == false )
  {
    ivAggressiveCatchIsAdvisable = true;
    TGgtLOGPOL(1,<<"GoalieTackle08: An aggressive catching may be "
      <<"advisable.");
  }
  //update aggressive catching decision (next step)
  ivAggressiveCatchIsAdvisableNextStep = false;
  if (    catchProbNext >  0.0
       && catchProbNext >= catchProbNow
       && catchProbNext >= tackleProbNow
       && catchProbNext >= tackleProbNext
       && (   WSinfo::ws->play_mode != PM_his_PenaltyKick
           || //in penalty mode:
              catchProbNext > catchSuccProbThrs4Penalty
           || ivIsGoalieTackleSituation
          )
       && Goalie_Bs03::catch_ban <= 2 )
  {
    if (Goalie_Bs03::catch_ban <= 1) 
      ivAggressiveCatchIsAdvisableNextStep = true;
    TGgtLOGPOL(1,<<"GoalieTackle08: An aggressive catching may be "
      <<"advisable in the NEXT STEP.");
    if (      ivBestCatchIndex 
           == TackleAndCatchProbabilityInformationHolder::cvcDOUBLEDASHPINDEX 
        ||    ivBestCatchIndex
           == TackleAndCatchProbabilityInformationHolder::cvcDOUBLEDASHMINDEX )
    {
      TGgtLOGPOL(1,<<"GoalieTackle08: Note: Actually, in the OVER-NEXT cycle.");
      ivAggressiveCatchIsAdvisableNextStep = true;
    }
  } 
  //update immediate tackle information
  Quadrangle2d checkArea( WSinfo::ball->pos, MY_GOAL_CENTER, 1.5, 14.0 );
  double tackSuccProbThrs4Penalty = 0.9; //TG09: erhoeht von 0.5; TG2010: reduziert von 0.975 auf 0.8
  if (   WSinfo::me->pos.distance(MY_GOAL_CENTER) //TG09
       > WSinfo::ball->pos.distance(MY_GOAL_CENTER) )
    tackSuccProbThrs4Penalty = 0.8; //TG2010: alt: 0.9
  if (checkArea.inside(WSinfo::me->pos) == false ) //TG2010
    tackSuccProbThrs4Penalty = 0.65;
  if (WSinfo::ball->pos.distance(MY_GOAL_CENTER) < 20.0)//TG09
    tackSuccProbThrs4Penalty = 0.5;
  ivImmediateTacklingIsAdvisable = false;
  if (    tackleProbNow >  0.0
       && tackleProbNow >= catchProbNow
       && (   tackleProbNow >= catchProbNext
           || (   WSinfo::ws->play_mode == PM_his_PenaltyKick
               && ballIsKickableForAnOpponent
               && tackleProbNow >= tackSuccProbThrs4Penalty ) )
       && (   tackleProbNow >= tackleProbNext
           || (   WSinfo::ws->play_mode == PM_his_PenaltyKick
               && ballIsKickableForAnOpponent 
               && tackleProbNow >= tackSuccProbThrs4Penalty ) )
       && (   WSinfo::ws->play_mode != PM_his_PenaltyKick
           || //in penalty mode:
              tackleProbNow > tackSuccProbThrs4Penalty
           || ivIsGoalieTackleSituation
          )
     )
  {
    ivImmediateTacklingIsAdvisable = true;
    TGgtLOGPOL(1,<<"GoalieTackle08: An immediate tackling may be "
      <<"advisable.");
  } 
  //update immediate tackle information (next step)
  ivTacklePreparationIsAdvisable = false;
  if (    tackleProbNext >  0.0
       && tackleProbNext >= catchProbNow
       && tackleProbNext >= catchProbNext
       && tackleProbNext >= tackleProbNow
       && (   WSinfo::ws->play_mode != PM_his_PenaltyKick
           || //in penalty mode:
              tackleProbNext > tackSuccProbThrs4Penalty
           || ivIsGoalieTackleSituation
          )
     )
  {
    ivTacklePreparationIsAdvisable = true;
    TGgtLOGPOL(1,<<"GoalieTackle08: A tackle preparation may be "
      <<"advisable.");
    if (      ivBestTackleIndex 
           == TackleAndCatchProbabilityInformationHolder::cvcDOUBLEDASHPINDEX 
        ||    ivBestTackleIndex 
           == TackleAndCatchProbabilityInformationHolder::cvcDOUBLEDASHMINDEX
        ||    ivBestTackleIndex
           == TackleAndCatchProbabilityInformationHolder::cvcTURNandDASHP100INDEX
       )
    {
      TGgtLOGPOL(1,<<"GoalieTackle08: Note: Actually, in the OVER-NEXT cycle.");
    }
    if (    ivBestTackleIndex 
         == TackleAndCatchProbabilityInformationHolder::cvcTURNandDASHP100andDASHP100INDEX )
    {
      TGgtLOGPOL(1,<<"GoalieTackle08: Note: Actually, in the OVER-OVER-NEXT cycle.");
    }
  } 
}

void
GoalieTackle08::updateForCurrentCycle()
{
  this->updateFutureBallPositions();
  this->updateCatchingInterceptInformation();
  this->updateCatchAndTackleInformation();
  TGgtLOGPOL(0,<<"GoalieTackle08: Best catch index is "<<ivBestCatchIndex
               <<", best tackle index is "<<ivBestTackleIndex);
  this->updateDirectActionInformation();
}

void
GoalieTackle08::updateFutureBallPositions()
{
  //future ball positions
  Vector currentBallPosition = WSinfo::ball->pos, 
         currentBallVelocity = WSinfo::ball->vel;
  ivBallCrossesMyGoalLineInNSteps = cvcFutureBallSteps;
  for (int t=0; t<cvcFutureBallSteps; t++)
  {
    ivpFutureBallPositions[t] = currentBallPosition;
    TGgtLOGPOL(1,<<_2D<<VC2D(currentBallPosition,
                            0.085, "5555ff"));
    currentBallPosition += currentBallVelocity;
    currentBallVelocity *= ServerOptions::ball_decay;
    if (   ivBallCrossesMyGoalLineInNSteps == cvcFutureBallSteps
        && ivpFutureBallPositions[t].getX() < -FIELD_BORDER_X )
    {
      ivBallCrossesMyGoalLineInNSteps = t;
      TGgtLOGPOL(2,<<"GoalieTackle08: Ball crosses goal line in "
        <<ivBallCrossesMyGoalLineInNSteps<<" steps.");
    }
  }
  //calculate distance to ball
  ivMyDistanceToBall = WSinfo::me->pos.distance(WSinfo::ball->pos); 
  //calculate distance to ball in next step
  ivMyDistanceToBallNextStep = (WSinfo::me->pos+WSinfo::me->vel).distance
                                               (ivpFutureBallPositions[1]);
  TGgtLOGPOL(1,<<"GoalieTackle08: Entered. Distance to ball "
    <<ivMyDistanceToBall<<" next step distance "<<ivMyDistanceToBallNextStep);
}




/*
//OLD, BUT POTENTIALLY INTERESTING CODE


bool Goalie03::test_catch(Cmd &cmd) {
  LOG_DAN(0, << "catchable?????????");
  if (!LEFT_PENALTY_AREA.inside(my_pos)){
    LOG_DAN(0, << "not inside my penalty area");
    return 0;
  }
  if ( is_ball_catchable ){ // If the ball is catchable
    LOG_DAN(0, << "The ball is catchable");
    Goalie_Bs03::i_have_ball = true;
    Goalie_Bs03::catch_ban = ServerOptions::catch_ban_cycle;
    last_catch_time = WSinfo::ws->time;
    basic_cmd->set_catch(Tools::my_angle_to(WSinfo::ball->pos));
    basic_cmd->get_cmd(cmd);
    return true;
  }
  return false;
} 

*/






void
GoalieTackle08::TackleAndCatchProbabilityInformationHolder
  ::indexToAction(int index, int & action, int & actionPar)
{
  if ( index == cvcTURNtoBALLINDEX )
  {
    action = GoalieTackle08::cvcTURNtoBALL;
    actionPar = 0;
  }
  else if ( index == cvcCATCHORTACKLENOWINDEX )
  {
    action = GoalieTackle08::cvcCATCHORTACKLENOW;
    actionPar = 0;
  }
  else if ( index == cvcDOUBLEDASHPINDEX )
  {
    action = GoalieTackle08::cvcDOUBLEDASHP;
    actionPar = 100;
  }
  else if ( index == cvcDOUBLEDASHMINDEX )
  {
    action = GoalieTackle08::cvcDOUBLEDASHM;
    actionPar = -100;
  }
  else if ( index == cvcTURNandDASHP100INDEX )
  {
    action = GoalieTackle08::cvcTURNandDASHP100;
    actionPar = 0;
  }
  else if ( index == cvcTURNandDASHP100andDASHP100INDEX )
  {
    action = GoalieTackle08::cvcTURNandDASHP100andDASHP100;
    actionPar = 0;
  }
  else // index ==> do a dash
  {
    action = GoalieTackle08::cvcDASH;
    double delta
      = 200.0 / ((double)(GoalieTackle08::cvcNumberOfModelledDashes-1));
    actionPar = (int)(-100.0 + delta * (double)index);
  }
}

void 
GoalieTackle08::TackleAndCatchProbabilityInformationHolder
  ::actionToIndex(int action, int actionPar, int &index)
{
  switch (action)
  {
    case GoalieTackle08::cvcDASH:
    {
      double delta
        = 200.0 / ((double)(GoalieTackle08::cvcNumberOfModelledDashes-1));
      index = (int) (((double)actionPar + 100.0) / delta);
      break;
    }
    case GoalieTackle08::cvcTURNtoBALL:
    {
      index = cvcTURNtoBALLINDEX;
      break;
    }
    case GoalieTackle08::cvcCATCHORTACKLENOW:
    {
      index = cvcCATCHORTACKLENOWINDEX;
      break;
    }
    case GoalieTackle08::cvcDOUBLEDASHP:
    {
      index = cvcDOUBLEDASHPINDEX;
      break;
    }
    case GoalieTackle08::cvcDOUBLEDASHM:
    {
      index = cvcDOUBLEDASHMINDEX;
      break;
    }
    case GoalieTackle08::cvcTURNandDASHP100:
    {
      index = cvcTURNandDASHP100INDEX;
      break;
    }
    case GoalieTackle08::cvcTURNandDASHP100andDASHP100:
    {
      index = cvcTURNandDASHP100andDASHP100INDEX;
      break;
    }
    default:
    {}
  }
  
}

void
GoalieTackle08::TackleAndCatchProbabilityInformationHolder
  ::calculateTackleAndCatchProbabilityForIndex
    (int index, Vector myPos, ANGLE myANG, Vector ballPos,
     double discountFutureActions)
{
  //calculate tackle probability
  double tackleSuccessProbability
          = Tools::get_tackle_success_probability( myPos, 
                                                   ballPos, 
                                                   myANG.get_value());
  //TG2010
  PlayerSet opps = WSinfo::valid_opponents;
  PPlayer closestOpponent
    = opps.closest_player_to_point( WSinfo::ball->pos );
  bool ballIsKickableForAnOpponent
    =     closestOpponent != NULL 
       &&   closestOpponent->pos.distance(WSinfo::ball->pos)
          < closestOpponent->kick_radius;
  if (   WSinfo::ws->play_mode == PM_his_PenaltyKick
      && ballIsKickableForAnOpponent
     )
    tackleSuccessProbability
          = Tools::get_tackle_success_probability( myPos, 
                                                   ballPos, 
                                                   myANG.get_value(),
                                                   true);

                                                   
  //calculate catch "probability"
  /*
       |
    1.0+----------------
       |                \
       |                  \
       |                    \
    0.5+                      +
       |                       \
       |                        \
       |                         \
       +---------------+------+---+------------>
                    save   real   bestCase
    Note: We actually allow catch porobabilities larger than 1.0, in
          order to be able to rank different dashes better. 
  */
  double catchSuccessProbability,
        distToBall         = myPos.distance( ballPos ),
        realCatchRange     = GoalieTackle08::cvcRealCatchRadius,
        saveCatchRange     =   realCatchRange
                             - GoalieTackle08::cvcExpectedPerceptionNoise,
        bestCaseCatchRange =   realCatchRange 
                             + GoalieTackle08::cvcExpectedPerceptionNoise; 
  if ( distToBall <= saveCatchRange ) //<1.24
  {
    catchSuccessProbability = 1.0;
    //if ( distToBall > (WSinfo::me->radius+ServerOptions::ball_size)*1.2 )//TG09
    catchSuccessProbability += (saveCatchRange - distToBall);
    //TG09: punish situations with ball to near to body
    if ( distToBall < ServerOptions::player_size + 0.2 )
      catchSuccessProbability += (distToBall - (ServerOptions::player_size+0.2));
  }
  else if ( distToBall <= realCatchRange )
    catchSuccessProbability =   (   (distToBall - saveCatchRange)
                                  / (realCatchRange - saveCatchRange) )
                              * (-0.5) + 1.0; //from 1.0 down to 0.5 
  else if ( distToBall <= bestCaseCatchRange )
    catchSuccessProbability =   (   (distToBall - realCatchRange)
                                  / (bestCaseCatchRange - realCatchRange) )
                              * (-0.5) + 0.5; //from 1.0 down to 0.5 
  else  //i.e. distToBall > bestCaseCatchRange
    catchSuccessProbability = 0.0;
  //TG09: be careful outside the penalty area: WE ARE NOT ALLOWED TO CATCH THERE!!!
  if ( LEFT_PENALTY_AREA.inside(ballPos) == false )
    catchSuccessProbability = 0.0;  
  //store both values
  double usedDiscounting
    = (index == cvcCATCHORTACKLENOWINDEX) ? 1.0 : discountFutureActions;
  if (usedDiscounting < 1.0)
  {
    TGgtLOGPOL(2,<<"GoalieTackle08::TackleAndCatchProbabilityInformationHolder:"
     <<" WARNING: Ball may enter goal, discounting by "<<usedDiscounting);
  } 
  ivCatchProbabilities[index]  = catchSuccessProbability * usedDiscounting;
  ivTackleProbabilities[index] = tackleSuccessProbability * usedDiscounting;
}








