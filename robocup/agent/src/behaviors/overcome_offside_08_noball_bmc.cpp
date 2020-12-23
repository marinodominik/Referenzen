#include "overcome_offside_08_noball_bmc.h"

#include "ws_memory.h"
#include <stdlib.h>

/* initialize some static variables */
bool OvercomeOffside08Noball::cvInitialized = false;	
Vector 
  OvercomeOffside08Noball::cvPassBallPositionArray
                           [OvercomeOffside08Noball::cvc_OOT_FUTURE_BALL_POSITIONS];
const double OvercomeOffside08Noball::cvc_ASSUMED_OOT_PASS_SPEED = 2.7;
const double OvercomeOffside08Noball::cvc_INACCAPTABLE_OOT_PASS  = 1000.0;
PPlayer OvercomeOffside08Noball::cvPassReceivingPlayer = NULL;
long  OvercomeOffside08Noball::cvTimeOfRecentPassRequest = -1;
int   OvercomeOffside08Noball::cvDegAngleOfRecentPassRequest = 0;


bool
OvercomeOffside08Noball
  ::determineBestOOTPassCorridor( PPlayer passStartPlayer,
                                  PPlayer passReceivingPlayer,
                                  double  offsideLine,
                                  int   & bestAngleParameter,
                                  Vector& passInterceptionPoint )
{
  if (passStartPlayer == NULL || passReceivingPlayer == NULL)
  {
    TGooLOGPOL(0,<<"OO08Noball: SEVERE ERROR: Got NULL pointer for"
      <<" pass start/receiving player.");
    return false;
  }
  //set variables
  cvPassReceivingPlayer = passReceivingPlayer;
  Vector passStartPosition 
    = estimateOOTPassStartPosition( passStartPlayer->pos,
                                    offsideLine );
  double bestAngleEvaluation = cvc_INACCAPTABLE_OOT_PASS;
  int minCheckAngle  = -60, //in degrees
      maxCheckAngle  =  60,
      angleIncrement =  10,
      bestAngle, bestIcptStepGuess, icptStepGuess;
  for (int ang=minCheckAngle; ang <=maxCheckAngle; ang += angleIncrement)
  {
    double currentPassValue
      = evaluateOOTPassCorridor( passStartPosition,
                                 ang,
                                 offsideLine,
                                 icptStepGuess );
    if (    currentPassValue < bestAngleEvaluation
         && currentPassValue < 30.0 )
    {
      //new best found
      bestAngleEvaluation = currentPassValue;
      bestAngle = ang;
      bestIcptStepGuess = icptStepGuess;
    }
  }
  if ( bestAngleEvaluation >= cvc_INACCAPTABLE_OOT_PASS )
  {
  TGooLOGPOL(2,<<"OO08Noball: NO suitable OOT pass corridor found. "
    <<"Return false.");
    return false;
  }
  TGooLOGPOL(2,<<"OO08Noball: Suitable OOT pass corridor found. "
    <<"Best pass angle is "<<bestAngle);
  //store time
  cvTimeOfRecentPassRequest     = WSinfo::ws->time;
  //try to enforce consistency!
  if (    cvTimeOfRecentPassRequest == WSinfo::ws->time - 1
       && abs(cvDegAngleOfRecentPassRequest - bestAngle) <= angleIncrement
     )
  {
    TGooLOGPOL(2,<<"==08Noball: ENFORCE CONSISTENCY -> best angle is "
      <<cvDegAngleOfRecentPassRequest<<", instead of "<<bestAngle);
    bestAngle = cvDegAngleOfRecentPassRequest;
  }
  else
    cvDegAngleOfRecentPassRequest = bestAngle;
  //set return values
  bestAngleParameter = bestAngle;
  fillOOTPassBallPositionsArray( passStartPosition,
                                 bestAngleParameter );
  passInterceptionPoint = cvPassBallPositionArray[bestIcptStepGuess]; 
  //debug output
  Vector passEndPoint;
  passEndPoint.init_polar(20.0, DEG2RAD(bestAngleParameter));
  passEndPoint += passStartPosition;
  TGooLOGPOL(0,<<_2D<<VL2D(passStartPosition,
    passEndPoint,"7777ff"));
  TGooLOGPOL(0,<<_2D<<VL2D(cvPassReceivingPlayer->pos,
    cvPassBallPositionArray[bestIcptStepGuess],"aaaaff"));
  for (int i=1; i<(bestIcptStepGuess+3); i++)
    if (i==1 || i==bestIcptStepGuess)
    {TGooLOGPOL(0,<<_2D<<VC2D(cvPassBallPositionArray[i],0.3,"aaaaff"));}
  return true;
}

Vector
OvercomeOffside08Noball
  ::estimateOOTPassStartPosition( Vector startPlayerPos,
                                  double offsideLine )
{
  Vector passStartPosition = startPlayerPos;
  double assumedPassTartPlayerApproachmentTowardsOffsideLine
    = 0.1 * (offsideLine - cvPassReceivingPlayer->pos.getX());
  if (assumedPassTartPlayerApproachmentTowardsOffsideLine > 0.6)
    assumedPassTartPlayerApproachmentTowardsOffsideLine = 0.6;
  if (assumedPassTartPlayerApproachmentTowardsOffsideLine < 0.0)
    assumedPassTartPlayerApproachmentTowardsOffsideLine = 0.0;
  passStartPosition.setX(
          (1.0-assumedPassTartPlayerApproachmentTowardsOffsideLine)
        * startPlayerPos.getX()
      +   assumedPassTartPlayerApproachmentTowardsOffsideLine
        * offsideLine );
  return passStartPosition;
}

double
OvercomeOffside08Noball
  ::evaluateOOTPassCorridor( Vector passStartPosition,
                             int    passAngle, //degree
                             double offsideLine,
                             int  & icptStepGuess)
{
  //large values are bad, small values are good
  double returnValue = 0.0;
  Vector passEndPoint;
  passEndPoint.init_polar(20.0, DEG2RAD(passAngle));
  passEndPoint += passStartPosition;
  //exclude passes with direct opps in passway
  Quadrangle2d directPassWay( passStartPosition, passEndPoint, 0.1, 8.0);
  PlayerSet opps = WSinfo::valid_opponents;
  opps.keep_players_in( directPassWay );
  if ( opps.num > 0 )
  {
    TGooLOGPOL(2,<<"OO08Noball: Must exclude pass direction "<<passAngle
     <<" due to "<<opps.num<<" opponents in DIRECT pass way.");
   return cvc_INACCAPTABLE_OOT_PASS;
  }  
  
  //ok, the pass has not been excluded ... we can continue
  TGooLOGPOL(2,<<"OO08Noball: Evaluate pass starting at "
    <<passStartPosition<<" with angle "<<passAngle);
  
  //fill up future ball positions
  fillOOTPassBallPositionsArray( passStartPosition, passAngle );
  
  //evaluate the earliest possible interception time for me
  int smallestNumberOfIcptSteps
    = getSmallestNumberOfInterceptionSteps( passStartPosition, 
                                            passAngle,
                                            offsideLine );
  if ( smallestNumberOfIcptSteps >= cvc_OOT_FUTURE_BALL_POSITIONS-1 )
  {
    TGooLOGPOL(2,<<"OO08Noball: Must exclude pass direction "<<passAngle
     <<" since I would need too many steps to intercept it.");
   return cvc_INACCAPTABLE_OOT_PASS;
  }  
  TGooLOGPOL(3,<<"OO08Noball: I (real_max_speed="
    <<cvPassReceivingPlayer->speed_max<<") get that pass after "
    <<smallestNumberOfIcptSteps<<" steps at position "
    <<cvPassBallPositionArray[smallestNumberOfIcptSteps]
    <<" ==> Impair pass score by "<<smallestNumberOfIcptSteps<<" points.");
  icptStepGuess = smallestNumberOfIcptSteps;
  returnValue += smallestNumberOfIcptSteps / 2;

  //discard the case when there are many teammates near the position where
  //i may intercept the ball
  PlayerSet ballLeader = WSinfo::valid_teammates_without_me;
  ballLeader.keep_players_in_circle( WSinfo::ball->pos, 3.0 );
  PlayerSet consideredTeammates = WSinfo::valid_teammates_without_me;
  if (ballLeader.num>0) consideredTeammates.remove( ballLeader[0] );
  consideredTeammates.keep_players_in_circle
    ( cvPassBallPositionArray[smallestNumberOfIcptSteps], 
      WSinfo::me->pos
        .distance(cvPassBallPositionArray[smallestNumberOfIcptSteps]) * 0.9 );
  if ( consideredTeammates.num > 0 )
  {
    TGooLOGPOL(2,<<"OO08Noball: Must exclude pass direction "<<passAngle
     <<" since there are "<<consideredTeammates.num<<" teammates very near to"
     <<" the point where I would get the ball finally.");
    return cvc_INACCAPTABLE_OOT_PASS;
  }  

  //discard the case where i get the ball immediately behind the ol
  if (   cvPassBallPositionArray[smallestNumberOfIcptSteps].getX()
       - offsideLine < 3.0 )
  {
    TGooLOGPOL(2,<<"OO08Noball: Must exclude pass direction "<<passAngle
     <<" since I would get the ball directly behind the offside line"
     <<", delta="<<(cvPassBallPositionArray[smallestNumberOfIcptSteps].getX()
       - offsideLine)<<".");
   return cvc_INACCAPTABLE_OOT_PASS;
  }
  //discard the case where i get the ball out of the field
  if (     cvPassBallPositionArray[smallestNumberOfIcptSteps].getX()
         > FIELD_BORDER_X 
       ||  fabs(cvPassBallPositionArray[smallestNumberOfIcptSteps].getY())
         > FIELD_BORDER_Y - 1.5
     )
  {
    TGooLOGPOL(2,<<"OO08Noball: Must exclude pass direction "<<passAngle
     <<" since I would get the ball out of the field"
     <<", at "<<cvPassBallPositionArray[smallestNumberOfIcptSteps]<<".");
   return cvc_INACCAPTABLE_OOT_PASS;
  }

  //evaluate how much the pass brings me beyond the offside line
  int advancementBeyondOL
    =  - (int)( cvPassBallPositionArray[smallestNumberOfIcptSteps].getX()
               - offsideLine ); 
  TGooLOGPOL(3,<<"OO08Noball: I get the ball "
    <<(cvPassBallPositionArray[smallestNumberOfIcptSteps].getX()- offsideLine)
    <<"m behind his offside line. ==> Impair pass score by "
    <<advancementBeyondOL<<" points.");
  returnValue += advancementBeyondOL / 5;

  //useful vectors
  Vector runWayVector =   cvPassBallPositionArray[smallestNumberOfIcptSteps] 
                        - cvPassReceivingPlayer->pos;
  Vector sprintingPlayerStartPosOnOffsideLine
    = Tools::point_on_line( runWayVector,
                            cvPassReceivingPlayer->pos,
                            offsideLine );

  //evaluate the case where i may interfere the ball leader
  Quadrangle2d runWayCheckArea( sprintingPlayerStartPosOnOffsideLine,
                                cvPassReceivingPlayer->pos,
                                3.0, 
                                runWayVector.norm() );
  if ( runWayCheckArea.inside( passStartPosition ) )
  {
    TGooLOGPOL(2,<<"OO08Noball: Must exclude pass direction "<<passAngle
     <<" since I would interfere with the passing player.");
   return cvc_INACCAPTABLE_OOT_PASS;
  }  

  //evaluate the danger produced by his goalie
  PPlayer hisGoalie = WSinfo::his_goalie;
  if (    hisGoalie == NULL
       && WSinfo::ws->his_goalie_number != 0 )
    hisGoalie = WSinfo::alive_opponents
                        .get_player_by_number(WSinfo::ws->his_goalie_number);
  if (hisGoalie)
  {
    if (     cvPassBallPositionArray[smallestNumberOfIcptSteps]
               .distance( hisGoalie->pos )
           > cvPassBallPositionArray[smallestNumberOfIcptSteps]
               .distance( sprintingPlayerStartPosOnOffsideLine ) + 5.0
        && (    RIGHT_PENALTY_AREA.inside
                 ( cvPassBallPositionArray[smallestNumberOfIcptSteps] ) == false
             &&    WSmemory::get_his_goalie_classification() 
                != HIS_GOALIE_OFFENSIVE
           )
       )
    {
      TGooLOGPOL(3,<<"OO08Noball: His goalie represents no danger."); 
    }
    else
    {
      double goalieDanger
        =   cvPassBallPositionArray[smallestNumberOfIcptSteps]
               .distance( sprintingPlayerStartPosOnOffsideLine )
          - cvPassBallPositionArray[smallestNumberOfIcptSteps]
             .distance( hisGoalie->pos );
      TGooLOGPOL(3,<<"OO08Noball: His goalie is dangerous, may intercept"
        <<" the ball as fast as me. ==> Impair pass score by "
        <<goalieDanger);
      returnValue += 1.5 * (double)goalieDanger;
      //goalie will intercept very likely
      if (    (    goalieDanger > 5.0
                && RIGHT_PENALTY_AREA
                   .inside(cvPassBallPositionArray[smallestNumberOfIcptSteps]) ) 
           || (    goalieDanger > 0.0 
                && WSmemory::get_his_goalie_classification() == HIS_GOALIE_OFFENSIVE )  )
      {
        TGooLOGPOL(2,<<"OO08Noball: Must exclude pass direction "<<passAngle
         <<" since his GOALIE will intercept the ball very likely.");
        return cvc_INACCAPTABLE_OOT_PASS;
      }
    }         
  }

  //evaluate the danger produced by opponents nearby
  Quadrangle2d extendedPassWay( passStartPosition, passEndPoint, 4.0, 30.0);
  PlayerSet icptOpps = WSinfo::valid_opponents;
  icptOpps.keep_players_in( extendedPassWay );
  if (icptOpps.num > 0)
  {
    TGooLOGPOL(3,<<"OO08Noball: Pass is dangerous due to "
      <<icptOpps.num<<" opponents in the *EXTENDED* passway.");
  }
  for (int i=0; i<icptOpps.num; i++)
  {
    Vector icptOppLotfuss = Tools::get_Lotfuss( passStartPosition,
                                                passEndPoint,
                                                icptOpps[i]->pos );
    double stepsToIcptOppLotfuss
       =   icptOppLotfuss.distance(passStartPosition)
         / (cvc_ASSUMED_OOT_PASS_SPEED*0.85);
    double icptOppWay
       = Tools::get_dist2_line( passStartPosition,
                                passEndPoint,
                                icptOpps[i]->pos );
    double estimatedOppIcptSteps
       =   (icptOppWay - icptOpps[i]->kick_radius) //subtract kick radius 
         / icptOpps[i]->speed_max; //est.speed

    double impairmentFactor = 10.0;
    double impairment
       =   impairmentFactor
         * Tools::max( 0.0, stepsToIcptOppLotfuss - estimatedOppIcptSteps);
    TGooLOGPOL(3,<<"OO08Noball: Opponent "<<icptOpps[i]->number
      <<" may intercept the ball. ==> Impair pass score by "
      <<impairment<<" points.");
    returnValue += impairment;
  }
  
  TGooLOGPOL(3,<<"OO08Noball: FINAL PASS SCORE (ang="<<passAngle<<") is "
    <<returnValue<<" points.");

  return returnValue;
}

void
OvercomeOffside08Noball
  ::fillOOTPassBallPositionsArray( Vector passStartPosition,
                                   int    passAngle )
{
  Vector currentBallPos = passStartPosition,
         ballVel;
  ballVel.init_polar( cvc_ASSUMED_OOT_PASS_SPEED, DEG2RAD(passAngle) );
  for ( int i=0; i < cvc_OOT_FUTURE_BALL_POSITIONS; i++ )
  {
    cvPassBallPositionArray[i] = currentBallPos;
    currentBallPos += ballVel;
    ballVel *= ServerOptions::ball_decay;
  }
}

int
OvercomeOffside08Noball
  ::getSmallestNumberOfInterceptionSteps( Vector passStartPosition,
                                          int    passAngle,
                                          double offsideLine )
{
  int returnValue = cvc_OOT_FUTURE_BALL_POSITIONS-1;   
  for (int i=0; i<cvc_OOT_FUTURE_BALL_POSITIONS; i++)
  {
    //intercept passes only behind the offside line
    if ( cvPassBallPositionArray[i].getX() < offsideLine )
      continue;
    //intercept passes only using a way that goes forward
    Vector runWayVector =   cvPassBallPositionArray[i] 
                          - cvPassReceivingPlayer->pos;
    Angle runWayAngle = fabs(runWayVector.ARG().get_value_mPI_pPI());
    if ( runWayAngle > 50.0*(PI/180.0) )
      continue;
    Vector sprintingPlayerStartPosOnOffsideLine
      = Tools::point_on_line( runWayVector,
                              cvPassReceivingPlayer->pos,
                              offsideLine );
    double sprintingPlayerWayToGo = sprintingPlayerStartPosOnOffsideLine
                                   .distance( cvPassBallPositionArray[i] );
    double sprintingPlayerRequiredStepsToGo
      =   sprintingPlayerWayToGo 
        / cvPassReceivingPlayer->speed_max;
    sprintingPlayerRequiredStepsToGo += RAD2DEG(runWayAngle) / 8.0;
    if (   sprintingPlayerRequiredStepsToGo < (double)returnValue
        && sprintingPlayerRequiredStepsToGo < i)
    {
      TGooLOGPOL(0,<<"DEBUG: i="<<i<<": from posOnOL="<<sprintingPlayerStartPosOnOffsideLine
      <<" to ball in "<<i<<" steps at thenPos="<<cvPassBallPositionArray[i]
      <<" where i need "<<sprintingPlayerRequiredStepsToGo<<" for a distance"
      <<" of "<<sprintingPlayerWayToGo<<"m");
      returnValue = (int)sprintingPlayerRequiredStepsToGo + 1;
    }
  }
  return returnValue;
}

bool
OvercomeOffside08Noball
  ::isPassTotallyInacceptable( Vector  passStartPosition,   
                               PPlayer passReceivingPlayer, 
                               double  offsideLine,
                               int     passAngle) //in degree, e.g. -30 or 50
{
  //do calculation
  Vector passEndPoint;
  passEndPoint.init_polar(20.0, DEG2RAD(passAngle));
  passEndPoint += passStartPosition;
  Quadrangle2d directPassWay( passStartPosition, passEndPoint, 0.1, 8.0);
  PlayerSet opps = WSinfo::valid_opponents;
  opps.keep_players_in( directPassWay );
  if ( opps.num > 0 )
  {
    TGooLOGPOL(2,<<"OO08Noball: Pass check for angle "<<passAngle
      <<": INACCEPTABLE (opps in direct pass way).");
   return false;
  }  
  //fill up future ball positions
  fillOOTPassBallPositionsArray( passStartPosition, passAngle );
  
  //evaluate the earliest possible interception time for me
  int smallestNumberOfIcptSteps
    = getSmallestNumberOfInterceptionSteps( passStartPosition, 
                                            passAngle,
                                            offsideLine );
  if ( smallestNumberOfIcptSteps >= cvc_OOT_FUTURE_BALL_POSITIONS-1 )
  {
    TGooLOGPOL(2,<<"OO08Noball: Pass check for angle "<<passAngle
      <<": INACCEPTABLE (I would need too many steps to intercept it).");
   return true;
  }  
  TGooLOGPOL(2,<<"OO08Noball: Pass check for angle "<<passAngle
    <<": ACCEPTABLE.");
  return false;
}

bool
OvercomeOffside08Noball
  ::isPassTotallyInacceptable( PPlayer passStartPlayer,   
                               PPlayer passReceivingPlayer, 
                               double  offsideLine,
                               int     passAngle) //in degree, e.g. -30 or 50
{
  //set var
  cvPassReceivingPlayer = passReceivingPlayer;
  //exclude passes with direct opps in passway
  Vector passStartPosition 
    = estimateOOTPassStartPosition( passStartPlayer->pos,
                                    offsideLine );
  return 
    isPassTotallyInacceptable( passStartPosition,   
                               passReceivingPlayer, 
                               offsideLine,
                               passAngle);
}

void
OvercomeOffside08Noball
  ::setPassReceivingPlayer( PPlayer p )
{
  cvPassReceivingPlayer = p;
}

