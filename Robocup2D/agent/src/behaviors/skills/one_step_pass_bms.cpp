#include "one_step_pass_bms.h"

#include "../../basics/wmoptions.h"
#include "ws_memory.h"
#include "intention.h"
#include "positioning.h"

#if 1
#define   TGospLOGPOL(YYY,XXX)        LOG_POL(YYY,XXX)
#else
#define   TGospLOGPOL(YYY,XXX)
#endif

        //TG08: 22->25
#define ONE_STEP_PASS_ATTACK_DISTANCE_FROM_GOAL  (FIELD_BORDER_X-25.0)

//############################################################################
// KLASSENVARIABLEN
//############################################################################
bool OneStepPass::cvInitialized = false;

//############################################################################
// STATISCHE METHODEN
//############################################################################

//============================================================================
// init()
//============================================================================
bool
OneStepPass::init(char const * conf_file, int argc, char const* const* argv)
{
  if (cvInitialized) return true;
  cvInitialized = true;
  //Score05_Sequence wird in dessen Konstruktor initialisiert!
  //==> also nicht noetig:
  //    cvInitialized &= Score05_Sequence::init( conf_file, argc, argv );
  return cvInitialized;
}

//############################################################################
// KONSTRUKTOREN / DESTRUKTOREN
//############################################################################
OneStepPass::OneStepPass()
{
  for (int i=0; i<cvsMEMORYSIZE; i++)
    ivpMemoryRecentBallKickableTimePoints[ i ] = -1;
  ivMemoryCounter = 0;
  ivForeseeMode = false;
  ivLastSuccessfulForesee = -1;
  ivOffsideConqueringChance = false;
  
  ivpScore05SequenceBehavior           = NULL;
}
OneStepPass::~OneStepPass()
{
}

//############################################################################
// INSTANZENMETHODEN
//############################################################################

//============================================================================
// canPlayerReachPositionIn1Step()
//============================================================================
bool
OneStepPass::canPlayerReachPositionIn1Step( PlayerSet relevantOpponents,
                                            Vector position,
                                            PPlayer & icptPlayer )
{
  if (relevantOpponents.num == 0) return false;
  Cmd_Body dummyCmd;
  Vector newPos, newVel; ANGLE newAng; int dummyStamina;
  for (int i=0; i < relevantOpponents.num; i++)
  {
    for (int dash=-100; dash <=100; dash +=25)
    {
      dummyCmd.unset_lock();
      dummyCmd.unset_cmd();
      dummyCmd.set_dash( dash );
      Tools::simulate_player(  relevantOpponents[i]->pos, 
                               relevantOpponents[i]->vel,
                               relevantOpponents[i]->ang, 
                               4000,
                               dummyCmd,
                               newPos, 
                               newVel,
                               newAng, 
                               dummyStamina,
                               relevantOpponents[i]->stamina_inc_max,
                               relevantOpponents[i]->inertia_moment,
                               relevantOpponents[i]->dash_power_rate, 
                               relevantOpponents[i]->effort, 
                               relevantOpponents[i]->decay );
      if (  position.distance(newPos) 
          < relevantOpponents[i]->kick_radius * 1.1 ) //some safety
      {
        icptPlayer = relevantOpponents[i];
        return true;
      }
    }
  }
  return false;
}

//============================================================================
// evaluatePosition()
//============================================================================
void 
OneStepPass::evaluatePosition( Vector  consideredPosition,
                               PPlayer consideredTeammate,
                               double  checkVelocity,
                               int     &teammateScore,
                               int     logLevel )
{
    //punish backward passes in offside conquering scenarios
    if (    ivOffsideConqueringChance == true
         && consideredPosition.getX() < WSinfo::me->pos.getX() )
    { 
      teammateScore -=  30; 
      TGospLOGPOL(logLevel,<<"OneStepPass: Teammate "
        <<consideredTeammate->number<<" is behind me, and I am in offside"
        <<" conquering mode. Score reduced by "<<30
        <<" points. ==> score = "<<teammateScore); 
    }
  
    //punish targets to which the passway is not free at all
    double checkPassWayLength = (consideredPosition - ivBallPosition).norm();
    Vector passWayEnd = (consideredPosition - ivBallPosition);
    passWayEnd.normalize(checkPassWayLength-1.0);
    passWayEnd += ivBallPosition; //to disregard nearby opps
    Quadrangle2d 
      checkAreaPassway
                      ( ivBallPosition, 
                        passWayEnd,
                        2.0,
                        0.4*consideredPosition.distance(ivBallPosition));
    PlayerSet validOpponents = WSinfo::valid_opponents;
    validOpponents.keep_players_in_quadrangle
                      ( ivBallPosition, 
                        passWayEnd,
                        2.0,
                        0.4*consideredPosition.distance(ivBallPosition));
    if ( validOpponents.num > 0 )
    { 
      int decrementFactor = 15;
      if (ivForeseeMode) decrementFactor = 5;
      teammateScore -=  decrementFactor * validOpponents.num; 
      TGospLOGPOL(logLevel,<<"OneStepPass: Teammate "
        <<consideredTeammate->number<<" is a dangerous target due to "
        <<validOpponents.num<<" opponents"
        <<" in passway. Score reduced by "<<(decrementFactor * validOpponents.num)
        <<" points. ==> score = "<<teammateScore); 
    }
    TGospLOGPOL(0,<<_2D<<checkAreaPassway);
                                   
    //punish targets to which the *extended* passway is not free 
    passWayEnd = (consideredPosition - ivBallPosition);
    passWayEnd.normalize(checkPassWayLength-0.15);
    if (    consideredPosition.distance(consideredTeammate->pos) >= 1.0 
         && consideredTeammate->age > 0 ) //TG08: zeile hinzugefuegt
      passWayEnd.normalize(  checkPassWayLength //extend passway check
                           + consideredPosition.distance(consideredTeammate->pos) );
    passWayEnd += ivBallPosition; //to disregard nearby opps
    validOpponents = WSinfo::valid_opponents;
    validOpponents.keep_players_in_quadrangle
                      ( ivBallPosition, 
                        passWayEnd,
                        4.0,
                        0.9*consideredPosition.distance(ivBallPosition));
    if ( validOpponents.num > 0 )
    { 
      TGospLOGPOL(logLevel,<<"OneStepPass: Teammate "
        <<consideredTeammate->number<<" is a dangerous target due to "
        <<validOpponents.num<<" opponents"
        <<" in *extended* passway.");
      for (int vo=0; vo<validOpponents.num; vo++)
      {
        Vector voInterceptLotfuss
          = Tools::get_Lotfuss( ivBallPosition,
                                consideredPosition,
                                validOpponents[vo]->pos );
        double stepsToInterceptLotfuss
          =   voInterceptLotfuss.distance(ivBallPosition)
            / (checkVelocity*0.85);
        double voIcptWay
          = Tools::get_dist2_line( ivBallPosition,
                                   consideredPosition,
                                   validOpponents[vo]->pos );
        double estimatedOppIcptSteps //subtract kick radius
          = (voIcptWay - validOpponents[vo]->kick_radius) / 1.05; //est.speed
        double decrement_1_factor = 10.0;//TG09: 7->10
        //TG09
        Vector icptPasswayInterceptVector = voInterceptLotfuss - validOpponents[vo]->pos;
        if (   validOpponents[vo]->age==0 && validOpponents[vo]->age_vel==0
            && validOpponents[vo]->age_ang==0
            && validOpponents[vo]->vel.norm() >= 0.25
            && fabs((  icptPasswayInterceptVector.ARG() 
                     - validOpponents[vo]->vel.ARG()).get_value_mPI_pPI()) < 45.0/180.0*PI
            && fabs((  icptPasswayInterceptVector.ARG() 
                     - validOpponents[vo]->ang).get_value_mPI_pPI()) < 45.0/180.0*PI
           )
        {
          decrement_1_factor = 20.0;
          TGospLOGPOL(logLevel,<<"OneStepPass: Opponent "<<validOpponents[vo]->number
            <<" is moving very cleverly.");
        }
        if (ivForeseeMode) decrement_1_factor = 3.0;
        double decrement_1
          = decrement_1_factor
            * Tools::max( 0.0, stepsToInterceptLotfuss - estimatedOppIcptSteps);
        //  = 8.0 * (ServerOptions::ball_speed_max / checkVelocity)
        //    * ( 1.0 - ( Tools::get_dist2_line( ivBallPosition,
        //                                       consideredPosition,
        //                                       validOpponents[vo]->pos )
        //                    / opponentMaxDist ) );
        double decrement_2
          = 3.0 * (ServerOptions::ball_speed_max / checkVelocity)
            * ( 1.0 - ( ivBallPosition.distance(validOpponents[vo]->pos)
                        / ivBallPosition.distance(consideredPosition)));
        double decrement_3
          = 3.0 * (ServerOptions::ball_speed_max / checkVelocity)
            * ( 1.0 - ( consideredPosition.distance(validOpponents[vo]->pos)
                        / ivBallPosition.distance(consideredPosition)));
        teammateScore -= (int)( decrement_1 + decrement_2 + decrement_3 );
        TGospLOGPOL(logLevel,<<"OneStepPass: Reduce score w.r.t. opponent"
          <<validOpponents[vo]->number<<" by "
          <<decrement_1<<"+"<<decrement_2<<"+"<<decrement_3<<" = "
          <<(int)( decrement_1 + decrement_2 + decrement_3 )<<" points."
          <<" ==> score = "<<teammateScore<<"   [INFO:"
          <<stepsToInterceptLotfuss<<"/"
          <<estimatedOppIcptSteps<<"]");
      }
      //teammateScore -=  8 * validOpponents.num; 
      //TGospLOGPOL(1,<<"OneStepPass: Teammate "
      //  <<consideredTeammate->number<<" is a dangerous target due to "
      //  <<validOpponents.num<<" opponents"
      //  <<" in *extended* passway. Score reduced by "<<(8 * validOpponents.num)
      //  <<" points. ==> score = "<<teammateScore); 
    }
                         
    //punish targets that are surrounded by opponents
    PlayerSet nearbyOpponents = WSinfo::valid_opponents;
      //here: consider the teammate itself (consideredTeammate->pos) and
      //      not the considered target position (consideredPosition)
    nearbyOpponents.keep_players_in_circle(consideredTeammate->pos, 4.0);
    teammateScore -= 7 * nearbyOpponents.num; //TG09: 5->7
    TGospLOGPOL(logLevel,<<"OneStepPass: Teammate "
      <<consideredTeammate->number<<" has "
      <<nearbyOpponents.num<<" opponents nearby. Score reduced by "
      <<(7 * nearbyOpponents.num)
      <<" points. ==> score = "<<teammateScore); 
                                    
    //punish target players that have not been seen recently
    teammateScore -=  1 * consideredTeammate->age;
    TGospLOGPOL(logLevel,<<"OneStepPass: Teammate "
      <<consideredTeammate->number<<" has age "
      <<consideredTeammate->age<<". Score reduced by "
      <<(1 * consideredTeammate->age)
      <<" points. ==> score = "<<teammateScore); 
  
    //punish too short and too long passes (ideal = 12m)
    teammateScore
      -= (int)(fabs(ivBallPosition.distance(consideredPosition) - 12.0));
    TGospLOGPOL(logLevel,<<"OneStepPass: Passway to teammate "
      <<consideredTeammate->number<<" is "
      <<ivBallPosition.distance(consideredPosition)<<"m long. "
      <<"Score reduced by "
      <<(int)(fabs(ivBallPosition.distance(consideredPosition) - 12.0))
      <<" points. ==> score = "<<teammateScore); 
    
    //reward large velocities
    teammateScore += (int)(checkVelocity * 9.0);
    TGospLOGPOL(logLevel,<<"OneStepPass: Resulting pass to teammate "
      <<consideredTeammate->number<<" has velocity of "
      <<checkVelocity<<"m/s. Score increased by "
      <<(int)(checkVelocity * 9.0)
      <<" points. ==> score = "<<teammateScore); 
      
    //punish if velocity does not correspond to pass length
    const double idealNumberOfStepsForPass = 5.0;
    teammateScore 
      -= (int)pow
              ( fabs((  ivBallPosition.distance(consideredPosition)
                 / (0.9*checkVelocity)) - idealNumberOfStepsForPass), 1.5 );
    TGospLOGPOL(logLevel,<<"OneStepPass: Resulting pass to teammate "
      <<consideredTeammate->number<<" has velocity of "
      <<checkVelocity<<"m/s which corresponds not perfectly to passway"
      <<" length of"<<(ivBallPosition.distance(consideredPosition))
      <<"m. Score decreased by "
      <<(int)pow
              ( fabs((  ivBallPosition.distance(consideredPosition)
                 / (0.9*checkVelocity)) - idealNumberOfStepsForPass), 1.5 )
      <<" points. ==> score = "<<teammateScore); 
      
    //reward / punish approaching the goal
    teammateScore 
      += (int)(  ivBallPosition.distance(HIS_GOAL_CENTER)
               - consideredPosition.distance(HIS_GOAL_CENTER));
    TGospLOGPOL(logLevel,<<"OneStepPass: Resulting pass to teammate "
      <<consideredTeammate->number<<" yields an approachment to his goal of "
      <<(ivBallPosition.distance(HIS_GOAL_CENTER)
         - consideredPosition.distance(HIS_GOAL_CENTER))<<"m. Add that to the "
      <<" score. ==> score = "<<teammateScore); 

    //punish target if an opponent can get ball in next cycle
    Vector checkBallVelocity( (consideredPosition-ivBallPosition).ARG() );
    checkBallVelocity.normalize( checkVelocity );
    PlayerSet relevantOpponents = WSinfo::valid_opponents;
    relevantOpponents.keep_players_in_circle
                      ( ivBallPosition + checkBallVelocity, 3.0 );
    PPlayer icptPlayer = NULL;
    if (   canPlayerReachPositionIn1Step( relevantOpponents,
                                          ivBallPosition + checkBallVelocity,
                                          icptPlayer ) 
        && ivForeseeMode == false)
    {
      teammateScore -= 50; 
      TGospLOGPOL(0,<<"OneStepPass: Resulting pass to teammate "
        <<consideredTeammate->number<<" can be intercepted by "
        <<"opponent "<<icptPlayer->number<<" in the next cycle. "
        <<"Score decreased by 50 points. ==> score ="<<teammateScore);
    } 

    //punish targets that are not identical to the teammate
    Vector theConsideredTeammatesPosition = consideredTeammate->pos;
    if (    consideredTeammate->age == 0 
         && consideredTeammate->age_vel == 0
         && consideredTeammate->age_ang == 0 
         && fabs(   (  consideredTeammate->vel.ARG()
                     - consideredTeammate->ang)
                    .get_value_mPI_pPI() )          <  PI / 6.0 )
      theConsideredTeammatesPosition = consideredTeammate->pos
                                       + (1.25*consideredTeammate->vel);
    double teammateMovingImpairmentFactor = 1.0;
    Vector passDelta = consideredPosition - consideredTeammate->pos;
    if (    consideredTeammate->age == 0 
         && consideredTeammate->age_vel == 0 )
    {
      if ( fabs( (passDelta.ARG() - consideredTeammate->vel.ARG())
                 .get_value_mPI_pPI() ) > PI*(45./180.)
         )
        teammateMovingImpairmentFactor = 1.5;
      if (    fabs( (passDelta.ARG() - consideredTeammate->vel.ARG()) //TG09
                    .get_value_mPI_pPI() ) < PI*(30./180.)
           && consideredTeammate->vel.norm() >= 0.25
         )
        teammateMovingImpairmentFactor = 0.5;
      if (    fabs( (passDelta.ARG() - consideredTeammate->vel.ARG()) //TG09
                    .get_value_mPI_pPI() ) < PI*(15./180.)
           && consideredTeammate->vel.norm() >= 0.25
         )
        teammateMovingImpairmentFactor = 0.0;
    }
    teammateScore
      -= lround((5.0 / ((consideredPosition.distance(ivBallPosition)/checkVelocity)-1.0))
                *(checkVelocity-1.0)
                *(teammateMovingImpairmentFactor)
                *(1.25*consideredPosition.distance(theConsideredTeammatesPosition)) ); 
    TGospLOGPOL(logLevel,<<"OneStepPass: Resulting pass to teammate "
      <<consideredTeammate->number<<" is not going directly to that  "
      <<"teammate (deviation="<<(ivBallPosition.distance(HIS_GOAL_CENTER))
      <<"). Score decreased by "
      << lround((5.0 / ((consideredPosition.distance(ivBallPosition)/checkVelocity)-1.0))
                *(checkVelocity-1.0)
                *(teammateMovingImpairmentFactor)//TG08: zeile eingefuegt
                *(1.25*consideredPosition.distance(theConsideredTeammatesPosition) ) )
      <<" points. ==> score = "<<teammateScore
      <<" #># FINAL SCORE: "<<teammateScore); 
      
    
}

//============================================================================
// findPassTarget()
//============================================================================
bool
OneStepPass::findPassTarget( Vector &target,
                             int    &targetPlayerNumber,
                             double &resultingVelocity,
                             double &requiredKickPower,
                             ANGLE  &requiredKickDirection  )
{
  return
    findPassTargetForVirtualState
                       ( WSinfo::me->pos,
                         WSinfo::me->ang,
                         WSinfo::ball->pos,
                         WSinfo::ball->vel, 
                         target, 
                         targetPlayerNumber,
                         resultingVelocity,
                         requiredKickPower,
                         requiredKickDirection  );
}

//============================================================================
// findPassTargetForVirtualState
//============================================================================
bool 
OneStepPass::findPassTargetForVirtualState
                       ( Vector myVirtualPosition,
                         ANGLE  myVirtualBodyAngle,
                         Vector ballVirtualPosition,
                         Vector ballVirtualVelocity,
                         Vector &target,
                         int    &targetPlayerNumber,
                         double &resultingVelocity,
                         double &requiredKickPower,
                         ANGLE  &requiredKickDirection  )
{
  ivBallPosition  = ballVirtualPosition;
  ivBallVelocity  = ballVirtualVelocity;
  ivMyPosition    = myVirtualPosition;
  ivMyBodyAngle   = myVirtualBodyAngle;
  
  //create pseudo-players
  Player pseudoPlayersArray[ NUM_ATTACK_ROLES ];
  PlayerSet pseudoPlayers;
  //Problem hier: Es muss bekannt sein, welcher Spieler im Rahmen des
  //              aktuell laufenden Angriffs welche Rolle inne hat. 
  if ( ivForeseeMode == true )
  {
    int myCurrentRole, nextBallPossessingTeammateRole;
    int roleAssignmentsArray[ NUM_ATTACK_ROLES ];
    double currentCriticalOffsideLine;
    Vector nextBallPossessionPoint;
    if ( myCurrentRole != nextBallPossessingTeammateRole )
    {
      TGospLOGPOL(0,<<"OneStepPass: ERROR in determining the attack roles"
        <<" of my teammates. My role ("<<myCurrentRole<<") deviates from the"
        <<" role of the next ball possessing teammate ("
        <<nextBallPossessingTeammateRole<<") ... I AM the next ball possessing"
        <<" teammmate. Mismatch!");
    }
    else
    {
      for (int r=0; r<NUM_ATTACK_ROLES; r++)
      {
        PPlayer currentPlayer 
          = WSinfo::alive_teammates
                    .get_player_by_number(roleAssignmentsArray[r]);
        if (currentPlayer == NULL) continue;
        if (currentPlayer->number == WSinfo::me->number) continue;
        if (currentPlayer->age < 6) continue;
        Vector currentPlayerDefaultPosition
          = OpponentAwarePositioning::getAttackScoringAreaDefaultPositionForRole
                                      ( r,
                                        nextBallPossessionPoint,
                                        nextBallPossessingTeammateRole,
                                        currentCriticalOffsideLine );
        if (currentPlayer->pos.distance( currentPlayerDefaultPosition ) < 5.0)
          continue;
        PlayerSet nearbyTmms = WSinfo::valid_teammates;
        nearbyTmms.keep_players_in_circle( currentPlayerDefaultPosition, 4.0 );
        nearbyTmms.keep_players_with_max_age( 2 );
        if ( nearbyTmms.num > 0 ) continue;
        //ok, now we consider that player as a PSEUDO player
        pseudoPlayersArray[ pseudoPlayers.num ].pos = currentPlayerDefaultPosition;
        pseudoPlayersArray[ pseudoPlayers.num ].age = 1;
        pseudoPlayersArray[ pseudoPlayers.num ].number = 100+currentPlayer->number;
        //hacky misuse of direct opponent number to distinguish real 
        //player from pseudo players, below
        pseudoPlayersArray[ pseudoPlayers.num ].direct_opponent_number = 100;
        TGospLOGPOL(0,<<"OneStepPass: Consider PSEUDO PLAYER for real player "
          <<"with number "<<currentPlayer->number<<" and age "<<currentPlayer->age
          <<" at pseudo position "<<currentPlayerDefaultPosition);
        TGospLOGPOL(0,<<_2D<<C2D( currentPlayerDefaultPosition.getX()+0.1,
          currentPlayerDefaultPosition.getY()+0.1, 0.5, "006600" ));
        TGospLOGPOL(0,<<_2D<<VSTRING2D(currentPlayerDefaultPosition, currentPlayer->number, "006600"));
        pseudoPlayers.append( & pseudoPlayersArray[ pseudoPlayers.num ] );
      }
    }
  }

  //which target players to consider
  PlayerSet consideredTargetPlayers = WSinfo::valid_teammates;
  if (pseudoPlayers.num > 0)
  {
    TGospLOGPOL(0,"OneStepPass: Foresee mode: DO consider PSEUDO PLAYERS ("
      <<pseudoPlayers.num<<").");
    consideredTargetPlayers.join( pseudoPlayers );
  }
  else
  {
    TGospLOGPOL(0,"OneStepPass: Foresee mode: NO PSEUDO PLAYERS considered.");
  }

  //start calculations and evaluations of potential onestep pass targets
  int teammateScores[ consideredTargetPlayers.num ];
  targetPlayerNumber = -1;
  int bestScore = -100;
  TGospLOGPOL(0,"OneStepPass: Considered ball pos="<<ballVirtualPosition
    <<" and vel="<<ballVirtualVelocity<<", my pos="<<myVirtualPosition);
  for (int i=0; i<consideredTargetPlayers.num; i++) teammateScores[i]=0;
  
  for (int i=0; i<consideredTargetPlayers.num; i++)
  {
    PPlayer consideredTeammate = consideredTargetPlayers[i];
    TGospLOGPOL(0,<<"OneStepPass: Check one-step pass to teammate "
      <<consideredTeammate->number<<" (age="<<consideredTeammate->age
      <<",age_vel="<<consideredTeammate->age_vel<<",age_ang="
      <<consideredTeammate->age_ang<<").");
    //no passes to defenders and to goalie
    if ( consideredTeammate->number < 6 ) 
    { 
      teammateScores[i]=-100;
      TGospLOGPOL(0,<<"OneStepPass: Teammate "
        <<consideredTeammate->number<<" is no midfielder/attacker."); 
      continue; 
    }
    //exclude myself
    if ( consideredTeammate->number == WSinfo::me->number ) 
    { 
      teammateScores[i]=-100; 
      TGospLOGPOL(0,<<"OneStepPass: Teammate "
        <<consideredTeammate->number<<" is myself."); 
      continue; 
    }
    //exclude back passes in offside conquering mode
    if (    ivOffsideConqueringChance
         && consideredTeammate->pos.getX() < ivBallPosition.getX() )
    {
      teammateScores[i]=-100; 
      TGospLOGPOL(0,<<"OneStepPass: Teammate "
        <<consideredTeammate->number<<" is behind me while I am in offside "
        <<" conquering mode."); 
      continue; 
    }
    //exclude players from whom i just got a pass
    PPlayer recentPassGivingTeammate 
      = WSinfo::valid_teammates_without_me.get_player_with_most_recent_pass_info();
    if (   recentPassGivingTeammate
        && recentPassGivingTeammate->number == consideredTeammate->number
        && recentPassGivingTeammate->pass_info.age >= 0
        && WSinfo::ws->time - recentPassGivingTeammate->pass_info.abs_time < 10 
       )
    {
      PlayerSet oppsHasslingMe = WSinfo::valid_opponents;
      oppsHasslingMe.keep_players_in_circle( ivMyPosition, 2.0 );
      if (    oppsHasslingMe.num == 0
           || consideredTeammate->pos.getX() > WSinfo::his_team_pos_of_offside_line() - 2.0 )
      {
        teammateScores[i]=-100; 
        TGospLOGPOL(0,<<"OneStepPass: Teammate "
          <<consideredTeammate->number<<" has just passed to me, "
          <<" got info from him with age="<<recentPassGivingTeammate->pass_info.age<<"."); 
        continue;
      } 
    }
    //exclude passes that lead me out of the attack area
    if ( consideredTeammate->pos.getX() < ONE_STEP_PASS_ATTACK_DISTANCE_FROM_GOAL )
    { 
      teammateScores[i]=-100; 
      TGospLOGPOL(0,<<"OneStepPass: Teammate "
        <<consideredTeammate->number<<" is is out of attacking area."); 
      continue; 
    }
    //exclude players that are *obviously* in offside
    if (   consideredTeammate->pos.getX()
         > WSinfo::his_team_pos_of_offside_line() + 1.5 )
    {
      teammateScores[i]=-100; 
      TGospLOGPOL(0,<<"OneStepPass: Teammate "
        <<consideredTeammate->number<<" is oviously in offside."); 
      continue; 
    }
    //exclude players that are too much behind
    if ( ivMyPosition.getX() - consideredTeammate->pos.getX() > 16.0 )
    { 
      teammateScores[i]=-100; 
      TGospLOGPOL(0,<<"OneStepPass: Teammate "
        <<consideredTeammate->number<<" is too much behind me."); 
      continue; 
    }
    //exclude players that are too near
    if ( consideredTeammate->pos.distance( ivMyPosition ) < 3.5 )
    { 
      teammateScores[i]=-100; 
      TGospLOGPOL(0,<<"OneStepPass: Teammate "
        <<consideredTeammate->number<<" is too near to me ("
        <<(consideredTeammate->pos.distance( ivMyPosition ))<<"m)."); 
      continue; 
    }
    //exclude players that are too distant
    if ( consideredTeammate->pos.distance( ivMyPosition ) > 25.0 )
    { 
      teammateScores[i]=-100; 
      TGospLOGPOL(0,<<"OneStepPass: Teammate "
        <<consideredTeammate->number<<" is too distant from me."); 
      continue; 
    }
    //brief check whether my teammate may score
    bool consideredTeammateCanScore 
      = Tools::can_actually_score( consideredTeammate->pos );
    if ( consideredTeammateCanScore )
    {
      TGospLOGPOL(0,<<"OneStepPass: WOW! Teammate "<<consideredTeammate->number
        <<" can score a goal!");
    } 
    //exclude players that are not up-to-date
    if (   (   consideredTeammate->age > 1        //TG08: bisher
            && ivForeseeMode == false             //      age>0 && foresee==false
            && consideredTeammateCanScore == false
           )
        || (   consideredTeammate->age > 0
            && ivForeseeMode == false 
            && consideredTeammate->pos.distance(ivBallPosition) < 10.0
            && consideredTeammateCanScore == false
           )
        || (   consideredTeammate->age > 3
           )
       )
    {
      teammateScores[i]=-100; 
      TGospLOGPOL(0,<<"OneStepPass: Teammate "
        <<consideredTeammate->number<<" is too old (age="
        <<consideredTeammate->age<<")."); 
      continue;
    }
    //exclude players that are more distanct from his goal
    //when i am not attacked
    PlayerSet oppsAroundMe = WSinfo::valid_opponents;
    oppsAroundMe.keep_players_in_circle(ivMyPosition,4.0);
    if (    oppsAroundMe.num == 0    
         &&   consideredTeammate->pos.distance(HIS_GOAL_CENTER) 
            > ivMyPosition.distance(HIS_GOAL_CENTER) + 2.5
         &&   consideredTeammate->pos.getX() + 1.0
            < ivMyPosition.getX()
       )
    {
      teammateScores[i]=-100; 
      TGospLOGPOL(0,<<"OneStepPass: Teammate "
        <<consideredTeammate->number<<" stands more distant to his goal"
        <<" while I am not attacked by opponent players."); 
      continue;
    }
    //exclude players that are clearly behind me
    //when i am not attacked
    if (    oppsAroundMe.num == 0    
         && ivBallPosition.getX() >= WSinfo::his_team_pos_of_offside_line()
         &&   consideredTeammate->pos.getX() + 3.0
            < WSinfo::his_team_pos_of_offside_line()
         && fabs(ivBallPosition.getY() - consideredTeammate->pos.getY()) < 3.0
         && ivBallPosition.distance(HIS_GOAL_CENTER) > 20.0 )
    {
      teammateScores[i]=-100; 
      TGospLOGPOL(0,<<"OneStepPass: Teammate "
        <<consideredTeammate->number<<" stands so much behind me that"
        <<" a pass to him would be inacceptable."); 
      continue;
    }
    //exclude players to which no one-step pass is possible
    double checkVelocity, checkKickPower;
    ANGLE checkKickDirection;
    if ( this->isOneStepPassPossibleTo( consideredTeammate->pos, 
                                        checkVelocity,
                                        checkKickPower,
                                        checkKickDirection ) == false )
    { 
      teammateScores[i]=-100; 
      TGospLOGPOL(0,<<"OneStepPass: One-step pass to teammate "
        <<consideredTeammate->number<<" is impossible."); 
      continue; 
    }
    
    //exclude player who have a very nearby opponent, and where the passway
    //is so short and the pass velocity so high that it may happen that the
    //ball gets easily into the kickrange of them both
    PlayerSet oppsNearTeammate = WSinfo::valid_opponents;
    oppsNearTeammate.keep_players_in_circle( consideredTeammate->pos, 2.5 ); 
    oppsNearTeammate.keep_and_sort_closest_players_to_point(1, consideredTeammate->pos);
    PlayerSet oppsExtremelyNearTeammate = WSinfo::valid_opponents;
    oppsExtremelyNearTeammate.keep_players_in_circle( consideredTeammate->pos, 1.0 ); 
    oppsExtremelyNearTeammate.keep_and_sort_closest_players_to_point
                              (1, consideredTeammate->pos);
    if (  (    oppsNearTeammate.num > 0
            && oppsNearTeammate[0]->age <= 1
            && ivBallPosition.distance(consideredTeammate->pos) / checkVelocity < 6.0
          )
        ||
          (    oppsExtremelyNearTeammate.num > 0
            && oppsExtremelyNearTeammate[0]->age <= 3
            && ivBallPosition.distance(consideredTeammate->pos) / checkVelocity < 6.0
          )
       )
    {
      teammateScores[i]=-100; 
      TGospLOGPOL(0,<<"OneStepPass: Teammate "
        <<consideredTeammate->number<<" has a very close-by opponent so that"
        <<" the ball may easily get into the kick range of them both "
        <<"(opps="<<oppsNearTeammate.num<<",extrOpps?"
        <<oppsExtremelyNearTeammate.num<<"frac="
        <<(ivBallPosition.distance(consideredTeammate->pos) / checkVelocity)<<")"<<"."); 
      continue; 
    }
    
    //--end of exclusions-----------------------------------------------------

    Vector suggestedTarget = consideredTeammate->pos; 
    if (consideredTeammate->age_vel==0) 
      suggestedTarget += consideredTeammate->vel;
    this->evaluatePosition( suggestedTarget,
                            consideredTeammate,
                            checkVelocity,
                            teammateScores[i], // <- evaluation value 
                            1 //loglevel
                          );
    //improve target if possible (but not in foresee!)
    int minPlusM = -3, maxPlusM = 3; 
    if (ivOffsideConqueringChance) maxPlusM = 6;
    if (//    ivForeseeMode == false
        // &&
            suggestedTarget.distance(ivMyPosition) > 6.0 
//TG09: ZUI         && teammateScores[i] > -30 //disconsider very bad targets
       )
      for ( int plusM = minPlusM; plusM <= maxPlusM; plusM ++ )
      {
        Vector alternativeTargetPoint = consideredTeammate->pos;
        alternativeTargetPoint.addToX( (double)plusM );
        if (consideredTeammate->age_vel==0) 
          alternativeTargetPoint += consideredTeammate->vel;
        double altCheckVelocity, altCheckKickPower; ANGLE altCheckKickDirection;
        if (    alternativeTargetPoint.getX() < FIELD_BORDER_X - 2.0
             && alternativeTargetPoint.distance(ivBallPosition) > 3.0 //TG08: zeile eingefuegt
             && this->isOneStepPassPossibleTo( alternativeTargetPoint, 
                                               altCheckVelocity,
                                               altCheckKickPower,
                                               altCheckKickDirection ) == true )
        {
          int altTeammateScore = 0;
          this->evaluatePosition( alternativeTargetPoint,
                                  consideredTeammate,
                                  altCheckVelocity,
                                  altTeammateScore,
                                  4 //logLevel
                                );
          if ( altTeammateScore > teammateScores[i] )
          {
            TGospLOGPOL(1,<<"OneStepPass: I modified the target for passing"
              <<" to teammate "<<consideredTeammate->number<<" by "<<plusM
              <<" meters. **SCORE** **INCREASES** from "<<teammateScores[i]
              <<" to "<<altTeammateScore<<" points!");
            teammateScores[i] = altTeammateScore;
            suggestedTarget = alternativeTargetPoint;
            checkVelocity = altCheckVelocity;
            checkKickPower = altCheckKickPower;
            checkKickDirection = altCheckKickDirection;
          }
          else
          {
            TGospLOGPOL(1,<<"OneStepPass: I modified the target for passing"
              <<" to teammate "<<consideredTeammate->number<<" by "<<plusM
              <<" meters. SCORE DOES NOT INCREASE! Score "
              <<" is "<<altTeammateScore<<" points (best="<<teammateScores[i]<<")!");
          }
        }
      }
    //improve target if possible along its movement (but not in foresee!)
    if (    ivForeseeMode == false
         && consideredTeammate->age == 0 
         && consideredTeammate->age_vel == 0
         && consideredTeammate->age_ang == 0 
         && fabs(   (  consideredTeammate->vel.ARG()
                     - consideredTeammate->ang)
                    .get_value_mPI_pPI() )              <  PI / 6.0
         && teammateScores[i] > -30 //disconsider very bad targets
       )
    {
      TGospLOGPOL(0,<<"OneStepPass: I just saw the considered teammate "
        <<consideredTeammate->number<<", and its vel coincides with its ang.");
      for ( int plusCM = 50; plusCM <= 200; plusCM += 50 )//TG08: von 150 auf 200
      {
        double plusM = ((double)plusCM)/100.0;
        Vector alternativeTargetPoint = consideredTeammate->pos;
        Vector offsetVector( consideredTeammate->ang );
        offsetVector.normalize( plusM );
        alternativeTargetPoint += offsetVector;
        double altCheckVelocity, altCheckKickPower; ANGLE altCheckKickDirection;
        if (    alternativeTargetPoint.getX() < FIELD_BORDER_X - 2.0
             && alternativeTargetPoint.distance(ivBallPosition) > 3.0
             && this->isOneStepPassPossibleTo( alternativeTargetPoint, 
                                               altCheckVelocity,
                                               altCheckKickPower,
                                               altCheckKickDirection ) == true )
        {
          int altTeammateScore = 0;
          this->evaluatePosition( alternativeTargetPoint,
                                  consideredTeammate,
                                  altCheckVelocity,
                                  altTeammateScore,
                                  4 //logLevel
                                );
          if ( altTeammateScore > teammateScores[i] )
          {
            TGospLOGPOL(1,<<"OneStepPass: I modified the target for passing"
              <<" to teammate "<<consideredTeammate->number<<" along its"
              <<" current movement by "<<plusM
              <<" meters. **SCORE** **INCREASES** from "<<teammateScores[i]
              <<" to "<<altTeammateScore<<" points!");
            teammateScores[i] = altTeammateScore;
            suggestedTarget = alternativeTargetPoint;
            checkVelocity = altCheckVelocity;
            checkKickPower = altCheckKickPower;
            checkKickDirection = altCheckKickDirection;
          }
          else
          {
            TGospLOGPOL(1,<<"OneStepPass: I modified the target for passing"
              <<" to teammate "<<consideredTeammate->number<<" along its"
              <<" current movement by "<<plusM
              <<" meters. SCORE DOES NOT INCREASE! Score "
              <<" is "<<altTeammateScore<<" points (best="<<teammateScores[i]<<")!");
          }
        }
      }
    }
    
    //score manipulation w.r.t. scoring option
    if (consideredTeammateCanScore)
    {
      TGospLOGPOL(0,"OneStepPass: I THINK that target player "
        <<consideredTeammate->number<<" can score. => Increase score by 8 points. ");
      teammateScores[i] += 8;
    }
    //evaluate score
    TGospLOGPOL(0,<<"OneStepPass: SCORE FOR TARGET PLAYER  "
      <<consideredTeammate->number<<" IS .. ## "<<teammateScores[i]<<" ##");
    if ( teammateScores[i] > bestScore )
    {
      targetPlayerNumber = consideredTeammate->number;
      target             = suggestedTarget;
      resultingVelocity  = checkVelocity;
      requiredKickPower  = checkKickPower;
      requiredKickDirection = checkKickDirection - ivMyBodyAngle;
      bestScore          = teammateScores[i]; 
    }
  }
  if (    targetPlayerNumber != -1 
       && (    bestScore > 0 
            || (bestScore > -10 && ivForeseeMode) )
     )
    return true;
  return false;
}

//============================================================================
// foresee()
//============================================================================
bool
OneStepPass::foresee(Cmd &cmd)
{
  TGospLOGPOL(0,"OneStepPass: CALL TO FORESEE! #############################################");
  ivForeseeMode = true;
  
  //attack situation
  if (    WSinfo::me->pos.getX() < ONE_STEP_PASS_ATTACK_DISTANCE_FROM_GOAL - 2.0
       && WSinfo::his_team_pos_of_offside_line() - WSinfo::me->pos.getX() > 6.0)
  {
    TGospLOGPOL(0,<<"OneStepPass: Foresee: Behavior not usable: I am not advanced"
      <<" enough and not near enough to offside line.");
    return false;
  }
  
  Vector newmypos, newmyvel, newballpos, newballvel;
  ANGLE newmyang;

  if(WSinfo::is_ball_pos_valid() == false || WSinfo::is_ball_kickable() == true)
    return false;  // not a foresee situation
    
  //determine successor state
  if (   Tools
           ::is_ball_kickable_next_cycle(  cmd,
                                           newmypos,
                                           newmyvel,
                                           newmyang,
                                           newballpos,
                                           newballvel) 
      == false)
    return false; // ball is not kickable next cycle.
  
  //test if i may shoot a goal next cycles
  Intention dummyIntention;
#if LOGGING && BASIC_LOGGING
  long time1Score05 = Tools::get_current_ms_time(), time2Score05;
#endif
  bool iAmGoingToScore = false;
  if (ivpScore05SequenceBehavior->test_shoot2goal(dummyIntention, &cmd) == true)
  { 
    TGospLOGPOL(0, << "OneStepPass Foresee: WOW, I will SCORE next time!");
#if LOGGING && BASIC_LOGGING
    time2Score05 = Tools::get_current_ms_time();
#endif
    TGospLOGPOL(0, << "                     Score05 test required"
      <<(time2Score05-time1Score05)<<"ms.");
    iAmGoingToScore = true;
  }
#if LOGGING && BASIC_LOGGING
  time2Score05 = Tools::get_current_ms_time();
#endif
  TGospLOGPOL(0, << "                     Score05 test required"
    <<(time2Score05-time1Score05)<<"ms.");

  //one-step pass checkings
  Vector target;
  int targetPlayerNumber = -1;
  double resultingVelocity, requiredKickPower;
  ANGLE requiredKickDirection;
  bool willOneStepPassBePossible 
    = this->findPassTargetForVirtualState
                       ( newmypos,
                         newmyang,
                         newballpos,
                         newballvel, 
                         target, 
                         targetPlayerNumber,
                         resultingVelocity,
                         requiredKickPower,
                         requiredKickDirection  );
  if (   willOneStepPassBePossible 
      && targetPlayerNumber != WSinfo::me->number
      && targetPlayerNumber > 5)
  {
    if (targetPlayerNumber > 100)
    {
      targetPlayerNumber -= 100;
      TGospLOGPOL(0,<<"OneStepPass: I set attention to a pseudo player (num="
        <<targetPlayerNumber<<")");
    } 
    //set attention
    Tools::set_attention_to_request( targetPlayerNumber );
    //set communication
    Vector passVelocity( target - ivBallPosition );
    passVelocity.normalize( resultingVelocity );
    if (targetPlayerNumber > 100)
      cmd.cmd_say.set_pass(newmypos,passVelocity,WSinfo::ws->time+1);
    else
      cmd.cmd_say.set_pass(newmypos,passVelocity,WSinfo::ws->time);
    //set neck
    bool neckRequestSet = false;
    //for server v12 and above, i do not have to care about collisions
    if (   ClientOptions::server_version >= 12.0
        && WMoptions::use_server_based_collision_detection == true)
    {
    }
    else
    {
      if (   newmypos.distance(newballpos) 
           < WSinfo::me->radius + ServerOptions::ball_size+.1)
      { //ball collision danger
        TGospLOGPOL(0,<<"OneStepPass: Foresee: Ball might collide with me "
          <<"next cycle!");
        Angle toball = (newballpos-newmypos).arg();
        if ( Tools::could_see_in_direction(toball) )
        {
          TGospLOGPOL(0, << "OneStepPass: Foresee: Ball might collide with me "
            <<"next cycle, therefore, I look towards it with angle="<<toball);
          Tools::set_neck_request(NECK_REQ_LOOKINDIRECTION, toball, true); // forced
          neckRequestSet = true;
        }
      }
      else 
      {
        TGospLOGPOL(0,<<"OneStepPass: Foresee: Ball WON'T collide with me "
          <<"next cycle (dist="<<newmypos.distance(newballpos)<<")!");
      }
    }
    if (iAmGoingToScore == false) //note: score behavior sets no attention
    {
      //check for looking to the goalie  //TG08
      int goalieAgeThreshold = 1;
      if (    WSinfo::his_goalie 
           && WSinfo::his_goalie->pos.distance(WSinfo::me->pos) < 8.0 ) 
        goalieAgeThreshold = 0;
      PlayerSet oppsOnTheWayToHisGoal = WSinfo::valid_opponents;
      oppsOnTheWayToHisGoal.remove(WSinfo::his_goalie);
      oppsOnTheWayToHisGoal.keep_players_in_quadrangle
        ( WSinfo::me->pos, HIS_GOAL_CENTER, 4.0, 14.0 );
      if (   WSinfo::me->pos.distance(HIS_GOAL_CENTER) < 22.0
          && WSinfo::his_goalie
          && WSinfo::his_goalie->pos.distance(WSinfo::me->pos) < 12.0
          && WSinfo::his_goalie->age > goalieAgeThreshold
          && oppsOnTheWayToHisGoal.num == 0
         )
      {
        ANGLE toGoalie = (WSinfo::his_goalie->pos - WSinfo::me->pos).ARG();
        if (Tools::could_see_in_direction( toGoalie ) )
        {
          TGospLOGPOL(0,<<"OneStepPass: Set neck to goalie: look to dir "
            <<RAD2DEG(toGoalie.get_value_mPI_pPI()));
          Tools::set_neck_request(NECK_REQ_LOOKINDIRECTION, toGoalie, true);
          neckRequestSet = true;
        }
      }
      //standard looking
      if ( neckRequestSet == false )
      {    
        ANGLE ball2targetdirC = (target - WSinfo::ball->pos).ARG(), //dir.of target
              ball2targetdirL = ball2targetdirC + ANGLE( PI*(15.0/180.0) ),
              ball2targetdirR = ball2targetdirC - ANGLE( PI*(15.0/180.0) ),
              ball2targetdir  = ball2targetdirC;
        Vector rotatedTargetLeft = target - WSinfo::ball->pos;
        rotatedTargetLeft.rotate(PI*(15.0/180.0));
        rotatedTargetLeft += WSinfo::ball->pos;
        Vector rotatedTargetRight = target - WSinfo::ball->pos;
        rotatedTargetRight.rotate(PI*(-15.0/180.0));
        rotatedTargetRight += WSinfo::ball->pos;
        PlayerSet opponentsInPassway = WSinfo::valid_opponents;
        opponentsInPassway.keep_players_in_quadrangle
                           ( WSinfo::ball->pos,
                             rotatedTargetLeft,
                             6.0,
                             0.8*rotatedTargetLeft.distance(WSinfo::ball->pos));
        int leftCheckOpps = opponentsInPassway.num;
        if (Tools::could_see_in_direction(ball2targetdirL) == false)
          leftCheckOpps = -100;
        opponentsInPassway = WSinfo::valid_opponents;
        opponentsInPassway.keep_players_in_quadrangle
                           ( WSinfo::ball->pos,
                             rotatedTargetRight,
                             6.0,
                             0.8*rotatedTargetLeft.distance(WSinfo::ball->pos));
        int rightCheckOpps = opponentsInPassway.num,
            centerCheckOpps = 0;
        if (Tools::could_see_in_direction(ball2targetdirR) == false)
          rightCheckOpps = -100;
        if (Tools::could_see_in_direction(ball2targetdirC) == false)
          centerCheckOpps = -100;
              
        if (leftCheckOpps > rightCheckOpps && leftCheckOpps > centerCheckOpps) 
          ball2targetdir = ball2targetdirL;
        else
        if (leftCheckOpps < rightCheckOpps && rightCheckOpps > centerCheckOpps)  
          ball2targetdir = ball2targetdirR;
        
        if (   WSmemory::last_seen_in_dir(ball2targetdir) >=0
            && Tools::could_see_in_direction(ball2targetdir) )
        { 
          TGospLOGPOL(0,<<"OneStepPass: Intention is to play OneStepPass"
            <<"-> look to receiver ("<<targetPlayerNumber
            <<"), direction: "<<RAD2DEG(ball2targetdir.get_value()));
          Tools::set_neck_request( NECK_REQ_LOOKINDIRECTION, 
                                   ball2targetdir.get_value(), 
                                   true); // forced
          neckRequestSet = true;
        }
        else
        {
          TGospLOGPOL(0,<<"OneStepPass: No neck request set, information fresh "
            <<"or cannot see into direction "<<RAD2DEG(ball2targetdir.get_value()));
        }
      }
    }
    //finish
    ivLastSuccessfulForesee = WSinfo::ws->time;
    TGospLOGPOL(0,<<"OneStepPass: FORESEE returns true! #############################################");
    return true;
  }
  TGospLOGPOL(0,<<"OneStepPass: FORESEE returns false! #############################################");
  return false;
}

//============================================================================
// get_cmd()
//============================================================================
bool
OneStepPass::get_cmd( Cmd & cmd )
{
  ivForeseeMode = false;
  
  Vector target;
  int    targetPlayerNumber = -1;
  double  resultingVelocity = 0.0,
         requiredKickPower = 0.0;
  ANGLE  requiredKickDirection;
  
  if ( this->findPassTarget( target, 
                             targetPlayerNumber,
                             resultingVelocity,
                             requiredKickPower,
                             requiredKickDirection ) == false )
  {
    TGospLOGPOL(0,<<"OneStepPass: Did not find a suitable pass target. GIVING UP!");
    return false;
  }

  TGospLOGPOL(0,<<"OneStepPass: Found a pass target: "<<target<<", pl="
    <<targetPlayerNumber<<" [OCM="<<ivOffsideConqueringChance<<"]");
  TGospLOGPOL(0,<<_2D<<VL2D(WSinfo::ball->pos,
                           target,"8b008b"));
  
  TGospLOGPOL(0,<<"OneStepPass: Pass to that target is possible. It requires"
    <<" a kick with power="<<requiredKickPower<<" and kick direction="
    <<RAD2DEG(requiredKickDirection.get_value_mPI_pPI())
    <<". The resulting velocity is "<<resultingVelocity);
  
  TGospLOGPOL(0,<<"OneStepPass: Pass to that target is also safe. Play it.");
  cmd.cmd_body.set_kick( requiredKickPower, 
                         requiredKickDirection.get_value_mPI_pPI() );
  //also communicate the pass
  Vector passVelocity(target - ivBallPosition);
  passVelocity.normalize(resultingVelocity);
  cmd.cmd_say.set_pass(WSinfo::ball->pos,passVelocity,WSinfo::ws->time);
  return true;
}

//============================================================================
// isSuitableOneStepPassSituation()
//============================================================================
bool
OneStepPass::isSuitableOneStepPassSituation()
{
  //TG16: Do not use OneStepPass against HERMES //TG17: no more team differentiation
   /*if ( WSinfo::get_current_opponent_identifier() == TEAM_IDENTIFIER_HERMES )
   {
     TGospLOGPOL(0,<<"OneStepPass: Behavior not usable for HERMES.");
     return false;
   }*/

  bool returnValue = true;
  //only if i just got the ball
  PlayerSet oppsAroundMe = WSinfo::valid_opponents;
  oppsAroundMe.keep_players_in_circle(WSinfo::me->pos,4.0);
  bool ivOffsideConqueringChance = false;
  if (      ivpMemoryRecentBallKickableTimePoints[ ivMemoryCounter ]
         == WSinfo::ws->time - 1 
     )
  {
    ivOffsideConqueringChance
      =    oppsAroundMe.num > 1
        && WSinfo::me->pos.getX() < FIELD_BORDER_X - 5.0
        && WSinfo::his_team_pos_of_offside_line() - WSinfo::me->pos.getX() < 5.0;
    if (ivOffsideConqueringChance == true)
    {
      TGospLOGPOL(0,<<"OneStepPass: Behavior may be usable: I have had the ball"
        <<" for more than one time step, but I am in OFFSIDE CONQUERING mode.");
    }
    else
    {
      //TG08: new special case: allow this behavior to be usable (although
      //the ball has been in the kickrange for more than one step) if the
      //player is being attacked by an opponent
      if (   oppsAroundMe.num > 1
          || (oppsAroundMe.num == 1 && oppsAroundMe[0]->age <= 1) )
      {
        TGospLOGPOL(0,<<"OneStepPass: Behavior may be usable: I have had the ball"
          <<" for more than one time step, but I am in BEING HASSLED mode.");
      }
      else
      {
        returnValue = false;
        TGospLOGPOL(0,<<"OneStepPass: Behavior not usable: I have had the ball"
          <<" for more than one time step and I am not attacked. "
          <<"Wball06 should take care.");
      }
    }
  }
  else//just debug output
  {
    TGospLOGPOL(0,<<"OneStepPass: Behavior may be usable: I have JUST GOT THE"
      <<" BALL.");
  }
  //set instance variables
  ivMemoryCounter = (ivMemoryCounter + 1) % cvsMEMORYSIZE;
  ivpMemoryRecentBallKickableTimePoints[ ivMemoryCounter ] = WSinfo::ws->time;
  
  //leave this behavior
  if ( returnValue == false ) return false;
  
  //attack situation
  if (    WSinfo::me->pos.getX() < ONE_STEP_PASS_ATTACK_DISTANCE_FROM_GOAL
       &&   WSinfo::his_team_pos_of_offside_line() - WSinfo::me->pos.getX()
          > 11.0 //TG08: threshold increased from 6 to 11
     )
  {
    TGospLOGPOL(0,<<"OneStepPass: Behavior not usable: I am not advanced"
      <<" enough and not near enough to offside line.");
    return false;
  }
  //check if score behavior is correctly set
  if ( ivpScore05SequenceBehavior == NULL )
  {
    TGospLOGPOL(0,<<"OneStepPass: Behavior not usable: The score behavior"
      <<" has not been set!");
    return false;
  }
  //check if i can score
  Intention dummyIntention;
  if ( ivpScore05SequenceBehavior->test_shoot2goal( dummyIntention ) )
  {
    TGospLOGPOL(0,<<"OneStepPass: Behavior not usable: I actually"
      <<" can score using Score05. Wball06 will do that (ballVel="<<WSinfo::ball->vel<<".");
    return false;
  }
  if ( ivpOneStepScoreBehavior->isSuitableOneStepScoreSituation() )
  {
    TGospLOGPOL(0,<<"OneStepPass: Behavior not usable: I actually"
      <<" can score using OneStepScore. Wball06 will do that (ballVel="<<WSinfo::ball->vel<<".");
    return false;
  }

  //i should better go myself
  PlayerSet oppsBeforeMe = WSinfo::valid_opponents;
  Vector beforeMe = WSinfo::me->pos; beforeMe.addToX( 12.0 );
  oppsBeforeMe.keep_players_in_quadrangle
               (WSinfo::me->pos,beforeMe,8.0,16.0);
  PlayerSet oppsVeryCloseToMe = WSinfo::valid_opponents;
  oppsVeryCloseToMe.keep_players_in_circle(WSinfo::me->pos,2.5);
  if (   WSinfo::his_team_pos_of_offside_line() - WSinfo::me->pos.getX() < 6.0//TG09: increased from 3.0
      && WSinfo::me->pos.getX() < FIELD_BORDER_X - 8.5 //TG08: 6->8.5
      && (   oppsAroundMe.num == 0
          || (   oppsVeryCloseToMe.num == 0 
              && WSinfo::his_team_pos_of_offside_line() - WSinfo::me->pos.getX() < 2.2 )//TG09
         )
      && (    oppsBeforeMe.num == 0 
           || (   oppsBeforeMe[0] == WSinfo::his_goalie
               && WSinfo::his_goalie 
               && WSinfo::his_goalie->pos.distance(WSinfo::me->pos) > 4.0 )
         )
     )
  {
    TGospLOGPOL(0,<<"OneStepPass: Behavior not usable: I have much free"
      <<" room in front of me, I'd better go myself.");
    return false;
  }
  //i am alone in front of his goal
  oppsAroundMe = WSinfo::valid_opponents;
  oppsAroundMe.keep_and_sort_closest_players_to_point( 1, WSinfo::me->pos );
  PlayerSet oppsOnMyWayToHisGoal = WSinfo::valid_opponents;
  oppsOnMyWayToHisGoal.keep_players_in_quadrangle
                       (WSinfo::me->pos,HIS_GOAL_CENTER,3.0,14.0);
  oppsOnMyWayToHisGoal.remove( WSinfo::his_goalie );
  oppsOnMyWayToHisGoal.keep_and_sort_closest_players_to_point
    ( oppsOnMyWayToHisGoal.num, WSinfo::me->pos );
  /*if (   oppsOnMyWayToHisGoal.num == 0
      && (    oppsAroundMe.num == 0
           || oppsAroundMe[0]->pos.distance(WSinfo::me->pos) > 1.0 )
      && fabs(WSinfo::me->pos.y) < 10.0
      && WSinfo::me->pos.x >   WSinfo::his_team_pos_of_offside_line() 
                             - (1.5*WSinfo::me->kick_radius)
     )*/
  //TG09: replaced!
  PlayerSet possibleScoringTeammates = WSinfo::valid_teammates_without_me;
  for (int i=0; i<possibleScoringTeammates.num; i++)
    if ( Tools::can_score( possibleScoringTeammates[i]->pos )  )
    {
      break;
    }
  if (   fabs(WSinfo::me->pos.getY()) < 15.0
      && (    oppsAroundMe.num == 0
           || oppsAroundMe[0]->pos.distance(WSinfo::me->pos) > 4.0 )
      && (    oppsOnMyWayToHisGoal.num == 0
           || (     oppsOnMyWayToHisGoal[0]->pos.distance(WSinfo::me->pos)
                  > 0.5 * WSinfo::me->pos.distance(HIS_GOAL_CENTER)
               && oppsOnMyWayToHisGoal[0]->pos.distance(WSinfo::me->pos) > 6.0 )
         )
     )
  {
    TGospLOGPOL(0,<<"OneStepPass: Behavior not usable: I am alone in front"
      <<" of his goal, I'd better go myself.");
    return false;
  }
  //test for "in trouble" 
  PlayerSet troublingOpps = WSinfo::valid_opponents;
  troublingOpps.keep_players_in_circle
                (WSinfo::ball->pos, 2.0*ServerOptions::kickable_area); 
  troublingOpps.keep_and_sort_closest_players_to_point(1,WSinfo::ball->pos);
  if (    troublingOpps.num > 0 
       &&   WSinfo::ball->pos.distance( troublingOpps[0]->pos ) 
          < 0.98 * troublingOpps[0]->kick_radius ) 
  {
    TGospLOGPOL(0,<<"OneStepPass: Behavior not usable: I am in trouble,"
      <<" an opponent controls the ball, too. Wball06 handles that!");
    return false;
  }
  //test for "two teammates controls ball"
  PPlayer otherBallControllingTeammate 
    = WSinfo::valid_teammates_without_me
              .closest_player_to_point(WSinfo::ball->pos);
  if (    otherBallControllingTeammate != NULL
       &&   WSinfo::ball->pos.distance( otherBallControllingTeammate->pos ) 
          < 0.98 * otherBallControllingTeammate->kick_radius ) 
  {
    TGospLOGPOL(0,<<"OneStepPass: Behavior not usable: Two teammates contol"
      <<" the ball. Wball06 handles that!");
    return false;
  }              
  
  return returnValue;
}

//============================================================================
// isSuitableOneStepPassSituationForExtremePanic()
//============================================================================
bool
OneStepPass::isSuitableOneStepPassSituationForExtremePanic()
{
  PlayerSet hasslingOpps = WSinfo::valid_opponents;
  hasslingOpps.keep_players_in_circle
                (WSinfo::me->pos, ServerOptions::visible_distance);
  hasslingOpps.keep_and_sort_closest_players_to_point
                (2,WSinfo::me->pos);
  PlayerSet helpingTmms = WSinfo::valid_teammates_without_me;
  helpingTmms.keep_players_with_max_age( 1 );
  if (    hasslingOpps.num > 1 
       && helpingTmms.num > 0 )
  {
    TGospLOGPOL(0,<<"OneStepPass: I am heavily attacked, but found "
      <<helpingTmms.num<<" potentially helping teammates.");
    for (int i=0; i<helpingTmms.num; i++)
    {
      TGospLOGPOL(0,<<"            teammate "<<helpingTmms[i]->number);
    }
  }
  return false;
}

//============================================================================
// isOneStepPassPossibleTo()
//============================================================================
bool
OneStepPass::isOneStepPassPossibleTo( Vector target, 
                                      double &resultingVelocity,
                                      double &requiredKickPower,
                                      ANGLE  &requiredKickDirection )
{
  //Geometrie ...
  int   anzahlTestZielGeschwindigkeiten = 9; //TG08: 3.0 und 2.85 hinzugefuegt
  double testZielGeschwindigkeiten[] = {3.0,2.85,2.7,2.5,2.3,2.1,1.9,1.7,1.5};
  for (int i=0; i<anzahlTestZielGeschwindigkeiten; i++)
  {
    double zielGeschwindigkeit = testZielGeschwindigkeiten[i];
    Vector wunschBallGeschwindigkeit = target - ivBallPosition;
    wunschBallGeschwindigkeit.normalize( zielGeschwindigkeit ); 
    Vector notwendigerSchussvektor 
      = wunschBallGeschwindigkeit - ivBallVelocity;

    if (notwendigerSchussvektor.norm() > ServerOptions::ball_speed_max)
    {
      TGospLOGPOL(0,<<"OneStepPass: Zielgeschwindigkeit "<<zielGeschwindigkeit
        <<" nicht erreichbar.");
      continue;
    }
    double stepsToTarget = (   (target-ivMyPosition)
                            - ivBallVelocity).norm() / zielGeschwindigkeit;
    if (stepsToTarget < 2.0 && zielGeschwindigkeit > 2.0)
    {
      TGospLOGPOL(0,<<"OneStepPass: Zielgeschwindigkeit "<<zielGeschwindigkeit
        <<" waere erreichbar, ist aber zu hoch fuer einen Weg von "
        <<(target - ivMyPosition - ivBallVelocity).norm()<<"m.");
      continue;
    }
   
   
    ANGLE ballAngle 
      = (ivBallPosition-ivMyPosition).ARG() - ivMyBodyAngle;
    double ballDistNetto
      =   (ivBallPosition-ivMyPosition).norm() 
        - WSinfo::me->radius 
        - ServerOptions::ball_size;         
    double maxSchussstaerkeBeiRuhemdemBall
      =   ServerOptions::ball_speed_max 
        * (  1.00 
           - 0.25 * fabs(ballAngle.get_value_mPI_pPI()) / PI 
           - 0.25 * ballDistNetto 
             / (  WSinfo::me->kick_radius - WSinfo::me->radius 
                - ServerOptions::ball_size));

    if (notwendigerSchussvektor.norm() > maxSchussstaerkeBeiRuhemdemBall)
    {
      TGospLOGPOL(0,<<"OneStepPass: Aus aktuelle Balllage kann die "
        <<"Zielgeschwindigkeit von "<<zielGeschwindigkeit
        <<" nicht erreicht werden. Ein Schussvektor von "
        <<notwendigerSchussvektor<<" ("<<notwendigerSchussvektor.norm()
        <<") waere noetig.");
      continue;
    }
        
    TGospLOGPOL(0,<<"OneStepPass: Aus aktuelle Balllage kann die "
      <<"Zielgeschwindigkeit von "<<zielGeschwindigkeit
      <<" erreicht werden. Ein Schussvektor von "
      <<notwendigerSchussvektor<<" ("<<notwendigerSchussvektor.norm()
      <<") ist noetig.");
    
    resultingVelocity = zielGeschwindigkeit;
    requiredKickPower = 100.0 * (   notwendigerSchussvektor.norm()
                                  / maxSchussstaerkeBeiRuhemdemBall );
    requiredKickDirection
      .set_value( notwendigerSchussvektor.ARG().get_value_0_p2PI() );
      
    return true;          
  }
  
  return false;
}

//============================================================================
// setScoreBehaviors()
//============================================================================
void
OneStepPass::setScoreBehaviors( BodyBehavior* scoreBehavior, BodyBehavior* oneStepScore )
{
  ivpScore05SequenceBehavior 
    = (Score05_Sequence*) scoreBehavior;
  ivpOneStepScoreBehavior
    = (OneStepScore*) oneStepScore;
}


