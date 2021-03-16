#include "mod_learnPositioningLearner.h"

#include "coach.h"
#include "defs.h"
#include "logger.h"
#include "valueparser.h"

#define EPISODE_USABLE_FOR_LEARNING_FAILURE      0
#define EPISODE_USABLE_FOR_LEARNING_SUCCESS      1
#define EPISODE_IGNORABLE_FOR_LEARNING          -1

const char ModLearnPositioningLearner::modName[]="ModLearnPositioningLearner";


//============================================================================
// Konstruktor
//============================================================================
/**
 * Konstruktor
 */
ModLearnPositioningLearner::ModLearnPositioningLearner()
{
  ivSuccessEpisodesCounter  = 0;
  ivFailureEpisodesCounter  = 0;
  ivIgnoreEpisodesCounter   = 0;
  ivContinuingEpisodeUsageShare = 0.0;
}

//============================================================================
// init()
//============================================================================
/**
 * Initialisierung
 */
bool 
ModLearnPositioningLearner::init(int argc,char **argv)
{
  LOG_FLD(0,"\nInitialising Module ModLearnPositioningLearner ...");
  //in-/decrement these parameters so that they can be handled by VP
  argc--;
  argv++;
  //read config file
  ValueParser vp1(Options::coach_conf, "ModLearnPositioningLearner");
  
  vp1.get("collectedEpisodeFile", ivEpisodeFileName, cvMaxStringLength);
  vp1.get("requiredSuccessEpisodesForLearning", 
           ivRequiredSuccessEpisodesForLearning);
  vp1.get("requiredTotalEpisodesForLearning", 
           ivRequiredTotalEpisodesForLearning);  
  vp1.get("continuingEpisodeUsageShare", ivContinuingEpisodeUsageShare);
  
  vp1.get("basicRewardFactor", ivBasicRewardFactor);
  vp1.get("finalRewardSuccess", ivFinalRewardSuccess);
  vp1.get("finalRewardFailure", ivFinalRewardFailure);
  vp1.get("weight_averageGonePlayerWaysInMeters", 
           ivRewardWeightAverageGonePlayerWaysInMeters);
  vp1.get("weight_averageDistanceToDirectOpponents",
           ivRewardWeightAverageDistanceToDirectOpponents);
  vp1.get("weight_penaltyIfBallInKickrange", 
           ivRewardWeightPenaltyIfBallInKickrange);
  vp1.get("weight_averageDistanceToBall",
           ivRewardWeightAverageDistanceToBall);
           
  vp1.get("successIffGoal", ivSuccessIffGoal);
  vp1.get("successIffGoalApproached", ivSuccessIffGoalApproached);
  vp1.get("goalApproachStopThreshold", ivGoalApproachStopThreshold);
  
  return true;
}

//============================================================================
// destroy()
//============================================================================
/**
 * Finalisierung
 */
bool 
ModLearnPositioningLearner::destroy()
{
  if (ivpEpisodeFileHandle)
    fclose( ivpEpisodeFileHandle );
  return true;
}

//============================================================================
// allPlayersAliveCheck
//============================================================================
/**
 * Diese Methode ist radikal. Sobald sie erkennt, dass ein Spieler - egal ob
 * ein Mit- oder Gegenspieler - abgest?rzt ist, beendet sie den Coach 
 * mit Programm-Rueckgabewert -1.
 */
void
ModLearnPositioningLearner::allPlayersAliveCheck()
{
  for (int i=0;i<TEAM_SIZE;i++) 
    if (!fld.myTeam[i].alive) 
      exit(10);
  for (int i=0;i<TEAM_SIZE;i++) 
    if (!fld.hisTeam[i].alive) 
      exit(10);
}

//============================================================================
// behave()
//============================================================================
/**
 * Hauptmethode, einmal pro Zyklus aufgerufen
 */
bool 
ModLearnPositioningLearner::behave()
{
  this->allPlayersAliveCheck();
  
  LOG_FLD(1,"ModLearnPositioningLearner: Entered behave at t="<<fld.getTime());
  if ( fld.getPM() != PM_play_on )
  {
    LOG_FLD(1,"ModLearnPositioningLearner: Play mode is NOT PM_play_on ("
      <<fld.getPM()<<"), write out episode.");
    if (ivEpisode.getLength() > 10)
    {
      ivEpisode.addNextState(); //final state
      bool usableForLearning = this->classifyCompletedEpisode();
      cout<<"EPISODE'S USABILITY HAS BEEN CLASSIFIED AS "<<usableForLearning<<endl;
      double finalReward = this->calculateSuggestedFinalReward();
      bool success = (finalReward < ivFinalRewardFailure);
      ivEpisode.addSuggestedImmediateReward( finalReward );
      if (usableForLearning == false)
        this->writeOutEpisode( EPISODE_IGNORABLE_FOR_LEARNING );
      else
      {
        if (success)
          this->writeOutEpisode( EPISODE_USABLE_FOR_LEARNING_SUCCESS );
        else
          this->writeOutEpisode( EPISODE_USABLE_FOR_LEARNING_FAILURE );
      }
    }
    ivEpisode.reset();
    this->checkForFinishingTraining();
  }
  else
  {
    LOG_FLD(1,"ModLearnPositioningLearner: Play mode is PM_play_on, "
      <<"extend episode (curLength="<<ivEpisode.getLength()<<").");
    if (ivLastPlaymode != PM_play_on) ivSequenceStartTime = fld.getTime();
    if (   ivLastPlaymode == PM_play_on
        && fld.getTime() - ivSequenceStartTime > 1)
    {
      ivEpisode.addNextState();
      ivEpisode.addSuggestedImmediateReward
                ( this->calculateSuggestedImmediateReward() );
    }
  }
  
  ivLastPlaymode = fld.getPM();
  return true;
}

//============================================================================
// calculateSuggestedImmediateReward()
//============================================================================
/**
 * Methode, die einen Vorschlag fuer den aktuellen Immediate Reward
 * erarbeitet.
 * Diese Methode ist deshalb vorgesehen, um schon fruehzeitig die 
 * Moeglichkeit eines effizienzsteigernden Reward-Shapings umzusetzen.
 * 
 * Betont werden muss, dass alle Rewards als Kosten aufgefasst und
 * interpretiert werden!
 */
double
ModLearnPositioningLearner::calculateSuggestedImmediateReward()
{
  if (ivEpisode.getLength() < 2)
    return ivBasicRewardFactor;
  
  //calculate average of ways gone by my players last cycle
  double way
    = ivEpisode.getAverageWayGoneByTeammates( ivEpisode.getLength() - 1 );
  double rewardComponentWaysGone
    = way * ivRewardWeightAverageGonePlayerWaysInMeters;
    
  //calculate average distance of team players to their direct opponents
  double distanceToDONormalization = 0.2; //5m distance is "normal"
  double distToDO
    = ivEpisode.getAverageDistanceToDirectOpponents
                ( ivEpisode.getLength() - 1 );
  double rewardComponentDistanceToDirectOpponents
    =   distanceToDONormalization
      * distToDO * ivRewardWeightAverageDistanceToDirectOpponents;
    
  //calculate penalty for not playing passes
  bool ballPossession = ivEpisode.ballKickableForTeammate
                                  ( ivEpisode.getLength()-1 );
  double ballVelocity = ivEpisode.ballVelocityAtTime(ivEpisode.getLength()-1);
  double rewardComponentPenaltyForNotPassing
    = ( ballPossession || ballVelocity < 1.0
       ? 1 : 0) * ivRewardWeightPenaltyIfBallInKickrange;
    
  //calculate average distance of team players to the ball
  double distanceToBallNormalization = 1.0/20.0; //20m distance is "normal"
  double distToBall
    = ivEpisode.getAverageDistanceToBall( ivEpisode.getLength() - 1 );
  double rewardComponentDistanceToBall
    =   distanceToBallNormalization
      * distToBall * ivRewardWeightAverageDistanceToBall;
    
  //determine suggestion
  return
    ivBasicRewardFactor * (
                              rewardComponentWaysGone
                            + rewardComponentDistanceToDirectOpponents
                            + rewardComponentPenaltyForNotPassing
                            + rewardComponentDistanceToBall
                          );
}

//============================================================================
// calculateSuggestedFinalReward
//============================================================================
/**
 * Methode, die einen Vorschlag fuer den abschliessenden Reward einer
 * Episode erarbeitet.
 * 
 * Folgende Kriterien sind beruecksichtigt:
 *  - Torefolg == SUCCESS
 *  - dem gegnerischen Tor naeher als ivGoalApproachStopThreshold 
 *    Meter gekommen == SUCCESS
 * 
 */
double
ModLearnPositioningLearner::calculateSuggestedFinalReward()
{
  if (ivSuccessIffGoal)
  {
    for (int i=0; i<ivEpisode.getLength(); i++)
      if (    fabs(ivEpisode.getBallPosition(i).getY()) < 7.02
           && ivEpisode.getBallPosition(i).getX() > FIELD_BORDER_X )
      {
        LOG_FLD(2, "ModLearnPositioningLearner: Final reward calculation: "
          <<"GOAL at t="<<i<<" => SUCCESS");
        return ivFinalRewardSuccess;
      }
  }
  if (ivSuccessIffGoalApproached)
  {
    for (int i=0; i<ivEpisode.getLength(); i++)
      if (     ivEpisode.getBallPosition(i).distance( HIS_GOAL_CENTER )
             < ivGoalApproachStopThreshold
          && ivEpisode.ballKickableForTeammate(i) )
      {
        LOG_FLD(2, "ModLearnPositioningLearner: Final reward calculation: "
          <<"At t="<<fld.getTime()<<" ball has approaches the opponent goal"
          <<" and was kickable for a teammate.");
        int tmnhg;
        if ( (tmnhg = ivEpisode.getNumberOfTeammatePlayersAroundPoint
                               ( i,
                                 HIS_GOAL_CENTER,
                                 ivGoalApproachStopThreshold * 2.0 )) >= 1  )
        {
          LOG_FLD(2, "ModLearnPositioningLearner: Final reward calculation: "
            <<"At t="<<fld.getTime()<<" there are "<<tmnhg<<" teammates"
            <<" nearer than "<<ivGoalApproachStopThreshold * 2.0
            <<"m to his goal center. ==> That's good!");
          return ivFinalRewardSuccess;
        }
        else
        {
          LOG_FLD(2, "ModLearnPositioningLearner: Final reward calculation: "
            <<"At t="<<fld.getTime()<<" there are "<<tmnhg<<" teammates"
            <<" nearer than "<<ivGoalApproachStopThreshold * 1.5
            <<"m to his goal center. ==> That's not enough (FAILURE)!");
        }
      }
      else
      {
        LOG_FLD(2, "ModLearnPositioningLearner: Final reward calculation: "
          <<"At t="<<fld.getTime()<<" ball is "
          <<ivEpisode.getBallPosition(i).distance( HIS_GOAL_CENTER )
          <<"m distant from goal and kickable="<<ivEpisode.ballKickableForTeammate(i)<<".");
      }
  }
  LOG_FLD(2, "ModLearnPositioningLearner: Final reward calculation: No "
    <<"success criterion is fulfilled.");
  return ivFinalRewardFailure;
}

//============================================================================
// checkForFinishingTraining
//============================================================================
/**
 * Diese Methode ueberprueft, ob genuegend Episoden eingesammelt worden sind,
 * um auf dieser Basis einen Lernvorgang zu starten.
 * 
 */
void
ModLearnPositioningLearner::checkForFinishingTraining()
{
  if (
          ivSuccessEpisodesCounter >= ivRequiredSuccessEpisodesForLearning
       &&    (ivSuccessEpisodesCounter + ivFailureEpisodesCounter) 
          >= ivRequiredTotalEpisodesForLearning
     )
  {
    //finish collecting episodes & start learning
    LOG_FLD(0,"ModLearnPositioningLearner: I WILL FINALIZE MYSELF (#episodes: "
      <<ivSuccessEpisodesCounter<<" successfull, "<<(ivSuccessEpisodesCounter
      + ivFailureEpisodesCounter)<<" total)"<<std::flush);
    exit(0);
  }
}

//============================================================================
// classifyCompletedEpisode
//============================================================================
/**
 * Methode, die die Entscheidung trifft, ob diese Episode fuer das Lernen
 * verwendet werden sollte.
 * 
 */
bool
ModLearnPositioningLearner::classifyCompletedEpisode()
{
  LOG_FLD(1,"ModLearnPositioningLearner::classifyCompletedEpisode: ENTER");
  
  //ueberpruefung hinsichtlich play-modi -> ball ins aus oder abseitsstellung
  //implizieren bspw. nicht verwendbare episoden
  if ( fld.getPM() != PM_play_on )
  {
    LOG_FLD(0,"INFO: SPIELMODUS am Episodenende = "<<fld.getPM());
  }
  
  //vorausberechnung der praedikate
  bool ballAtEndOfEpisodeNotExclusivelyKickableForTeammate
    =    (ivEpisode.ballKickableForTeammate( ivEpisode.getLength()-1 )) == false
      || (ivEpisode.ballKickableForOpponent( ivEpisode.getLength()-1 )) == true;
  LOG_FLD(1,"ModLearnPositioningLearner::classifyCompletedEpisode: "
    <<"ballAtEndOfEpisodeNotExclusivelyKickableForTeammate="
    <<ballAtEndOfEpisodeNotExclusivelyKickableForTeammate);
  double distanceFromBallToTeammateThatWasLastAtBall = 0.0,
        distanceFromBallToTeammateThatWasLastAtBallAtEndOfEpisode = 0.0;
  int lastAtBallTime = -1;
  for (int i=ivEpisode.getLength()-1; i>=0; i--)
    if (ivEpisode.ballKickableForTeammate(i))
    {
      lastAtBallTime = i;
      break;
    }
  if (lastAtBallTime >= 0)
  {
    Vector ballPos = ivEpisode.getBallPosition( ivEpisode.getLength()-1 );
    distanceFromBallToTeammateThatWasLastAtBall
      = ivEpisode.getDistanceOfNearestPlayerToPos( lastAtBallTime, 
                                                   ballPos );
    distanceFromBallToTeammateThatWasLastAtBallAtEndOfEpisode
      = ivEpisode.getDistanceOfNearestPlayerToPos( ivEpisode.getLength()-1,
                                                   ballPos );
  }
  LOG_FLD(1,"ModLearnPositioningLearner::classifyCompletedEpisode: "
    <<"distanceFromBallToTeammateThatWasLastAtBall="
    <<distanceFromBallToTeammateThatWasLastAtBall);
  LOG_FLD(1,"ModLearnPositioningLearner::classifyCompletedEpisode: "
    <<"distanceFromBallToTeammateThatWasLastAtBallAtEndOfEpisode="
    <<distanceFromBallToTeammateThatWasLastAtBallAtEndOfEpisode);
  int numberOfOpponentsNearPassPlayingTeammate;
  if (lastAtBallTime >= 0)
  {
    Vector ballPos = ivEpisode.getBallPosition( lastAtBallTime );
    numberOfOpponentsNearPassPlayingTeammate
      = ivEpisode.getNumberOfOpponentPlayersAroundPoint( lastAtBallTime,
                                                         ballPos,
                                                         5.0 );
  }
  LOG_FLD(1,"ModLearnPositioningLearner::classifyCompletedEpisode: "
    <<"numberOfOpponentsNearPassPlayingTeammate="
    <<numberOfOpponentsNearPassPlayingTeammate);
  
  if (    ballAtEndOfEpisodeNotExclusivelyKickableForTeammate
          //for passes
       && distanceFromBallToTeammateThatWasLastAtBall > 3.0
          //for self passes
       && distanceFromBallToTeammateThatWasLastAtBallAtEndOfEpisode > 3.0 
       && numberOfOpponentsNearPassPlayingTeammate < 3
     )
  {
    //spieler wollte einen regulaeren pass spielen, der leider, leider
    //schief ging und abgefangen wurde. solch eine episode ignorieren wir.
    LOG_FLD(0,"ModLearnPositioningLearner::classifyCompletedEpisode: NOT USABLE :-(")
    return false;
  }
  LOG_FLD(0,"ModLearnPositioningLearner::classifyCompletedEpisode: USABLE :-)")
  return true;
}

//============================================================================
// BEGINN: nebensaechliche Methode, werden aktuell nicht verwendet
//============================================================================
/** Every time the referee sends a message, this function will be called. The parameter
    is true when the playmode has been changed. Note that there is a subtile difference
    between playmode changes and other referee messages...
*/
bool ModLearnPositioningLearner::onRefereeMessage(bool PMChange) {
  
  return true;
}

/** A string entered on the keyboard will be sent through this messages. If you process
    this string, you should return true. This will prevent sending the string to
    other modules as well. Return false if you don't process the message or if you want
    the string to be sent to other modules.
*/
bool ModLearnPositioningLearner::onKeyboardInput(const char *str) {

  return false;
}

/** Any hear message that does not come from the referee can be processed using this message.
    Unlike the keyboard input, a hear message will always be sent to all modules.
*/
bool ModLearnPositioningLearner::onHearMessage(const char *str) {

  return false;
}

/** This function will be called whenever a player type is changed. Note that player type
    changes can occur before the module is initialized, so you cannot be sure that you did not
    miss some changes before the beginning of the game.
    ownTeam is true, when the change happened in the own team. Remember that opponent player
    changes usually result in an unknown type (-1).

    This function will also be called, if ModAnalyse (or any other module...) makes new
    assumptions on the type of opponent players. You should then check what has changed
    in Player::possibleTypes[].
*/
bool ModLearnPositioningLearner::onChangePlayerType(bool ownTeam,int unum,int type) {

  return false;
}
//============================================================================
// ENDE: nebensaechliche Methode, werden aktuell nicht verwendet
//============================================================================

//============================================================================
// writeOutEpisode
//============================================================================
void 
ModLearnPositioningLearner::writeOutEpisode( int success )
{
  //increment counters
  if (success == EPISODE_USABLE_FOR_LEARNING_FAILURE) ivFailureEpisodesCounter++;
  if (success == EPISODE_USABLE_FOR_LEARNING_SUCCESS) ivSuccessEpisodesCounter++;
  if (success == EPISODE_IGNORABLE_FOR_LEARNING)      ivIgnoreEpisodesCounter++;
  //open file
  ivpEpisodeFileHandle = fopen( ivEpisodeFileName, "a" ); //append
  fprintf(ivpEpisodeFileHandle, "START_EPISODE\n");
  //first: write out general episode data
  fprintf(ivpEpisodeFileHandle, "LENGTH %d\n", ivEpisode.getLength() );
  fprintf(ivpEpisodeFileHandle, "CLASSIFICATION %d\n", success);
  for (int i=0; i<ivEpisode.getLength(); i++)
  {
    fprintf(ivpEpisodeFileHandle, "TIMESTEP %d\n", i);
    fprintf(ivpEpisodeFileHandle, "%s\n", ivEpisode.toStringState(i).c_str() );
    fprintf(ivpEpisodeFileHandle, "%s\n", ivEpisode.toStringReward(i).c_str() );
  }
  //second: write out states / rewards
  fprintf(ivpEpisodeFileHandle, "END_EPISODE\n");
  //close file
  fclose(ivpEpisodeFileHandle);
}
