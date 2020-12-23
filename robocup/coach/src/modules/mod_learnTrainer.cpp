#include "mod_learnTrainer.h"

#include "coach.h"
#include "logger.h"
#include "valueparser.h"
#include "geometry2d.h"


const char ModLearnTrainer::modName[]="ModLearnTrainer";

//=============================================================================
// METHOD init
//=============================================================================
/**
 * This will be called after the rest of the coach has been initialised
 * and the connection to the server has been established.
 * If waitForTeam is set, you may also assume that all own players are already
 * on the field. Put all module specific initialisation here.
 *
 * Die Methode ist recht lang und besteht aus mehreren Teilen:
 * 
 * [TEIL I]
 * * Setzen von Standardwerten fuer Instanzenvariablen
 * 
 * [TEIL II]
 * * Einlesen von Parametern aus Konfigurationsdatei(en)
 * 
 * [TEIL III]
 * * Anlegen der SituationCollection-Instanzen
 * * erfoglt separat fuer Trainings- und Evaluationssituationen
*/
bool 
ModLearnTrainer::init(int argc,char **argv) 
{
  LOG_FLD(0,"\nInitialising Module ModLearnTrainer ...");
  //in-/decrement these parameters so that they can be handled by VP
  argc--;
  argv++;
  ivWaitForOneCycle = 0;
  //we assume to not start looping immediately
  ivDoLoopOverSituations= false;

  //[TEIL I]
  //--DEFAULT-WERTE DER INSTANZENVARIABLEN SETZEN-------------------------
  ivLastProtocolledAt                =  -1;
  ivMaxSequenceRunningTime	 				 = 100000; // default: don't stop
  ivStopSequenceIfOppBall            = false;
  ivStopSequenceIfTeamBall           = false;
  ivStopSequenceIfBallNotKickable    = false;
  ivStopSequenceIfBallOutside        = false;
  ivStopSequenceIfBallIntercepted    = false;
  ivStopSequenceIfGoal               = false;
  ivStopSequenceIfTeamOffside        = false;
  ivStopSequenceIfBallTooDistant     = false;
  ivBallDistanceStopThreshold        = 10.0;
  ivStopSequenceIfGoalApproached     = false;
  ivGoalApproachStopThreshold        = 20.0;
  ivStopSequenceIfBallApproached     = false;
  ivBallApproachStopThreshold        = 1.0;
  ivStopSequenceIfBallTacklable      = false;
  ivDesiredTackleSuccessProbability  = 0.75;
  ivStopSequenceIfOpponentGoalShot   = false;
  ivStopAfterAsManySequenceRepetitions      = 1;
  ivStopSequenceAfterMaxSequenceRunningTime = false;
  ivStopAfterAsManyPlayerTypeReseedings     = 1;
  ivStartWithCornerKickLeft          = false;
  
  ivNumberOfSituations = 10;
  ivCreateRandomSituations = false;
  
  ivStatisticsFrequency				= 100;
  ivAutoStart						= true;
  
  ivSequenceLoopCounter = 0;
  ivSequenceCounter		= 0;
  ivSequenceStartTime	= 0;
  ivWaitTimeAfterSequenceStart = 10;
  ivSequenceStartWaitingCounter = 0;
  ivReseedCounter		= 0;
  ivTakenExtraTime = 0;
  ivExtraTime      = 0;//TG: fuer infprojrss geaendert von 2 auf 0
  
  ivStatisticsTotalTime	= 0;
  ivCurrentSequenceLength = 0;
  ivStatisticsMaxSequenceLength = 0;

  //Evaluation
  ivDoEvaluations						   	   = false;
  ivNumberOfEvaluationSituations			   = 100;
  ivDoEvaluationAfterAsManyTrainSequences      = 100;
  ivStartEvaluation							   = ivDoEvaluations;
  ivEvaluationSequenceCounter				   = 0;
    
  //[TEIL II]
  //--OPTIONEN AUS COACH-CONF-DATEI EINLESEN------------------------------
  ValueParser vp1(Options::coach_conf, "ModLearnTrainer");
    vp1.get("stopAfterAsManySequenceRepetitions", 
           ivStopAfterAsManySequenceRepetitions);
    vp1.get("maxSequenceRunningTime", ivMaxSequenceRunningTime);
    vp1.get("waitTimeAfterSequenceStart", ivWaitTimeAfterSequenceStart);
    vp1.get("stopSequenceIfOppBall",ivStopSequenceIfOppBall);
    vp1.get("stopSequenceIfTeamBall", ivStopSequenceIfTeamBall);
    vp1.get("stopSequenceIfBallNotKickable", ivStopSequenceIfBallNotKickable);
    vp1.get("stopSequenceIfBallOutside", ivStopSequenceIfBallOutside);
//    vp1.get("stopif_pass_success", nsParameter::stopif_pass_success);
    vp1.get("stopSequenceIfBallIntercepted", ivStopSequenceIfBallIntercepted);
    vp1.get("stopSequenceIfGoal", ivStopSequenceIfGoal);
    vp1.get("stopSequenceIfTeamOffside;", ivStopSequenceIfTeamOffside);
    vp1.get("stopSequenceIfBallTooDistant", ivStopSequenceIfBallTooDistant);
    vp1.get("ballDistanceStopThreshold", ivBallDistanceStopThreshold);
    vp1.get("stopSequenceIfGoalApproached", ivStopSequenceIfGoalApproached);
    vp1.get("goalApproachStopThreshold", ivGoalApproachStopThreshold);
    vp1.get("stopSequenceIfBallApproached", ivStopSequenceIfBallApproached);
    vp1.get("ballApproachStopThreshold", ivBallApproachStopThreshold);
    vp1.get("stopSequenceAfterMaxSequenceRunningTime", 
            ivStopSequenceAfterMaxSequenceRunningTime);
    vp1.get("stopSequenceIfBallTacklable", ivStopSequenceIfBallTacklable);
    vp1.get("desiredTackleSuccessProbability", ivDesiredTackleSuccessProbability);
    vp1.get("stopSequenceIfOpponentGoalShot", ivStopSequenceIfOpponentGoalShot);
    vp1.get("statisticsFrequency", ivStatisticsFrequency);
//    vp1.get("targetspeed", nsParameter::targetspeed);
//    vp1.get("speed_tolerance", nsParameter::ballspeed_tolerance);
    vp1.get("autoStart", ivAutoStart);
//    vp1.get("kickrange", nsParameter::kickrange);
    vp1.get("startWithCornerKickLeft", ivStartWithCornerKickLeft);
    vp1.get("stopAfterAsManyPlayerTypeReseedings", 
            ivStopAfterAsManyPlayerTypeReseedings);
    vp1.get("createRandomSituations", ivCreateRandomSituations );
    vp1.get("numberOfSituations", ivNumberOfSituations);
    vp1.get("learningMode",ivLearningMode);
    //Evaluation
    vp1.get("doEvaluations",ivDoEvaluations);
    vp1.get("numberOfEvaluationSequences",ivNumberOfEvaluationSituations);
    vp1.get("doEvaluationAfterAsManyTrainSequences",
             ivDoEvaluationAfterAsManyTrainSequences);

  //--BEHANDLUNG WEITERER NOTWENDIGER DATEIEN-----------------------------  
  ValueParser vp2(argc,argv);


  //Datei in der eine Menge Situationen gespeichert sind.
  sprintf(ivSituationFileName,"%s","\0");  
  sprintf(ivEvaluationSituationFileName,"%s","\0");  
  sprintf(ivProtocolFileName, "%s","\0");  

  vp1.get("situationFileLearnTrainer", ivSituationFileName, cvMaxStringLength);
  if ( strlen(ivSituationFileName) == 0)
    vp2.get("situationFileLearnTrainer", ivSituationFileName, cvMaxStringLength);
  vp1.get("evaluationSituationFileLearnTrainer", 
          ivEvaluationSituationFileName, cvMaxStringLength);
  if ( strlen(ivEvaluationSituationFileName) == 0)
    vp2.get("evaluationSituationFileLearnTrainer", 
            ivEvaluationSituationFileName, cvMaxStringLength);


  //Ausgabedatei - Ergebnisse und Nuetzliches werden dorthin geschrieben.
  vp1.get("protFileLearnTrainer", ivProtocolFileName, cvMaxStringLength);
  if ( strlen(ivProtocolFileName) == 0)
    vp2.get("protFileLearnTrainer", ivProtocolFileName, cvMaxStringLength);
  ivWriteProtocol = false;
  if ( strlen(ivProtocolFileName)>1 ) 
  {
    cout<<"Protfile "<<ivProtocolFileName<<endl<<flush;
    ivpProtocolFileHandle=fopen(ivProtocolFileName,"w");
    ivWriteProtocol=true;
  }

  cout<<"Writing Protocol for ModLearnTrainer: "<<ivWriteProtocol<<endl<<flush;

  //[TEIL III]
  //--ANLEGEN DER SITUATIONSSAMMLUNGEN----------------------------------------
  switch (ivLearningMode)
  {
    //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  	case 0: //normal/default
    //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    {
  	  ivpSituations 
	      = new SituationCollection(
	              SituationCollection::cvs_SITUATION_COLLECTION_TYPE_NORMAL);
      break;
    }
    //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  	case 1: //kick learning
    //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    {
  	  ivpSituations 
  	    = new SituationCollection(
  	            SituationCollection::cvs_SITUATION_COLLECTION_TYPE_KICK_LEARN);
  	  ivpEvaluationSituations
  	    = new SituationCollection(
  	            SituationCollection::cvs_SITUATION_COLLECTION_TYPE_KICK_LEARN);
  	  ivStopSequenceIfOppBall           = false;
  	  ivStopSequenceIfTeamBall          = false;
  	  ivStopSequenceIfBallNotKickable   = true;
  	  ivStopSequenceIfBallOutside       = false;
  	  ivStopSequenceIfBallIntercepted   = false;
  	  ivStopSequenceIfGoal              = false;
      ivStopSequenceIfTeamOffside       = true;
	  break;
    }
    //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  	case 2: //intercept learning
    //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    {
  	  ivpSituations 
  	    = new SituationCollection(
             SituationCollection
             ::cvs_SITUATION_COLLECTION_TYPE_INTERCEPT_LEARN);
  	  ivpEvaluationSituations
  	    = new SituationCollection(
              SituationCollection
              ::cvs_SITUATION_COLLECTION_TYPE_INTERCEPT_LEARN);
  	  ivStopSequenceIfOppBall         = false;
  	  ivStopSequenceIfTeamBall        = false;
  	  ivStopSequenceIfBallNotKickable = false;
  	  ivStopSequenceIfBallOutside     = false;
  	  ivStopSequenceIfBallIntercepted = true;
  	  ivStopSequenceIfGoal						= false;
      ivStopSequenceIfTeamOffside     = true;
  	  break;
    }
    //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
	  case 3: //intercept evaluation
    //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    {
  	  ivpSituations 
  	    = new SituationCollection(
              SituationCollection
              ::cvs_SITUATION_COLLECTION_TYPE_EVALUATE_INTERCEPT_LERN);	  
  	  ivpEvaluationSituations
  	    = new SituationCollection(
              SituationCollection
              ::cvs_SITUATION_COLLECTION_TYPE_EVALUATE_INTERCEPT_LERN);
  	  ivStopSequenceIfOppBall					= false;
  	  ivStopSequenceIfTeamBall				= false;
  	  ivStopSequenceIfBallNotKickable	= false;
  	  ivStopSequenceIfBallOutside			= false;
  	  ivStopSequenceIfBallIntercepted	= true;
  	  ivStopSequenceIfGoal						= false;
      ivStopSequenceIfTeamOffside     = false;
  	  break;
    }
    //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    case 4: //learning positioning for midfield / attack
    //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    {
      ivpSituations 
        = new SituationCollection(
              SituationCollection
              ::cvs_SITUATION_COLLECTION_TYPE_ATTACK_POSITIONING_LEARN);
      ivpEvaluationSituations
        = new SituationCollection(
              SituationCollection
              ::cvs_SITUATION_COLLECTION_TYPE_ATTACK_POSITIONING_LEARN);
      ivStopSequenceIfOppBall         = true;
      ivStopSequenceIfTeamBall        = false;
      ivStopSequenceIfBallNotKickable = false;
      ivStopSequenceIfBallOutside     = true;
      ivStopSequenceIfBallIntercepted = false;
      ivStopSequenceIfGoal            = true;
      ivStopSequenceIfTeamOffside     = true;
      break;
    }
    //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    case 5: //hassle learning
    //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    {
      ivpSituations 
        = new SituationCollection(
                SituationCollection::cvs_SITUATION_COLLECTION_TYPE_HASSLE_LEARN);
      ivpEvaluationSituations
        = new SituationCollection(
                SituationCollection::cvs_SITUATION_COLLECTION_TYPE_HASSLE_LEARN);
      ivStopSequenceIfOppBall           = false;
      ivStopSequenceIfTeamBall          = true;
      ivStopSequenceIfBallNotKickable   = false;
      ivStopSequenceIfBallOutside       = true;
      ivStopSequenceIfBallIntercepted   = true;
      ivStopSequenceIfGoal              = true;
      ivStopSequenceIfTeamOffside       = false;
      ivStopSequenceIfBallTooDistant    = true;
    break;
    }
    //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  	case 6: //infprojrss b3
    //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    {
  	  ivpSituations
  	    = new SituationCollection(
             SituationCollection
             ::cvs_SITUATION_COLLECTION_TYPE_INTERCEPT_LEARN);
  	  ivpEvaluationSituations
  	    = new SituationCollection(
              SituationCollection
              ::cvs_SITUATION_COLLECTION_TYPE_INTERCEPT_LEARN);
  	  ivStopSequenceIfOppBall         = false;
  	  ivStopSequenceIfTeamBall        = false;
  	  ivStopSequenceIfBallNotKickable = false;
  	  ivStopSequenceIfBallOutside     = true;
  	  ivStopSequenceIfBallIntercepted = false;
  	  ivStopSequenceIfGoal			  = true;
      ivStopSequenceIfTeamOffside     = false;
      ivLastPositionAPlayerToucedBall = Vector(0.0,0.0);
      ivLastTimeAPlayerTouchedBall    = 0;
  	  break;
    }
    //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  	case 7: //infprojrss b1
    //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    {
  	  ivpSituations
  	    = new SituationCollection(
             SituationCollection
             ::cvs_SITUATION_COLLECTION_TYPE_INTERCEPT_LEARN);
  	  ivpEvaluationSituations
  	    = new SituationCollection(
              SituationCollection
              ::cvs_SITUATION_COLLECTION_TYPE_INTERCEPT_LEARN);
  	  ivStopSequenceIfOppBall         = false;
  	  ivStopSequenceIfTeamBall        = false;
  	  ivStopSequenceIfBallNotKickable = false;
  	  ivStopSequenceIfBallOutside     = false;
  	  ivStopSequenceIfBallIntercepted = false;
  	  ivStopSequenceIfGoal			  = true;
      ivStopSequenceIfTeamOffside     = false;
      ivStopSequenceIfBallApproached  = true; // !!!!!!!!!
      //ivBallApproachStopThreshold = aus Datei eingelesen!
      ivLastPositionAPlayerToucedBall = Vector(0.0,0.0);
      ivLastTimeAPlayerTouchedBall    = 0;
  	  break;
    }
    //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  	case 8: //infprojrss b2
    //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    {
  	  ivpSituations
  	    = new SituationCollection(
             SituationCollection
             ::cvs_SITUATION_COLLECTION_TYPE_INTERCEPT_LEARN);
  	  ivpEvaluationSituations
  	    = new SituationCollection(
              SituationCollection
              ::cvs_SITUATION_COLLECTION_TYPE_INTERCEPT_LEARN);
  	  ivStopSequenceIfOppBall         = false;
  	  ivStopSequenceIfTeamBall        = false;
  	  ivStopSequenceIfBallNotKickable = false;
  	  ivStopSequenceIfBallOutside     = false;
  	  ivStopSequenceIfBallIntercepted = true;
  	  ivStopSequenceIfGoal			  = true;
      ivStopSequenceIfTeamOffside     = false;
      ivStopSequenceIfBallApproached  = false;
      ivLastPositionAPlayerToucedBall = Vector(0.0,0.0);
      ivLastTimeAPlayerTouchedBall    = 0;
  	  break;
    }
    //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  	case 9: //infprojrss c
    //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    {
  	  ivpSituations
  	    = new SituationCollection(
             SituationCollection
             ::cvs_SITUATION_COLLECTION_TYPE_KICK_LEARN);
  	  ivpEvaluationSituations
  	    = new SituationCollection(
              SituationCollection
              ::cvs_SITUATION_COLLECTION_TYPE_KICK_LEARN);
  	  ivStopSequenceIfOppBall         = true;
  	  ivStopSequenceIfTeamBall        = false;
  	  ivStopSequenceIfBallNotKickable = true;
  	  ivStopSequenceIfBallOutside     = true;
  	  ivStopSequenceIfBallIntercepted = false;
  	  ivStopSequenceIfGoal = true;
          ivStopSequenceIfTeamOffside     = false;
          ivStopSequenceIfBallApproached  = false;
          ivLastPositionAPlayerToucedBall = Vector(0.0,0.0);
          ivLastTimeAPlayerTouchedBall    = 0;
        break;
    }
  }

  if ( ivCreateRandomSituations )
  {
    LOG_FLD(1, "ModLearnTrainer: Creating "
               <<ivNumberOfSituations<<" random situations.");
    ivpSituations->createRandomTable(ivNumberOfSituations);
    
    if ( strlen(ivSituationFileName) ) 
    {
      LOG_FLD(1, "ModLearnTrainer: Saving all "
                 <<ivNumberOfSituations<<" random situations just created "
                 <<"to file"<<ivSituationFileName);
      ivpSituations->saveTable( ivSituationFileName );
    }
  }
  else
  {
    if ( strlen(ivSituationFileName) ) 
    {
      LOG_FLD(1, "ModLearnTrainer: Loading situation table from " 
                 << ivSituationFileName <<" (maximally "<<ivNumberOfSituations
                 <<" situations).");
      ivpSituations->loadTable(ivSituationFileName, 
                               ivNumberOfSituations);
    }
    else
    {
      LOG_FLD(1, "ModLearnTrainer: ERROR: On the one hand, I am told to NOT"
        <<" create random situations, on the other hand, no situation file"
        <<" is specified (exit)."<<flush);
      exit(0);
    }
  }

  if (ivDoEvaluations)
    if ( strlen(ivEvaluationSituationFileName) ) 
    {
      cout << "Loading Evaluation Situation Table from " << ivEvaluationSituationFileName<<endl<<flush;
      ivpEvaluationSituations->loadTable(ivEvaluationSituationFileName,
                                         ivNumberOfEvaluationSituations);
    }
    else
    {
      cout<<"Did not find a the file containing situations (filename=="<<
        ivEvaluationSituationFileName<<")"<<endl<<flush;
      cout<<" => Creating "<<ivNumberOfEvaluationSituations
          <<" Random Situations for Evaluation ... ";
	  ivpEvaluationSituations->createRandomTable(ivNumberOfEvaluationSituations);
  	cout<<"done."<<endl<<flush;
    }
  ivPlayerType = 0;
  return true;
}

//=============================================================================
// METHOD writeOutSituationProtocol
//=============================================================================
void 
ModLearnTrainer::writeOutSituationProtocol(int terminated)
{
//  if(nsSequence::loops<2) // forget the first nsSequence
//    return;
//  if(ivWriteProtocol == false)
//    return;
//  if(   nsSequence::last_protocolled_at == fld.getTime()  
//     && terminated != GOAL)
//    return;
//
//  nsSequence::last_protocolled_at = fld.getTime();
//  if(nsSequence::loops == nsParameter::stop_after_loops)
//  { 
//    cout << "\nProtokolliere..." << flush;
//    fprintf(ivpProtocolFileHandle, 
//      "%d %.3f %.3f %.3f %.3f %.3f %.3f %.3f %.3f %.3f %.3f %.3f\n", 
//      ivPlayerType,
//      fld.plType[ivPlayerType].player_speed_max, 
//      fld.plType[ivPlayerType].stamina_inc_max, 
//      fld.plType[ivPlayerType].player_decay, 
//      fld.plType[ivPlayerType].inertia_moment, 
//      fld.plType[ivPlayerType].dash_power_rate, 
//      fld.plType[ivPlayerType].player_size, 
//      fld.plType[ivPlayerType].kickable_margin, 
//      fld.plType[ivPlayerType].kick_rand, 
//      fld.plType[ivPlayerType].extra_stamina, 
//      fld.plType[ivPlayerType].effort_max, 
//      fld.plType[ivPlayerType].effort_min);
//    fprintf(ivpProtocolFileHandle, "%ld\n", nsStatistics::total_time);
//    fflush(ivpProtocolFileHandle);
//  }
}

//=============================================================================
// METHOD destroy
//=============================================================================
/** The framework will call this routine just before the connection to the server
    goes down. Up to this point, all data structures of the main coach are still intact.
*/
bool 
ModLearnTrainer::destroy() 
{
  if (ivpSituations)
    delete ivpSituations;
  if (ivpProtocolFileHandle)
    fclose(ivpProtocolFileHandle);
  return true;
}

//=============================================================================
// METHOD behave
//=============================================================================
/** Similar to the BS2kAgent behave routine, this one here will be called every cycle.
    It is currently synched to the see_global messages. Put code in here that
    should be called once per cycle just after the visual information has been updated.
*/
bool 
ModLearnTrainer::behave() 
{
	  LOG_FLD(0,"Entering behave() Module ModLearnTrainer ..."<<std::flush);
	  LOG_FLD(0,"Aktueller Spielmodus: "<<fld.getPM()<<std::flush);

  int  stopped; 
  long reseed_count=0;

  if ( ivAutoStart )
  {
  	ivDoLoopOverSituations = true;
    cout << "\nStarting first Sequence!" << flush;
    this->startNewSequence();
    fld.setPM( PM_play_on );
    ivAutoStart = false; //set it to false to not enter this block again
    return true;
  }

  stopped = checkSequenceStopCriterion();

  LOG_FLD(0,"From checkSequenceStopCriterion(): stopped = "<<stopped);

  //kick learn
  if (ivLearningMode == 1)
  if (   ivStopSequenceIfBallNotKickable
      && stopped == MOD_LEARNTRAINER_STATE_BALL_LEFT_KICKRANGE )
    stopped = MOD_LEARNTRAINER_STATE_TIME_OVER_CURRENT_SEQUENCE;
      
  //intercept learn
  if (ivLearningMode == 2 || ivLearningMode == 3)
  if (   ivStopSequenceIfBallIntercepted
      && stopped == MOD_LEARNTRAINER_STATE_BALL_INTERCEPTED )
    stopped = MOD_LEARNTRAINER_STATE_TIME_OVER_CURRENT_SEQUENCE;

  //positioning learn
  if (ivLearningMode == 4)
  {
	  if (   ivStopSequenceIfOppBall
		  && stopped == MOD_LEARNTRAINER_STATE_OP_HAS_BALL )
		stopped = MOD_LEARNTRAINER_STATE_TIME_OVER_CURRENT_SEQUENCE;
	  if (   (ivStopSequenceIfBallOutside || ivStopSequenceIfTeamOffside)
		  && stopped == MOD_LEARNTRAINER_STATE_GAME_INTERRUPTED )
		stopped = MOD_LEARNTRAINER_STATE_TIME_OVER_CURRENT_SEQUENCE;
	  if (   ivStopSequenceIfGoal
		  && stopped == MOD_LEARNTRAINER_STATE_GOAL )
		stopped = MOD_LEARNTRAINER_STATE_TIME_OVER_CURRENT_SEQUENCE;
	  if (   ivStopSequenceIfGoalApproached
		  && stopped == MOD_LEARNTRAINER_STATE_GOAL_APPROACHED )
		stopped = MOD_LEARNTRAINER_STATE_TIME_OVER_CURRENT_SEQUENCE;
  }

  //hassle learn
  if (ivLearningMode == 5)
  {
	  if (   ivStopSequenceIfBallIntercepted
		  && stopped == MOD_LEARNTRAINER_STATE_BALL_INTERCEPTED )
		stopped = MOD_LEARNTRAINER_STATE_TIME_OVER_CURRENT_SEQUENCE;
	  if (   ivStopSequenceIfBallTacklable
		  && stopped == MOD_LEARNTRAINER_STATE_BALL_TACKLABLE )
		stopped = MOD_LEARNTRAINER_STATE_TIME_OVER_CURRENT_SEQUENCE;
	  if (   ivStopSequenceIfOpponentGoalShot
		  && stopped == MOD_LEARNTRAINER_STATE_OPPONENT_GOAL_SHOT )
		stopped = MOD_LEARNTRAINER_STATE_TIME_OVER_CURRENT_SEQUENCE;
	  if (   ivStopSequenceIfBallOutside
		  && stopped == MOD_LEARNTRAINER_STATE_GAME_INTERRUPTED )
		stopped = MOD_LEARNTRAINER_STATE_TIME_OVER_CURRENT_SEQUENCE;
	  if (   ivStopSequenceIfGoal
		  && stopped == MOD_LEARNTRAINER_STATE_GOAL )
		stopped = MOD_LEARNTRAINER_STATE_TIME_OVER_CURRENT_SEQUENCE;
	  if (   ivStopSequenceIfBallTooDistant
		  && stopped == MOD_LEARNTRAINER_STATE_BALL_TOO_DISTANT )
		stopped = MOD_LEARNTRAINER_STATE_TIME_OVER_CURRENT_SEQUENCE;
  }

  //infprojrss b3
  if (   ivLearningMode == 6
      && stopped == MOD_LEARNTRAINER_STATE_GOAL )
  {
    stopped = MOD_LEARNTRAINER_STATE_TIME_OVER_CURRENT_SEQUENCE;
    LOG_FLD(0,"B3: Stop sequence due to goal.");
  }
  if (   ivLearningMode == 6
      && stopped == MOD_LEARNTRAINER_STATE_ILLEGAL_GOAL )
  {
    stopped = MOD_LEARNTRAINER_STATE_TIME_OVER_CURRENT_SEQUENCE;
    LOG_FLD(0,"B3: Stop sequence due to ILLEGAL goal.");
  }
  if (   ivLearningMode == 6
      && stopped == MOD_LEARNTRAINER_STATE_GAME_INTERRUPTED )
  {
    stopped = MOD_LEARNTRAINER_STATE_TIME_OVER_CURRENT_SEQUENCE;
    LOG_FLD(0,"B3: Stop sequence due to game interruption.");
  }
  if (   ivLearningMode == 6
      && fld.getPM() != PM_play_on
      && fld.getPM() != RM_goal_r
      && fld.getPM() != PM_free_kick_l )
    stopped = MOD_LEARNTRAINER_STATE_TIME_OVER_CURRENT_SEQUENCE;


  //infprojrss b1
  if (   ivLearningMode == 7
      && stopped == MOD_LEARNTRAINER_STATE_GOAL )
    stopped = MOD_LEARNTRAINER_STATE_TIME_OVER_CURRENT_SEQUENCE;
  Vector ballPos; 
  ballPos.setX( fld.ball.pos_x ); 
  ballPos.setY( fld.ball.pos_y );
  if (   ivLearningMode == 7
      && stopped == MOD_LEARNTRAINER_STATE_BALL_APPROACHED
      && ballPos.distance(HIS_GOAL_CENTER) < 5.0 )
    stopped = MOD_LEARNTRAINER_STATE_CONTINUE;
  if (   ivLearningMode == 7
      && fld.getPM() != PM_play_on
      && fld.getPM() != RM_goal_r
      && fld.getPM() != PM_free_kick_l )
    stopped = MOD_LEARNTRAINER_STATE_TIME_OVER_CURRENT_SEQUENCE;

  //infprojrss b2
  if (   ivLearningMode == 8
      && stopped == MOD_LEARNTRAINER_STATE_GOAL )
    stopped = MOD_LEARNTRAINER_STATE_TIME_OVER_CURRENT_SEQUENCE;
  if (   ivLearningMode == 8
      && stopped == MOD_LEARNTRAINER_STATE_BALL_INTERCEPTED
      && ballPos.distance(HIS_GOAL_CENTER) < 5.0 )
    stopped = MOD_LEARNTRAINER_STATE_CONTINUE;//Tor schiessen
/*  else
	  if (   ivLearningMode == 8 //for helios
	      && stopped == MOD_LEARNTRAINER_STATE_CONTINUE
	      && ballPos.distance(HIS_GOAL_CENTER) < 5.0 )
	    stopped = MOD_LEARNTRAINER_STATE_BALL_INTERCEPTED;*/
  if (   ivLearningMode == 8
      && fld.getPM() != PM_play_on
      && fld.getPM() != RM_goal_r
      && fld.getPM() != PM_free_kick_l )
    stopped = MOD_LEARNTRAINER_STATE_TIME_OVER_CURRENT_SEQUENCE;

  //infprojrss c
  if (   ivLearningMode == 9
      && stopped == MOD_LEARNTRAINER_STATE_GOAL  )
    stopped = MOD_LEARNTRAINER_STATE_TIME_OVER_CURRENT_SEQUENCE;
  else
  if (   ivLearningMode == 9
      && (stopped == MOD_LEARNTRAINER_STATE_BALL_INTERCEPTED)
      && ballPos.distance(HIS_GOAL_CENTER) < 5.0 )
    stopped = MOD_LEARNTRAINER_STATE_CONTINUE;//Tor schiessen
  else
  if (   ivStopSequenceIfBallNotKickable
      && stopped == MOD_LEARNTRAINER_STATE_BALL_LEFT_KICKRANGE )
    stopped = MOD_LEARNTRAINER_STATE_TIME_OVER_CURRENT_SEQUENCE;
  if (   ivLearningMode == 9
      && fld.getPM() != PM_play_on
      && fld.getPM() != RM_goal_r
      && fld.getPM() != PM_free_kick_l )
    stopped = MOD_LEARNTRAINER_STATE_TIME_OVER_CURRENT_SEQUENCE;
  Vector dribbleTarget( -FIELD_BORDER_X + PENALTY_AREA_LENGTH, PENALTY_AREA_WIDTH / 2.0 );
  Vector myPos(fld.myTeam[0].pos_x, fld.myTeam[0].pos_y);
  if (   ivLearningMode == 9
      && stopped == MOD_LEARNTRAINER_STATE_CONTINUE 
      && myPos.distance(dribbleTarget) < 1.0 )
    stopped = MOD_LEARNTRAINER_STATE_BALL_APPROACHED; // hack: actually means, dribble target approached!

  if (ivWaitForOneCycle > 0)
      stopped = MOD_LEARNTRAINER_STATE_TIME_OVER_CURRENT_SEQUENCE;

  LOG_FLD(0,"Starting the switch: stopped = "<<stopped);
  switch (stopped)
  {
    case MOD_LEARNTRAINER_STATE_BALL_APPROACHED:
    case MOD_LEARNTRAINER_STATE_BALL_INTERCEPTED:
    {
      //ACHTUNG: Diesen Sonderfall verwenden wir an dieser Stelle
      //lediglich, um es dem Spieler zu ermöglichen, nachdem er
      //den Ball erfolgreich abgefangen hat, diesen ins leere
      //gegnerische Tor schiessen zu koennen. Fuer die eigentliche
      //Intercept-Aufgabe ist dieser Trick nicht notwendig; er
      //dient lediglich dazu, im Rahmen der Praktikumswettkämpfe,
      //der Erfolg des Spielers am Ende am aktuellen Spielstand
      //ablesen zu koennen.

      MSG::sendMsg( MSG::MSG_MOVE,
                    "ball",
                    FIELD_BORDER_X - 0.3,
                    0.0,
                    RAD2DEG(0.0) ,
                    0.0,
                    0.0);
      char objectIdentifier[100];
      sprintf(objectIdentifier, "player %s %d", fld.getMyTeamName(), 1);//2 for helios
      MSG::sendMsg( MSG::MSG_MOVE,
                    objectIdentifier,
                    FIELD_BORDER_X - 0.3 - 0.4 - fld.myTeam[0].vel_x,
                    0.0 - fld.myTeam[0].vel_y,
                    RAD2DEG(0.0) ,
                    0.0,
                    0.0);
      break;
    }
    case MOD_LEARNTRAINER_STATE_CONTINUE:
    {
      ivCurrentSequenceLength++;

      if(fld.getPM() != PM_play_on)
      {
        MSG::sendMsg(MSG::MSG_CHANGE_MODE, PM_play_on);
        fld.setPM( PM_play_on );
      }
       
      if ( ivSequenceStartWaitingCounter > 0 )
      {
        ivSequenceStartWaitingCounter -- ;
        this->restartCurrentSequence();
      }
       
      /*if (    ivStartWithCornerKickLeft
             && ivSequenceStartTime >= fld.getTime() - 2)
          MSG::sendMsg(MSG::MSG_CHANGE_MODE, PM_corner_kick_l);
        else 
          if (fld.getPM()!= PM_play_on)
            fld.setPM(PM_play_on);*/
//cout<<"BEHAVE: ivSequenceLoopCounter  = "<<ivSequenceLoopCounter<<endl<<flush;
//cout<<"BEHAVE: ivSequenceCounter      = "<<ivSequenceCounter<<endl<<flush;
//cout<<"BEHAVE: ivCurrentSequenceLength= "<<ivCurrentSequenceLength<<endl<<flush;
//cout<<"BEHAVE: ivDoLoopOverSituations = "<<ivDoLoopOverSituations<<endl<<flush;
      ivStatisticsTotalTime	++;
      return true;
    }
    case MOD_LEARNTRAINER_STATE_TIME_OVER_CURRENT_SEQUENCE:
    {
      //aktuelle Sequenz ist hiermit beendet!
      ivStatisticsTotalTime--;
      ivCurrentSequenceLength--;
      writeOutSituationProtocol(stopped);
      if (ivCurrentSequenceLength > ivStatisticsMaxSequenceLength)
	    ivStatisticsMaxSequenceLength = ivCurrentSequenceLength;
      ivCurrentSequenceLength = 0;
      if (ivWaitForOneCycle > 0)
      {
        cout<<"wait4oneCycle ist true   .. und ivStartEval = "<<ivStartEvaluation<<endl;
        // nach dem Spiel ist vor dem Spiel
      	MSG::sendMsg(MSG::MSG_CHANGE_MODE, PM_play_on);
        fld.setPM( PM_play_on );
        ivWaitForOneCycle = 0;
      	if ( ivStartEvaluation == true )
      	{
          if (ivDoEvaluations)
          {
            ivStartEvaluation = true;
            this->startNewEvaluationSequence();
          }
      	}
      	else
      	{
          if ( ivDoEvaluations
               &&
               (
                 ivDoEvaluationAfterAsManyTrainSequences == 0
                 ||
                 ivSequenceCounter 
                   % ivDoEvaluationAfterAsManyTrainSequences == 0
               )
             )
          {
      	    ivStartEvaluation = true;
          }
          else
            ivStartEvaluation = false;
          this->startNewSequence(); 
      	}
      }
      else
      {
        if (ivTakenExtraTime < ivExtraTime)
        {
          ivTakenExtraTime ++;
          break;
        }      
        ivTakenExtraTime = 0;
        if (ivStartEvaluation==true)
        {
            MSG::sendMsg(MSG::MSG_CHANGE_MODE, PM_kick_in_l);
            fld.setPM( PM_kick_in_l );
        }
        else
        {
          //stop the ball
          MSG::sendMsg( MSG::MSG_MOVE,
                        "ball",
                        0.0,
                        0.0,
                        RAD2DEG(0.0) ,
                        0.0,
                        -0.0);
          MSG::sendMsg(MSG::MSG_CHANGE_MODE, PM_free_kick_l);
          fld.setPM( PM_free_kick_l );
        }
        ivWaitForOneCycle = 1;
      }
      break;
    }
    case MOD_LEARNTRAINER_STATE_OVERALL_TIMEOUT:
    {
      ivDoLoopOverSituations = false;
      //stop, as we have repeated each sequence 
  	  //ivStopAfterAsManySequenceRepetitions times
      writeOutSituationProtocol();
      MSG::sendMsg(MSG::MSG_CHANGE_MODE, PM_time_over);
      fld.setPM( PM_time_over );
      break;
    }
    case MOD_LEARNTRAINER_STATE_ENOUGH_PT_RESEEDINGS:
    {
      ivDoLoopOverSituations = false;
      break;
    }
  }
  return false;

 


    
/*  if ( ivSequenceLoopCounter >= ivStopAfterAsManySequenceRepetitions)
  {
    //tats?chlicher Stopp
	//Hier k?nnte ein Neustart mit neuem/n Spielertyp(en) erfolgen ...
    ivPlayerType++;
    if (    ivPlayerType > 6 
         && ivStopAfterAsManyPlayerTypeReseedings >= ivReseedCounter)
    {
      ivPlayerType = 1;
      cout << "\nReseeding Player Types!" << flush;
      reseed_count++;
      MSG::sendMsg(MSG::MSG_RESEED_HETRO);
    }
    if (ivPlayerType<7) 
      //Format: "(change_player_type TEAMNAME PLNUMBER PLTYPENUMBER)"
      MSG::sendMsg( MSG::MSG_CHANGE_PLAYER_TYPE, 
                    fld.getMyTeamName(), 
                    1, //Warum nur der erste Spieler?
                    ivPlayerType);
  }*/
  
}

//===========================================================================
// START NEW SEQUENCE
//===========================================================================
void 
ModLearnTrainer::startNewSequence()
{
/*      if (ivpSituations) delete ivpSituations;
      LOG_FLD(1, "ModLearnTrainer: Re-loading situation table from " 
                 << ivSituationFileName <<" (maximally "<<ivNumberOfSituations
                 <<" situations).");
      ivpSituations 
        = new SituationCollection(
              SituationCollection
              ::cvs_SITUATION_COLLECTION_TYPE_ATTACK_POSITIONING_LEARN);
      ivpSituations->loadTable(ivSituationFileName, 
                               ivNumberOfSituations);*/
  //Erstmal setzen wir alle fuer den Neustart einer Sequenz relevanten 
  //Parameter.
  ivSequenceCounter ++;
  cout << "\nSequence-Number:" << ivSequenceCounter<<flush;
  ivSequenceStartTime = fld.getTime();
  ivSequenceStartWaitingCounter = ivWaitTimeAfterSequenceStart;
  //Ganz wichtig: Wir wollen frische, unverbrauchte Spieler benutzen. 
  //Deswegen schicken wir an den SoccerServer "(recover)", was dazu
  //fuehrt das Stamina, Effert und Recovery auf die Werte zurueckgesetzte
  //werden, die sie am Anfang eines Spieles haben.
  MSG::sendMsg(MSG::MSG_RECOVER);
  //Start der n?chsten Situation/Sequenz.
  bool wasThisSequenceTheLastInTheSituationCollection 
         = ivpSituations->doNextSituation();  
  //Falls FALSE zur?ckgeliefert wurde, so handelte es sich um die letzte
  //erste Sequenz innerhalb der Sequenzsammlung, die gerade gesetzt. 
  //wurde Es wird daher mit einer
  //neuen "?u?eren" Schleife (?ber alle Situationen) begonnen.
  if (wasThisSequenceTheLastInTheSituationCollection == false)
    ivSequenceLoopCounter++;
cout<<"Number of Sequence Repetitions: "<<ivSequenceLoopCounter<<endl;
}

//===========================================================================
// RESTART CURRENT SEQUENCE
//===========================================================================
void 
ModLearnTrainer::restartCurrentSequence()
{
  cout << "\nRESTART Sequence: Sequence-Number:" << ivSequenceCounter<<flush;
  ivSequenceStartTime = fld.getTime();
  MSG::sendMsg(MSG::MSG_RECOVER);
  ivpSituations->redoCurrentSituation();  
}

//===========================================================================
// START EVALUATION SEQUENCE
//===========================================================================
void 
ModLearnTrainer::startNewEvaluationSequence()
{
  //Erstmal setzen wir alle fuer den Neustart einer Sequenz relevanten 
  //Parameter.
  ivEvaluationSequenceCounter ++;
  cout << "\nEVALUATION: Sequence-Number:" << ivEvaluationSequenceCounter<<flush<<endl;
  ivSequenceStartTime = fld.getTime();
  //Ganz wichtig: Wir wollen frische, unverbrauchte Spieler benutzen. 
  //Deswegen schicken wir an den SoccerServer "(recover)", was dazu
  //f?hrt das Stamina, Effert und Recovery auf die Werte zur?ckgesetzte
  //werden, die sie am Anfang eines Spieles haben.
  MSG::sendMsg(MSG::MSG_RECOVER);
  //Start der naechsten Situation/Sequenz.
  bool wasThisSequenceNotTheFirstInTheSituationCollection 
         = ivpEvaluationSituations->doNextSituation();  
  if (ivEvaluationSequenceCounter == ivNumberOfEvaluationSituations)
  {
    ivStartEvaluation = false;
    ivEvaluationSequenceCounter = 0;
    cout<<"RESETING: from EVALUATION mode to normal LEARNING mode"<<endl;
  }
}

//===========================================================================
// CHECK STOP CRITERION
//===========================================================================
int 
ModLearnTrainer::checkSequenceStopCriterion()
{
  LOG_FLD(0,"Entering checkSequenceStopCriterion() in Module ModLearnTrainer ..."<<std::flush);
  LOG_FLD(0,"Current PM is "<<fld.getPM()<<", learning mode is "<<ivLearningMode);

  Vector player,ball;
  if (ivSequenceLoopCounter > ivStopAfterAsManySequenceRepetitions)
    return MOD_LEARNTRAINER_STATE_OVERALL_TIMEOUT;

  if ( ivStopAfterAsManyPlayerTypeReseedings <= ivReseedCounter) 
    return MOD_LEARNTRAINER_STATE_ENOUGH_PT_RESEEDINGS;

  if(fld.getTime() >= ivSequenceStartTime + ivMaxSequenceRunningTime)
  {
    cout << "\nTIMEOVER for current sequence" <<endl << flush;
    return MOD_LEARNTRAINER_STATE_TIME_OVER_CURRENT_SEQUENCE;
  }
  
  if (ivStopSequenceIfBallNotKickable)
  {
  	for (int i=0; i<11; i++)
  	{
  		if ( !fld.myTeam[i].alive )
  		  continue;
  		player.setX( fld.myTeam[i].pos_x );
  		player.setY( fld.myTeam[i].pos_y );
  		ball.setX( fld.ball.pos_x );
  		ball.setY( fld.ball.pos_y );
  		float delta =   (player-ball).norm()
  		              - (ServerParam::ball_size)
  		              - (ServerParam::player_size);
  		if ( delta > ServerParam::kickable_margin )
  		{
  			cout<<"BALL NOT KICKABLE ANY MORE"<<endl<<flush;
  			return MOD_LEARNTRAINER_STATE_BALL_LEFT_KICKRANGE;
  		}
  	}
  }

  if(ivStopSequenceIfBallApproached)
  {
    for (int i = 0 ; i < 11 ; i++)
    { // do consider goalie
      if(!fld.myTeam[i].alive)
        continue;
      player.setX( fld.myTeam[i].pos_x );
      player.setY( fld.myTeam[i].pos_y );
      ball.setX( fld.ball.pos_x );
      ball.setY( fld.ball.pos_y );
      //cout<<"###"<<(player-ball).norm();
      if((player-ball).norm() <= ivBallApproachStopThreshold )
      {
	    cout << "\nBALL_APPROACHED" << flush;
       	return MOD_LEARNTRAINER_STATE_BALL_APPROACHED;
      }
    }
  }

  if(ivStopSequenceIfBallIntercepted)
  {
    for (int i = 0 ; i < 11 ; i++)
    { // do consider goalie
      if(!fld.myTeam[i].alive) 
        continue;
      player.setX( fld.myTeam[i].pos_x );
      player.setY( fld.myTeam[i].pos_y );
      ball.setX( fld.ball.pos_x );
      ball.setY( fld.ball.pos_y );
      //cout<<"###"<<(player-ball).norm();
      if((player-ball).norm() <= KICK_RANGE ) //INFO: KICK_RANGE = 0.7+0.3+0.085
      {
	    cout << "\nBALL_INTERCEPTED" << flush;
       	return MOD_LEARNTRAINER_STATE_BALL_INTERCEPTED;
      }
    }    
  }
  
  if (ivStopSequenceIfOppBall)
  {
    bool weAreStillInBallPossession = false;
    for (int i = 0 ; i < 11 ; i++)
    { // do consider goalie
      if(!fld.myTeam[i].alive) 
        continue;
      player.setX( fld.myTeam[i].pos_x );
      player.setY( fld.myTeam[i].pos_y );
      ball.setX( fld.ball.pos_x );
      ball.setY( fld.ball.pos_y );
      if((player-ball).norm() <= KICK_RANGE ) //INFO: KICK_RANGE = 0.7+0.3+0.085
        weAreStillInBallPossession = true;
    }
    if (weAreStillInBallPossession == false)
      for (int i = 0 ; i < 11 ; i++)
      { // do consider goalie
        if(!fld.hisTeam[i].alive) 
          continue;
        player.setX( fld.hisTeam[i].pos_x );
        player.setY( fld.hisTeam[i].pos_y );
        ball.setX( fld.ball.pos_x );
        ball.setY( fld.ball.pos_y );
        if((player-ball).norm() <= KICK_RANGE ) //INFO: KICK_RANGE = 0.7+0.3+0.085
        {
          cout << "\nBALL_INTERCEPTED BY OPPONENT" << flush;
          return MOD_LEARNTRAINER_STATE_OP_HAS_BALL;
        }
      }    
  }
  
  if (ivStopSequenceIfGoal)
  {
    LOG_FLD(0,"Checking for goal ... ");
    ball.setX( fld.ball.pos_x );
    ball.setY( fld.ball.pos_y );
    Vector ballVel;
    ballVel.setX( fld.ball.vel_x );
    ballVel.setY( fld.ball.vel_y );
    Vector prevBall = ball - (1.0/0.94)*ballVel;
    double certainty = 0.085; //ball radius
    if (    (    fabs(ball.getX()) > FIELD_BORDER_X + certainty
              && (fabs(ball.getY()) <= 7.01 || fabs(prevBall.getY()) <= 7.01)
            )
         || fld.getPM() == RM_goal_l //ball ist ins linke tor gegangen (gegentor)
         || fld.getPM() == RM_goal_r //ball ist ins rechte tor gegangen
       )
    {
      cout << "\nTOR!!! TOR!!! TOR!!!" << flush;
      LOG_FLD(0,"  ... GOAL !!!");
      return MOD_LEARNTRAINER_STATE_GOAL;
    }
    LOG_FLD(0,"  ... NO GOAL! ball at "<<ball.getX()<<","<<ball.getY()<<" PM="<<fld.getPM()<<" GOAL-PM:"<<RM_goal_l<<","<<RM_goal_r);
  }

  if (ivStopSequenceIfBallOutside)
  {
    double certainty = 0.085; //ball radius
    if (   fabs(ball.getX()) >= FIELD_BORDER_X + certainty
        || fabs(ball.getY()) >= FIELD_BORDER_Y )
    {
      cout << "\nGAME INTERRUPTED (ball im aus)" << flush;
      return MOD_LEARNTRAINER_STATE_GAME_INTERRUPTED;
    }
  }
  
  if (ivStopSequenceIfTeamOffside)
  {
    if (fld.getPM() == PM_offside_l)
    {
      cout << "\nGAME INTERRUPTED (ball im abseits)" << flush;
      return MOD_LEARNTRAINER_STATE_GAME_INTERRUPTED;
    }
  }  
  
  if (ivStopSequenceIfGoalApproached)
  {
    Vector hisGoalCenter(FIELD_BORDER_X, 0.0);
    if ( (ball-hisGoalCenter).norm() < ivGoalApproachStopThreshold )
    {
      for (int i = 0 ; i < 11 ; i++)
      { // do consider goalie
        if(!fld.myTeam[i].alive) continue;
        player.setX( fld.myTeam[i].pos_x );
        player.setY( fld.myTeam[i].pos_y );
        ball.setX( fld.ball.pos_x );
        ball.setY( fld.ball.pos_y );
        if((player-ball).norm() <= KICK_RANGE ) //INFO: KICK_RANGE = 0.7+0.3+0.085
        {
          cout << "\nGOAL APPROACHED ("<<ivGoalApproachStopThreshold<<"m)" << flush;
          return MOD_LEARNTRAINER_STATE_GOAL_APPROACHED;
        }
      }    
    }
  }
  
  if (ivStopSequenceIfBallTacklable)
  {
    double option_tackle_dist      = 2.0,
           option_tackle_exponent  = 6.0,
           option_tackle_width     = 1.0,
           option_tackle_back_dist = 0.5,
           tackleProbability;
    
    ANGLE playerAngle;
    for (int i=0; i<11; i++)
    {
      if ( !fld.myTeam[i].alive )
        continue;
      player.setX( fld.myTeam[i].pos_x );
      player.setY( fld.myTeam[i].pos_y );
      ball.setX( fld.ball.pos_x );
      ball.setY( fld.ball.pos_y );
      playerAngle = fld.myTeam[i].bodyAng;

      Vector player_2_ball = ball - player;
      player_2_ball.rotate( - playerAngle.get_value_mPI_pPI() );

      if (player_2_ball.getX() >= 0.0)    
      {
        tackleProbability =   pow( player_2_ball.getX() / option_tackle_dist, 
                                   option_tackle_exponent ) 
                            + pow( fabs(player_2_ball.getY()) / option_tackle_width, 
                                   option_tackle_exponent );
      } 
      else
      {
        tackleProbability =   pow( player_2_ball.getX() / option_tackle_back_dist, 
                                   option_tackle_exponent) 
                            + pow( fabs(player_2_ball.getY()) / option_tackle_width, 
                                   option_tackle_exponent );
      }

      if (tackleProbability >= 1.0) 
        tackleProbability = 0.0;
      else 
        if (tackleProbability < 0.0) 
          tackleProbability = 1.0;
        else 
          tackleProbability = (1.0 - tackleProbability);
    
      if (tackleProbability >= ivDesiredTackleSuccessProbability)
        return MOD_LEARNTRAINER_STATE_BALL_TACKLABLE;
    }
  }
  
  if (ivStopSequenceIfOpponentGoalShot)
  {
    Vector ballVel;
    for (int i=0; i<11; i++)
    {
      if ( !fld.hisTeam[i].alive )
        continue;
      player.setX( fld.hisTeam[i].pos_x );
      player.setY( fld.hisTeam[i].pos_y );
    }
    ball.setX( fld.ball.pos_x );
    ball.setY( fld.ball.pos_y );
    ballVel.setX( fld.ball.vel_x );
    ballVel.setY( fld.ball.vel_y );
    if (    (player-ball).norm() > 5.0
         && ballVel.norm() > 1.0 
         && ballVel.getY() < 0.0  )
      return MOD_LEARNTRAINER_STATE_OPPONENT_GOAL_SHOT;
  }

  if (ivStopSequenceIfBallTooDistant)
  {
    for (int i = 0 ; i < 11 ; i++)
    { 
      if(!fld.myTeam[i].alive) 
        continue;
      if (i!=0) continue;
      player.setX( fld.myTeam[i].pos_x );
      player.setY( fld.myTeam[i].pos_y );
      ball.setX( fld.ball.pos_x );
      ball.setY( fld.ball.pos_y );
      if ((player-ball).norm() >= ivBallDistanceStopThreshold ) 
      {
        cout << "\nBALL TOO DISTANT FROM ONE OF MY PLAYERS" << flush;
        return MOD_LEARNTRAINER_STATE_BALL_TOO_DISTANT;
      }
    }    
  }
  
  if ( ivLearningMode == 6 ) //infprojrss b3
  {
    //shot from not within the gray-shaded area
	XYRectangle2d targetGoalShotArea( Vector( 52.5 - 16.5, -34.0 ),
	                                  Vector( 52.5, -40.32/2.0));
	//    XYRectangle2d targetGoalShotArea( Vector( 52.5 - 16.5, -34.0 ),
	  //                                    Vector( 52.5, -4));
    ball.setX( fld.ball.pos_x );
    ball.setY( fld.ball.pos_y );
    Vector ballVel;
    ballVel.setX( fld.ball.vel_x );
    ballVel.setY( fld.ball.vel_y );
    for (int i = 0 ; i < 11 ; i++)
    {
      if(!fld.myTeam[i].alive)
        continue;
      player.setX( fld.myTeam[i].pos_x );
      player.setY( fld.myTeam[i].pos_y );

      if ((player-ball).norm() < KICK_RANGE)
      {
        ivLastTimeAPlayerTouchedBall = fld.getTime();
        ivLastPositionAPlayerToucedBall = player;
      }
      Vector nextBallPos = ball + ballVel;
      Vector nextBallVel = ballVel;
      nextBallVel *= 0.94;
      Vector overNextBallPos = nextBallPos + nextBallVel;
      if (targetGoalShotArea.inside( ivLastPositionAPlayerToucedBall ) == false)
        if (   (   overNextBallPos.getX() > FIELD_BORDER_X && ball.getX() < FIELD_BORDER_X
                && (    fabs( ball.getY() ) < 7.01
                    || fabs( nextBallPos.getY() ) < 7.01
                    || fabs( overNextBallPos.getY() ) < 7.01
                  )
               )
            || (   (player-ball).norm() < KICK_RANGE
                && ball.getX() > FIELD_BORDER_X - 3.0
                && fabs(ball.getY()) < 10.0)
           )
        {
          LOG_FLD(0,<<"WARNING: AN ILLEGAL GOAL WILL BE SHOT!");
          return MOD_LEARNTRAINER_STATE_ILLEGAL_GOAL;
        }
    }
    //goal kick for team right
    Vector abstossPunkt1(47.0, -9.0);
    Vector abstossPunkt2(47.0, 9.0);
    if (   (   (ball-abstossPunkt1).norm() < 1.0
            || (ball-abstossPunkt2).norm() < 1.0 )
        && ballVel.norm() < 0.01 )
        {
          LOG_FLD(0,<<"INFO: BALL WENT OUT OF FIELD!");
          return MOD_LEARNTRAINER_STATE_GAME_INTERRUPTED;
        }
  }
  return MOD_LEARNTRAINER_STATE_CONTINUE;
}


//=============================================================================
// NOTE: The following methods are currently not used.
//=============================================================================

/** Every time the referee sends a message, this function will be called. The parameter
    is true when the playmode has been changed. Note that there is a subtile difference
    between playmode changes and other referee messages...
*/
bool ModLearnTrainer::onRefereeMessage(bool PMChange) {
  LOG_FLD(0,<<"ModLearnTrainer::onRefereeMessage: PMChange="<<PMChange);
  return true;
}

/** A string entered on the keyboard will be sent through this messages. If you process
    this string, you should return true. This will prevent sending the string to
    other modules as well. Return false if you don't process the message or if you want
    the string to be sent to other modules.
*/
bool ModLearnTrainer::onKeyboardInput(const char *str) {

  return false;
}

/** Any hear message that does not come from the referee can be processed using this message.
    Unlike the keyboard input, a hear message will always be sent to all modules.
*/
bool ModLearnTrainer::onHearMessage(const char *str) {

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
bool ModLearnTrainer::onChangePlayerType(bool ownTeam,int unum,int type) {

  return false;
}
