#include "neuro_kick08_bms.h"
#define BASELEVEL 0

//============================================================================
//DECLARATION OF STATIC VARIABLES
//============================================================================
bool   NeuroKick08::cvInitialized = false;
double NeuroKick08::cvMaximalKickVelocityDeviation = 0.1;
double NeuroKick08::cvMaximalKickDirectionDeviation= DEG2RAD(10);
Net    *NeuroKick08::cvpNetArray[NUMBER_OF_NEURAL_NETS_NK08];

const double NeuroKick08ItrActions::ivKickPowerMin           = 5;
const double NeuroKick08ItrActions::ivKickPowerInc           = 5;
const double NeuroKick08ItrActions::ivKickPowerSteps         = 20;
const double NeuroKick08ItrActions::ivFinetuneKickPowerInc   = 1;
const double NeuroKick08ItrActions::ivFinetuneKickPowerSteps = 10;
const double NeuroKick08ItrActions::ivKickAngleMin           = 0;
const double NeuroKick08ItrActions::ivKickAngleInc           = 2*PI/72;//==360/72=5
const double NeuroKick08ItrActions::ivKickAngleSteps         = 62;
const double NeuroKick08ItrActions::ivFinetuneKickAngleInc   = 2*PI/360;
const double NeuroKick08ItrActions::ivFinetuneKickAngleSteps = 90;

const double NeuroKick08::cvcMinimalDesiredBallPlayerDistanceRelative  = 0.1;
const double NeuroKick08::cvcMaximalDesiredBallPlayerDistanceRelative  = 1.0;


void NeuroKick08ItrActions::reset( bool finetune, double orig_pwr, ANGLE orig_ang)
{
    if( !finetune )
    {
        ivAngleMin   = ANGLE( ivKickAngleMin );
        ivAngleInc   = ANGLE( ivKickAngleInc );
        ivAngleSteps = ivKickAngleSteps;

        ivPowerMin   = ivKickPowerMin;
        ivPowerInc   = ivKickPowerInc;
        ivPowerSteps = ivKickPowerSteps;
    }
    else
    {
        ivAngleMin   = orig_ang - ANGLE( 0.5 * ivFinetuneKickAngleSteps * ivFinetuneKickAngleInc );
        ivAngleInc   = ANGLE( ivFinetuneKickAngleInc );
        ivAngleSteps = ivFinetuneKickAngleSteps + 1;

        ivPowerMin   = orig_pwr - ( 0.5 * ivFinetuneKickPowerSteps * ivFinetuneKickPowerInc );
        ivPowerInc   = ivFinetuneKickPowerInc;
        ivPowerSteps = ivFinetuneKickPowerSteps + 1;
    }

    ivAngleDone = 0;
    ivPowerDone = 0;

    ivAngle = ivAngleMin;
    ivPower = ivPowerMin;
}

Cmd_Body* NeuroKick08ItrActions::next()
{
    if( ivPowerDone < ivPowerSteps && ivAngleDone < ivAngleSteps )
    {
        ivAction.unset_lock();
        ivAction.unset_cmd();

        ivAction.set_kick( ivPower, ivAngle.get_value_mPI_pPI() );
        ivAngle += ivAngleInc;
        if( ++ivAngleDone >= ivAngleSteps )
        {
            ivAngle     = ivAngleMin;
            ivAngleDone = 0;

            ivPower     += ivPowerInc;
            ivPowerDone++;
        }
        return &ivAction;
    }
    return NULL;
}

NeuroKick08State::NeuroKick08State()
{
    ivBallPositionRelativeAngle = 0.0;
    ivRelativeBallDistance      = 0.0;
    ivRelativeBallVelocity      = Vector( 0.0, 0.0 );
    ivRelativeBodyToKickAngle   = 0.0;
    ivMyWorldPosition           = Vector( 0.0, 0.0 );
    ivMyWorldVelocity           = Vector( 0.0, 0.0 );
    ivMyWorldANGLE              = ANGLE( 0.0 );
    ivBallWorldPosition         = Vector( 0.0, 0.0 );
    ivBallWorldVelocity         = Vector( 0.0, 0.0 );
    ivOpponentPosition          = Vector( 1000.0, 1000.0 );
    ivOpponentBodyAngle         = ANGLE( 0.0 );
    ivOpponentBodyAngleAge      = 0;
}
//============================================================================
// NeuroKick08::init()
//============================================================================
bool NeuroKick08::init ( char const * confFile, 
                         int          argc, 
                         char const* const* argv) 
{
  if(cvInitialized) 
    return true;
    
  if ( ! OneOrTwoStepKick::init(confFile,argc,argv) ) 
  {
    ERROR_OUT 
    << "\nCould not initialize OneOrTwoStepKick behavior - stop loading.";
    exit(1);
  }
  if ( ! OneStepKick::init(confFile,argc,argv) ) 
  {
    ERROR_OUT 
    << "\nCould not initialize OneStepKick behavior - stop loading.";
    exit(1);
  }

  //currently at most 2 nets are used
  for (int i=0; i<NUMBER_OF_NEURAL_NETS_NK08; i++)
    cvpNetArray[i] = new Net();
  
  char NN0_name[500];
  char NN1_name[500];
  char NN2_name[500];
  char NN3_name[500];
  char NN4_name[500];
  sprintf(NN0_name,"%s","../data/nets_neuro_kick08/kickLearn08_30.net\0");
  sprintf(NN1_name,"%s","../data/nets_neuro_kick08/kickLearn08_27.net\0");
  sprintf(NN2_name,"%s","../data/nets_neuro_kick08/kickLearn08_25.net\0");
  sprintf(NN3_name,"%s","../data/nets_neuro_kick08/kickLearn08_20.net\0");
  sprintf(NN4_name,"%s","../data/nets_neuro_kick08/kickLearn08_15.net\0");
  
  ValueParser vp(confFile,"Neuro_Kick08");
  //vp.set_verbose();
  vp.get("maximal_kick_velocity_deviation",cvMaximalKickVelocityDeviation);
  vp.get("maximal_kick_direction_deviation",cvMaximalKickDirectionDeviation);
  
  vp.get("NN0_name", NN0_name, 500);
  vp.get("NN1_name", NN1_name, 500);
  vp.get("NN2_name", NN2_name, 500);
  vp.get("NN3_name", NN3_name, 500);
  vp.get("NN4_name", NN4_name, 500);
  
  if (cvpNetArray[0]->load_net(NN0_name) == FILE_ERROR) 
  {
    ERROR_OUT<<"NeuroKick_bms: No net0 "<<NN0_name<<" found - stop loading\n";
    exit(0);
  }
  if (cvpNetArray[1]->load_net(NN1_name) == FILE_ERROR) 
  {
    ERROR_OUT<<"NeuroKick_bms: No net1 "<<NN1_name<<" found - stop loading\n";
    exit(0);
  }
  if (cvpNetArray[2]->load_net(NN2_name) == FILE_ERROR) 
  {
    ERROR_OUT<<"NeuroKick_bms: No net2 "<<NN2_name<<" found - stop loading\n";
    exit(0);
  }
  if (cvpNetArray[3]->load_net(NN3_name) == FILE_ERROR) 
  {
    ERROR_OUT<<"NeuroKick_bms: No net3 "<<NN3_name<<" found - stop loading\n";
    exit(0);
  }
  if (cvpNetArray[4]->load_net(NN4_name) == FILE_ERROR) 
  {
    ERROR_OUT<<"NeuroKick_bms: No net4 "<<NN4_name<<" found - stop loading\n";
    exit(0);
  }
  
  cvInitialized = true;
  return true;
}

//============================================================================
// CONSTRUCTOR
// NeuroKick08::NeuroKick08()
//============================================================================
NeuroKick08::NeuroKick08()
{
  ivDoFineTuning = false;
  ivInitInCycle = -1;
  ivFakeStateTime = -1;
  ivpOneOrTwoStepKickBehavior = new OneOrTwoStepKick();
  ivpOneStepKickBehavior      = new OneStepKick();
}

//============================================================================
// DESTRUCTOR
// NeuroKick08::~NeuroKick08()
//============================================================================
NeuroKick08::~NeuroKick08()
{
  for (int i=0; i<NUMBER_OF_NEURAL_NETS_NK08; i++)
    if (cvpNetArray[i]) delete cvpNetArray[i];
  if (ivpOneOrTwoStepKickBehavior)
    delete ivpOneOrTwoStepKickBehavior;
  if (ivpOneStepKickBehavior)
    delete ivpOneStepKickBehavior;
}

//============================================================================
// SKILL'S MAIN METHOD
// NeuroKick08::get_cmd()
//============================================================================
bool
NeuroKick08::get_cmd(Cmd & cmd)
{
  if ( !cvInitialized ) 
  {
    ERROR_OUT << "\nNeuroKick08 not initialized!";
    return false;
  }
  if ( WSinfo::ws->time != ivInitInCycle) 
  {
    ERROR_OUT << "\nNeuroKick08::get_cmd() called without prior initialization!";
    return false;
  }
  LOG_MOV(0,<<"Starting NeuroKick08 behavior (dir: "<< ivTargetKickAngle.get_value()
    <<", vel: "<<ivTargetKickVelocity<<").");
//  cout<<"NeuroKick08: Starting NeuroKick08 behavior (dir: "<< ivTargetKickAngle.get_value()
//    <<", vel: "<<ivTargetKickVelocity<<")."<<endl;
  return decideForAnAction(cmd);
}

//============================================================================
// SKILL'S INITIALIZATION METHOD
// NeuroKick08::kick_to_pos_with_initial_vel()
//============================================================================
void 
NeuroKick08::kick_to_pos_with_initial_vel  (  double          vel,
                                              const Vector & pos) 
{
  NeuroKick08State state = getCurrentState();
  ivTargetKickAngle    = (pos - state.ivBallWorldPosition).ARG();
  ivTargetKickVelocity = vel;
  ivTargetKickPosition = pos;
  ivDoTargetTracking   = true;
  ivInitInCycle = WSinfo::ws->time;
}

//============================================================================
// SKILL'S INITIALIZATION METHOD
// NeuroKick08::kick_to_pos_with_final_vel()
//============================================================================
void 
NeuroKick08::kick_to_pos_with_final_vel  ( double         vel,
                                           const Vector &pos )
{
  NeuroKick08State state = getCurrentState();
  ivTargetKickAngle    = (pos - state.ivBallWorldPosition).ARG();

  ivTargetKickVelocity = computeInitialVelocityForFinalVelocity
                           ( ( WSinfo::ball->pos - pos ).norm() ,
                             vel );
  //Does not work if target_vel > ball_speed_max... use max speed instead!
  if ( ivTargetKickVelocity > ServerOptions::ball_speed_max) 
  {
    ivTargetKickVelocity = ServerOptions::ball_speed_max;
    LOG_ERR(0,<<"NeuroKick: Point "<<pos<<" too far away, using max vel!");
  }

  ivTargetKickPosition = pos;
  ivDoTargetTracking   = true;
  ivInitInCycle = WSinfo::ws->time;
}

//============================================================================
// SKILL'S INITIALIZATION METHOD
// NeuroKick08::kick_to_pos_with_max_vel()
//============================================================================
void 
NeuroKick08::kick_to_pos_with_max_vel( const Vector &pos ) 
{
  kick_to_pos_with_initial_vel ( ServerOptions::ball_speed_max, pos );
}

//============================================================================
// SKILL'S INITIALIZATION METHOD
// NeuroKick08::kick_in_dir_with_max_vel()
//============================================================================
void NeuroKick08::kick_in_dir_with_max_vel(const ANGLE &dir ) 
{
  kick_in_dir_with_initial_vel( ServerOptions::ball_speed_max, dir );
}

//============================================================================
// SKILL'S INITIALIZATION METHOD
// NeuroKick08::kick_in_dir_with_initial_vel()
//============================================================================
void NeuroKick08::kick_in_dir_with_initial_vel( double vel,
                                                const ANGLE &dir ) 
{
  ivTargetKickAngle    = dir;
  ivTargetKickVelocity = vel;
  ivDoTargetTracking   = false;
  ivInitInCycle = WSinfo::ws->time;
}

//============================================================================
// HELPER METHOD FOR CLASSIFYING FINAL STATES
// NeuroKick08::checkIfStateIsASuccessState(.)
//============================================================================
bool
NeuroKick08::checkIfStateIsASuccessState( NeuroKick08State & succState,
                                          double            & deviation)
{
  //set the features, do the rotation
  this->completeState( succState );

  if (succState.ivRelativeBallDistance < 1.0)
  {
    //ball is still kickable
    return false;
  }
  Vector finalBallVector = succState.ivBallWorldVelocity;
  double finalBallVelocity = finalBallVector.norm();
  ANGLE finalBallANGLE    = finalBallVector.ARG();
  ANGLE deltaANGLE        = finalBallANGLE - ivTargetKickAngle;
  if ( fabs( deltaANGLE.get_value_mPI_pPI() ) > cvMaximalKickDirectionDeviation ) 
  {
    //too much angular deviation
    return false;
  }
  double deltaVelocity
    = fabs( finalBallVelocity-(ivTargetKickVelocity*ServerOptions::ball_decay));
  if ( deltaVelocity > cvMaximalKickVelocityDeviation )
  {
    //too much speed deviation
    return false;
  }
  //set return value
  deviation = deltaVelocity + 2.0*fabs( deltaANGLE.get_value_mPI_pPI() );  
  return true;
}

//============================================================================
// HELPER METHOD FOR INITIALIZATION
// NeuroKick08::computeInitialVelocityForFinalVelocity()
//============================================================================
double
NeuroKick08::computeInitialVelocityForFinalVelocity( double distance,
                                                     double finalVelocity )
{
  double remainingDistance = distance;
  double currentVelocity   = finalVelocity;
  while (remainingDistance > 0.0)
  {
    currentVelocity   /= ServerOptions::ball_decay;
    remainingDistance -= currentVelocity;
  }
  return currentVelocity;
}

//============================================================================
// NeuroKick08::completeState
//============================================================================
void 
NeuroKick08::completeState(NeuroKick08State &state)
{
  //fill elements for neural net input
  ANGLE tmpAngle = (state.ivBallWorldPosition - state.ivMyWorldPosition).ARG();
  tmpAngle -= state.ivMyWorldANGLE;
  state.ivBallPositionRelativeAngle 
    = ( ( tmpAngle ).get_value_mPI_pPI()) * (180.0/PI);  //value in [-180,180]
float v1 = ( (state.ivBallWorldPosition - state.ivMyWorldPosition).norm()
          - ServerOptions::ball_size
          - ServerOptions::player_size);
float v2 = ( WSinfo::me->kick_radius 
          - ServerOptions::ball_size 
          - ServerOptions::player_size);
  state.ivRelativeBallDistance      
    = v1
      /
      v2  ;
  //cout<<"TESTAUSGABE: b:"<<state.ivBallWorldPosition<<" p:"<<state.ivMyWorldPosition<<" rel:"<<state.ivRelativeBallDistance<<endl;
  //cout<<"TESTAUSGABE: v1:"<<v1<<" v2:"<<v2<<" kr:"<<WSinfo::me->kick_radius<<" bs:"<<ServerOptions::ball_size<<" ps:"<<ServerOptions::player_size<<endl;
  state.ivRelativeBallVelocity      = state.ivBallWorldVelocity
                                      - state.ivMyWorldVelocity;
  tmpAngle = state.ivMyWorldANGLE;
  tmpAngle -= ivTargetKickAngle;
  state.ivRelativeBodyToKickAngle   = (  tmpAngle ).get_value_mPI_pPI()
                                      * (180.0/PI); //value in [-180,180]
  //now do the rotation according to the kick direction
  //note: we do not have to rotate the ball distance to the player as well
  //      as the relative angular distance between ball and player orientation
  state.ivRelativeBallVelocity.ROTATE( ANGLE(2.0*PI) - ivTargetKickAngle );
  //state.ivRelativeBodyToKickAngle ==> no rotation necessary, this has been
  //  realized above already (by subtracting ivTargetKickAngle!

}

//============================================================================
// NeuroKick08::getWSState(NeuroKick08State &state)
//============================================================================
void
NeuroKick08::getWSState(NeuroKick08State &state)
{
  //fill world information
  state.ivMyWorldPosition = WSinfo::me->pos;
  state.ivMyWorldVelocity = WSinfo::me->vel;
  state.ivMyWorldANGLE    = WSinfo::me->ang;
  state.ivBallWorldPosition = WSinfo::ball->pos;
  state.ivBallWorldVelocity = WSinfo::ball->vel;
  //fill opponent information
  PlayerSet pset= WSinfo::valid_opponents;
  pset.keep_and_sort_closest_players_to_point(1, state.ivBallWorldPosition);
  if ( pset.num )
  {
    state.ivOpponentPosition = pset[0]->pos;
    state.ivOpponentBodyAngle = pset[0]->ang;
    state.ivOpponentBodyAngleAge = pset[0]->age_ang;
  }
  else
  {
    state.ivOpponentPosition = Vector(1000,1000); // outside pitch
    state.ivOpponentBodyAngle = ANGLE(0);
    state.ivOpponentBodyAngleAge = 1000;
  }
}  

//============================================================================
// NeuroKick08::set_state()
//============================================================================
void 
NeuroKick08::set_state( const Vector &myPos, 
                        const Vector &myVel,
                        const ANGLE  &myAng,
                        const Vector &ballPos,
                        const Vector &ballVel)
{
  ivFakeState.ivMyWorldPosition     = myPos;
  ivFakeState.ivMyWorldVelocity     = myVel;
  ivFakeState.ivMyWorldANGLE        = myAng;
  ivFakeState.ivBallWorldPosition   = ballPos;
  ivFakeState.ivBallWorldVelocity   = ballVel;
  ivFakeStateTime = WSinfo::ws->time;
  ivInitInCycle = -1;
}

//============================================================================
// NeuroKick08::reset_state()
//============================================================================
void 
NeuroKick08::reset_state() 
{
  ivFakeStateTime = -1;
  ivInitInCycle   = -1;
}

//============================================================================
// NeuroKick08::getCurrentState()
//============================================================================
NeuroKick08State
NeuroKick08::getCurrentState()
{
  NeuroKick08State returnValue;
  if (ivFakeStateTime == WSinfo::ws->time) 
  {
    returnValue = ivFakeState;
  } 
  else 
  {
    getWSState(returnValue);
  }
  return returnValue;
}

//============================================================================
// NeuroKick08::isFailureState()
//============================================================================
bool             
NeuroKick08::isFailureState(const NeuroKick08State & s)
{
//cout<<"NeuroKick08: Check for failure state: s.ivRelativeBallDistance                    = "<<s.ivRelativeBallDistance<<endl;
//cout<<"                                      cvcMaximalDesiredBallPlayerDistanceRelative = "<<cvcMaximalDesiredBallPlayerDistanceRelative<<endl;
  if (s.ivRelativeBallDistance > cvcMaximalDesiredBallPlayerDistanceRelative)
    return true; //ball too near to end of kickable area
  if (s.ivRelativeBallDistance < cvcMinimalDesiredBallPlayerDistanceRelative)
    return true; //ball too near to me (danger of collision) 
  return false; //no failure
}

//============================================================================
// NeuroKick08::setNeuralNetInput( )
//============================================================================
void             
NeuroKick08::setNeuralNetInput( NeuroKick08State       & state,
                                double                   targetVelocity,
                                ANGLE                    targetDir,
                                float                  * net_in )
{
  //NOTE: targetDir and targetVelocity are disregarded, 
  //      these parameters are considered for downward compatibility only

  //set neural net input
  net_in[0] = state.ivBallPositionRelativeAngle / 180.0;
  net_in[1] =   state.ivRelativeBallDistance
              * DEFAULT_PLAYER_KICKABLE_AREA_DURING_TRAINING;
  net_in[2] = state.ivRelativeBallVelocity.getX();
  net_in[3] = state.ivRelativeBallVelocity.getY();
  net_in[4] = state.ivRelativeBodyToKickAngle / 180.0;
}

//============================================================================
// NeuroKick08::chooseCurrentNet( )
//============================================================================
Net * 
NeuroKick08::chooseCurrentNet()
{
  if (ivTargetKickVelocity > 2.7) return cvpNetArray[0]; //fuer 3.0 trainiert 
  if (ivTargetKickVelocity > 2.0) return cvpNetArray[1]; //fuer 2.7 trainiert
  //das netz, das fuer 2.5 trainiert wurde, wird nicht verwendet!
  if (ivTargetKickVelocity > 1.5) return cvpNetArray[3]; //fuer 2.0 trainiert
  return cvpNetArray[4];                                 //fuer 1.5 trainiert                                 
}

//============================================================================
// NeuroKick08::evaluateState()
//============================================================================
double
NeuroKick08::evaluateState( NeuroKick08State & state)
{
  //set the features, do the rotation
  this->completeState( state );

  //lowest possible value is state is a failure state
  //(considering cases where the ball goes outside the kickable area
  //or collides with the player)
  if (isFailureState(state))
  {
//cout<<"NeuroKick08: Considered state is a failure state!"<<endl;
    return 0.0;
  }
  //check whether oneortwostepkick behavior considers this state ok
  //(takes opponents into consideration)
  MyState dummy;
  dummy.my_pos = state.ivMyWorldPosition - state.ivMyWorldVelocity;
  dummy.my_vel = state.ivMyWorldVelocity;
  dummy.op_pos = state.ivOpponentPosition;
  dummy.op_bodydir = state.ivOpponentBodyAngle;
  dummy.op_bodydir_age = state.ivOpponentBodyAngleAge;
  if ( ! ivpOneStepKickBehavior->is_pos_ok( dummy, 
                                            state.ivBallWorldPosition ) )
  {
    //cout<<"NeuroKick08: Future position "<<state.ivMyWorldPosition<<" and ball@"<<state.ivBallWorldPosition<<" is not ok (says OneStepKick)!"<<endl;
    return 0.0;
  } 
  //ask the net
  this->setNeuralNetInput( state,
                           ivTargetKickVelocity,
                           ivTargetKickAngle,
                           ivpCurrentNet->in_vec );
  ivpCurrentNet->forward_pass( ivpCurrentNet->in_vec,
                               ivpCurrentNet->out_vec );
  return ivpCurrentNet->out_vec[0];
}

//============================================================================
// NeuroKick08::decideForAnAction()
//============================================================================
bool
NeuroKick08::decideForAnAction(Cmd & cmd)
{
  NeuroKick08State currentState = this->getCurrentState();
  NeuroKick08State successorState, bestSuccessorState;
  Cmd_Body bestAction;
  bool     actionFound = false;
  double   bestSuccessorStateValue = -1.0;
  double   bestKickPower = 0.0;
  Angle    bestKickAngle = 0.0;
  
  //try to use oneortwostepkick behavior, if possible
  if ( 1 ) //just in case we would like to switch of the use of 12step kick
  {
    double resultingVelocityAfter1StepKick,
           resultingVelocityAfter2StepKick;
    Cmd    cmdForOneStep, cmdForTwoStep;
    //try oneortwostep kick
    ivpOneOrTwoStepKickBehavior->set_state( currentState.ivMyWorldPosition,
                                            currentState.ivMyWorldVelocity,
                                            currentState.ivMyWorldANGLE,
                                            currentState.ivBallWorldPosition,
                                            currentState.ivBallWorldVelocity,
                                            currentState.ivOpponentPosition,
                                            currentState.ivOpponentBodyAngle,
                                            currentState.ivOpponentBodyAngleAge);
    if (ivDoTargetTracking)
    {
      ivpOneOrTwoStepKickBehavior->kick_to_pos_with_initial_vel(
                                            ivTargetKickVelocity,
                                            ivTargetKickPosition );
    }
    else
    {
      ivpOneOrTwoStepKickBehavior->kick_in_dir_with_initial_vel(
                                            ivTargetKickVelocity,
                                            ivTargetKickAngle );
    }
    ivpOneOrTwoStepKickBehavior->get_cmd( cmdForOneStep,
                                          cmdForTwoStep );
    ivpOneOrTwoStepKickBehavior->get_vel( resultingVelocityAfter1StepKick,
                                          resultingVelocityAfter2StepKick );
//cout<<"NK08: 12stepkick: 1->"<<resultingVelocityAfter1StepKick<<", 2->"
 //   <<resultingVelocityAfter2StepKick<<" (target="<<ivTargetKickVelocity<<")"<<endl;
    //let us decide if the quality of the kick, which can be obtained via
    //a oneortwostepkick, is sufficient for us
    if (   fabs(  resultingVelocityAfter1StepKick - ivTargetKickVelocity )
         < cvMaximalKickVelocityDeviation  )
    {
      cmd.cmd_body.clone( cmdForOneStep.cmd_body );
      //cout<<"NeuroKick08: MAKE ONE STEP KICK!"<<endl;
      return true;
    }
    if (   fabs(  resultingVelocityAfter2StepKick - ivTargetKickVelocity )
         < 0.85 * cvMaximalKickVelocityDeviation  )
    {
      cmd.cmd_body.clone( cmdForTwoStep.cmd_body );
      //cout<<"NeuroKick08: MAKE TWO STEP KICK!"<<endl;
      return true;
    }
  }

  //cout<<"NeuroKick08: MAKE A REAL   N E U R O   KICK!"<<endl;

  //so, a oneortwostepkick will not suffice, we need a real neuro kick
  ivpCurrentNet = chooseCurrentNet();
  for (int fineTuning=0; fineTuning<2; fineTuning++)
  {
    if ( fineTuning == 0 )
      ivActionInterator.reset(false);
    else
      ivActionInterator.reset(true, bestKickPower, ANGLE(bestKickAngle) );
    //loop over all actions
    while (Cmd_Body const * chosenAction = ivActionInterator.next() )  
    {
      //cpmpute successor state
      Tools::model_cmd_main( currentState.ivMyWorldPosition,
                             currentState.ivMyWorldVelocity,
                             currentState.ivMyWorldANGLE,
                             currentState.ivBallWorldPosition,
                             currentState.ivBallWorldVelocity,
                             * chosenAction,
                             successorState.ivMyWorldPosition,
                             successorState.ivMyWorldVelocity,
                             successorState.ivMyWorldANGLE,
                             successorState.ivBallWorldPosition,
                             successorState.ivBallWorldVelocity );
      double successorStateValue, deviation;
      //check if we arrive at a success state
      if ( NeuroKick08::checkIfStateIsASuccessState( successorState,
                                                     deviation )
         )
      {
        successorStateValue = 2.0 - deviation;
        //successorStateValue = 2.0;
      }
      else
        //evaluate the successor state
        successorStateValue = this->evaluateState( successorState );
      //if (fineTuning==0) {cout<<"NeuroKick08: ITERATION, aktuell: "<<*chosenAction<<" erbringt "<<successorStateValue;
      //if (isFailureState(successorState)) cout<<" (failure)"; cout<<endl;}
      if (    bestSuccessorStateValue < 0.0
           || actionFound == false 
           || bestSuccessorStateValue < successorStateValue )
      {
        bestSuccessorStateValue = successorStateValue;
        bestSuccessorState = successorState;
        bestAction = * chosenAction;
        bestAction.get_kick( bestKickPower,
                             bestKickAngle );
        actionFound = true;
      }
/*double kp, ka;
chosenAction->get_kick(kp,ka);
cout<<"NK08: found action: p="<<kp<<" an="<<RAD2DEG(ka)
  <<" | v="<<successorStateValue<<"\t";
cout<<"      succ: "<<successorState.ivBallPositionRelativeAngle
  <<", "<< successorState.ivRelativeBallDistance
  <<", "<<  successorState.ivRelativeBallVelocity.x
  <<", "<<  successorState.ivRelativeBallVelocity.y
  <<", "<<  successorState.ivRelativeBodyToKickAngle<<endl;*/

    } //end of while
    if ( ! ivDoFineTuning || actionFound == false )
      break;
  } //end of for loop
  if (actionFound)
  {
    //cout<<"NeuroKick08: Chosen Action: "<<bestAction<<" with value="<<bestSuccessorStateValue<<endl;

    return cmd.cmd_body.clone( bestAction );
  }
  return false;
}

