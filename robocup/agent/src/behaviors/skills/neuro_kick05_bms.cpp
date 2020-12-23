#include "neuro_kick05_bms.h"
#define BASELEVEL 0

//============================================================================
//DECLARATION OF STATIC VARIABLES
//============================================================================
// TG17: moved outside off class def for c++11 compatibility
const double NeuroKick05ItrActions::ivKickPowerMin           = 5;
const double NeuroKick05ItrActions::ivKickPowerInc           = 5;
const double NeuroKick05ItrActions::ivKickPowerSteps         = 20;
const double NeuroKick05ItrActions::ivFinetuneKickPowerInc   = 1;
const double NeuroKick05ItrActions::ivFinetuneKickPowerSteps = 10;
const double NeuroKick05ItrActions::ivKickAngleMin           = 0;
const double NeuroKick05ItrActions::ivKickAngleInc           = 2*PI/62;//==360/62=5
const double NeuroKick05ItrActions::ivKickAngleSteps         = 62;
const double NeuroKick05ItrActions::ivFinetuneKickAngleInc   = 2*PI/360;
const double NeuroKick05ItrActions::ivFinetuneKickAngleSteps = 90;

bool   NeuroKick05::cvInitialized = false;
double NeuroKick05::cvMaximalKickVelocityDeviation = 0.1;
double NeuroKick05::cvMaximalKickDirectionDeviation= DEG2RAD(10);
Net    *NeuroKick05::cvpNetArray[NUMBER_OF_NEURAL_NETS];

const double NeuroKick05::cvcMinimalDesiredBallPlayerDistanceRelative  = 0.1;
const double NeuroKick05::cvcMaximalDesiredBallPlayerDistanceRelative  = 1.0;

//============================================================================
// NeuroKick05::init()
//============================================================================
bool NeuroKick05::init ( char const * confFile, 
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
  for (int i=0; i<NUMBER_OF_NEURAL_NETS; i++)
    cvpNetArray[i] = new Net();
  
  char NN0_name[500];
  char NN1_name[500];
  char NN2_name[500];
  char NN3_name[500];
  sprintf(NN0_name,"%s","../data/nets_neuro_kick05/kickLearn_maxspeednet27.net\0"); //maxspeed
  sprintf(NN1_name,"%s","../data/nets_neuro_kick05/kickLearn_highspeednet25.net\0");//highspeed
  sprintf(NN2_name,"%s","../data/nets_neuro_kick05/kickLearn_averagespeednet20.net\0");//averagespeed
  sprintf(NN3_name,"%s","../data/nets_neuro_kick05/kickLearn_slowspeednet15.net\0");//slowspeed
  
  ValueParser vp(confFile,"Neuro_Kick05");
  //vp.set_verbose();
  vp.get("maximal_kick_velocity_deviation",cvMaximalKickVelocityDeviation);
  vp.get("maximal_kick_direction_deviation",cvMaximalKickDirectionDeviation);
  
  vp.get("NN0_name", NN0_name, 500);
  vp.get("NN1_name", NN1_name, 500);
  vp.get("NN2_name", NN2_name, 500);
  vp.get("NN3_name", NN3_name, 500);
  
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
  
  cvInitialized = true;
  return true;
}

//============================================================================
// CONSTRUCTOR
// NeuroKick05::NeuroKick05()
//============================================================================
NeuroKick05::NeuroKick05()
{
  ivDoFineTuning = false;
  ivInitInCycle = -1;
  ivFakeStateTime = -1;
  ivpOneOrTwoStepKickBehavior = new OneOrTwoStepKick();
  ivpOneStepKickBehavior      = new OneStepKick();
}

//============================================================================
// DESTRUCTOR
// NeuroKick05::~NeuroKick05()
//============================================================================
NeuroKick05::~NeuroKick05()
{
  for (int i=0; i<NUMBER_OF_NEURAL_NETS; i++)
    if (cvpNetArray[i]) delete cvpNetArray[i];
  if (ivpOneOrTwoStepKickBehavior)
    delete ivpOneOrTwoStepKickBehavior;
  if (ivpOneStepKickBehavior)
    delete ivpOneStepKickBehavior;
}

//============================================================================
// SKILL'S MAIN METHOD
// NeuroKick05::get_cmd()
//============================================================================
bool
NeuroKick05::get_cmd(Cmd & cmd)
{
  if ( !cvInitialized ) 
  {
    ERROR_OUT << "\nNeuroKick05 not initialized!";
    return false;
  }
  if ( WSinfo::ws->time != ivInitInCycle) 
  {
    ERROR_OUT << "\nNeuroKick05::get_cmd() called without prior initialization!";
    return false;
  }
  LOG_MOV(0,<<"Starting NeuroKick05 behavior (dir: "<< ivTargetKickAngle.get_value()
    <<", vel: "<<ivTargetKickVelocity<<").");
  //cout<<"NeuroKick05: Starting NeuroKick05 behavior (dir: "<< ivTargetKickAngle.get_value()
  //  <<", vel: "<<ivTargetKickVelocity<<")."<<endl;
  return decideForAnAction(cmd);
}

//============================================================================
// SKILL'S INITIALIZATION METHOD
// NeuroKick05::kick_to_pos_with_initial_vel()
//============================================================================
void 
NeuroKick05::kick_to_pos_with_initial_vel( double       vel,
                                           const Vector &pos )
{
  NeuroKick05State state = getCurrentState();
  ivTargetKickAngle    = (pos - state.ivBallWorldPosition).ARG();
  ivTargetKickVelocity = vel;
  ivTargetKickPosition = pos;
  ivDoTargetTracking   = true;
  ivInitInCycle = WSinfo::ws->time;
}

//============================================================================
// SKILL'S INITIALIZATION METHOD
// NeuroKick05::kick_to_pos_with_final_vel()
//============================================================================
void 
NeuroKick05::kick_to_pos_with_final_vel( double       vel,
                                         const Vector &pos )
{
  NeuroKick05State state = getCurrentState();
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
// NeuroKick05::kick_to_pos_with_max_vel()
//============================================================================
void 
NeuroKick05::kick_to_pos_with_max_vel( const Vector &pos ) 
{
  kick_to_pos_with_initial_vel ( ServerOptions::ball_speed_max, pos );
}

//============================================================================
// SKILL'S INITIALIZATION METHOD
// NeuroKick05::kick_in_dir_with_max_vel()
//============================================================================
void NeuroKick05::kick_in_dir_with_max_vel(const ANGLE &dir ) 
{
  kick_in_dir_with_initial_vel( ServerOptions::ball_speed_max, dir );
}

//============================================================================
// SKILL'S INITIALIZATION METHOD
// NeuroKick05::kick_in_dir_with_initial_vel()
//============================================================================
void NeuroKick05::kick_in_dir_with_initial_vel( double vel,
                                                const ANGLE &dir )
{
  ivTargetKickAngle    = dir;
  ivTargetKickVelocity = vel;
  ivDoTargetTracking   = false;
  ivInitInCycle = WSinfo::ws->time;
}

//============================================================================
// HELPER METHOD FOR INITIALIZATION
// NeuroKick05::computeInitialVelocityForFinalVelocity()
//============================================================================
double
NeuroKick05::computeInitialVelocityForFinalVelocity( double distance,
                                                     double finalVelocity )
{
  double  remainingDistance = distance;
  double  currentVelocity   = finalVelocity;
  while (remainingDistance > 0.0)
  {
    currentVelocity   /= ServerOptions::ball_decay;
    remainingDistance -= currentVelocity;
  }
  return currentVelocity;
}

//============================================================================
// NeuroKick05::completeState
//============================================================================
void 
NeuroKick05::completeState(NeuroKick05State &state)
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
// NeuroKick05::getWSState(NeuroKick05State &state)
//============================================================================
void
NeuroKick05::getWSState(NeuroKick05State &state)
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
// NeuroKick05::set_state()
//============================================================================
void 
NeuroKick05::set_state( const Vector &myPos, 
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
// NeuroKick05::reset_state()
//============================================================================
void 
NeuroKick05::reset_state() 
{
  ivFakeStateTime = -1;
  ivInitInCycle   = -1;
}

//============================================================================
// NeuroKick05::getCurrentState()
//============================================================================
NeuroKick05State
NeuroKick05::getCurrentState()
{
  NeuroKick05State returnValue;
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
// NeuroKick05::isFailureState()
//============================================================================
bool             
NeuroKick05::isFailureState(const NeuroKick05State & s)
{
//cout<<"NeuroKick05: Check for failure state: s.ivRelativeBallDistance                    = "<<s.ivRelativeBallDistance<<endl;
//cout<<"                                      cvcMaximalDesiredBallPlayerDistanceRelative = "<<cvcMaximalDesiredBallPlayerDistanceRelative<<endl;
  if (s.ivRelativeBallDistance > cvcMaximalDesiredBallPlayerDistanceRelative)
    return true; //ball too near to end of kickable area
  if (s.ivRelativeBallDistance < cvcMinimalDesiredBallPlayerDistanceRelative)
    return true; //ball too near to me (danger of collision) 
  return false; //no failure
}

//============================================================================
// NeuroKick05::setNeuralNetInput( )
//============================================================================
void             
NeuroKick05::setNeuralNetInput( NeuroKick05State       & state,
                                double                   targetVelocity,
                                ANGLE                    targetDir,
                                float                  * net_in )
{
  //NOTE: targetDir and targetVelocity are disregarded, 
  //      these parameters are considered for downward compatibility only
  
  //set neural net input
  net_in[0] = state.ivBallPositionRelativeAngle / 180.0;
  net_in[1] = state.ivRelativeBallDistance;
  net_in[2] = state.ivRelativeBallVelocity.getX();
  net_in[3] = state.ivRelativeBallVelocity.getY();
  net_in[4] = state.ivRelativeBodyToKickAngle / 180.0;
}

//============================================================================
// NeuroKick05::chooseCurrentNet( )
//============================================================================
Net * 
NeuroKick05::chooseCurrentNet()
{
  if (ivTargetKickVelocity > 2.3) return cvpNetArray[0]; //max speed
  if (ivTargetKickVelocity > 2.0) return cvpNetArray[1]; //high speed
  if (ivTargetKickVelocity > 1.5) return cvpNetArray[2]; //average speed
  return cvpNetArray[3];                                 //slow speed
}

//============================================================================
// NeuroKick05::evaluateState()
//============================================================================
double
NeuroKick05::evaluateState( NeuroKick05State & state)
{
  //set the features, do the rotation
  this->completeState( state );

  //lowest possible value is state is a failure state
  //(considering cases where the ball goes outside the kickable area
  //or collides with the player)
  if (isFailureState(state))
  {
//cout<<"NeuroKick05: Considered state is a failure state!"<<endl;
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
    //cout<<"NeuroKick05: Future position "<<state.ivMyWorldPosition<<" and ball@"<<state.ivBallWorldPosition<<" is not ok (says OneStepKick)!"<<endl;
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
// NeuroKick05::decideForAnAction()
//============================================================================
bool
NeuroKick05::decideForAnAction(Cmd & cmd)
{
  NeuroKick05State currentState = this->getCurrentState();
  NeuroKick05State successorState, bestSuccessorState;
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
    //let us decide if the quality of the kick, which can be obtained via
    //a oneortwostepkick, is sufficient for us
    if (   fabs(resultingVelocityAfter1StepKick - ivTargetKickVelocity)
         < cvMaximalKickVelocityDeviation  )
    {
      cmd.cmd_body.clone( cmdForOneStep.cmd_body );
      //cout<<"NeuroKick05: MAKE ONE STEP KICK!"<<endl;
      return true;
    }
    if (   fabs(resultingVelocityAfter2StepKick - ivTargetKickVelocity)
         < 0.85 * cvMaximalKickVelocityDeviation  )
    {
      cmd.cmd_body.clone( cmdForTwoStep.cmd_body );
      //cout<<"NeuroKick05: MAKE TWO STEP KICK!"<<endl;
      return true;
    }
  }

  //cout<<"NeuroKick05: MAKE A REAL   N E U R O   KICK!"<<endl;

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
      //evaluate the successor state
      double successorStateValue = this->evaluateState( successorState );
      //if (fineTuning==0) {cout<<"NeuroKick05: ITERATION, aktuell: "<<*chosenAction<<" erbringt "<<successorStateValue;
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
    } //end of while
    if ( ! ivDoFineTuning || actionFound == false )
      break;
  } //end of for loop
  if (actionFound)
  {
    //cout<<"NeuroKick05: Chosen Action: "<<bestAction<<" with value="<<bestSuccessorStateValue<<endl;

    return cmd.cmd_body.clone( bestAction );
  }
  return false;
}

