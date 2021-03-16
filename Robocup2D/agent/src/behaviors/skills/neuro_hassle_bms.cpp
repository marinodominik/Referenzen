#include "neuro_hassle_bms.h"

#if 0
#define   TGLOGnhPOL(YYY,XXX)        LOG_POL(YYY,XXX)
#else
#define   TGLOGnhPOL(YYY,XXX)
//#define   TGLOGnhPOL(YYY,XXX)        LOG_POL(5,XXX)
#endif

#define IDENTIFIER_SUCCESS  1
#define IDENTIFIER_ABORT    0
#define IDENTIFIER_FAILURE -1
#define IDENTIFIER_REAL_FAILURE -2

#define OP_MODE_LEARN 0
#define OP_MODE_EXPLOIT 1

#define TOP_SHARE_REMAIN 0.2
#define BOTTOM_SHARE_REMAIN 0.15

//////////////////////////////////////////////////////////////////////////////
//==INNER CLASS Statistics====================================================
NeuroHassle::Statistics::Statistics()
{ 
  ivSequenceCounter = 0;
}
float 
NeuroHassle::Statistics::getGlidingWindowSuccessFailuresAborts(int winSize, 
                                                               int sfar)
{  
  if (ivSequenceCounter==0) return 0.0;
  int counter = 0,
      sfarCounter = 0;
  for (int i=0; 
       i < winSize && (ivSequenceCounter-1-i)>=0;
       i++ )
  {
    counter++;
    if (ivVSequenceSuccesses[ivSequenceCounter-1-i] == sfar) sfarCounter++;
  }
  return (float)sfarCounter/(float)counter;
}
float           
NeuroHassle::Statistics::getGlidingWindowSuccess(int winSize)
{  
  return getGlidingWindowSuccessFailuresAborts(winSize, IDENTIFIER_SUCCESS);//0==successes
}
float           
NeuroHassle::Statistics::getGlidingWindowFailures(int winSize)
{  
  return getGlidingWindowSuccessFailuresAborts(winSize, IDENTIFIER_FAILURE);//-1==failures
}
float           
NeuroHassle::Statistics::getGlidingWindowRealFailures(int winSize)
{  
  return getGlidingWindowSuccessFailuresAborts(winSize, IDENTIFIER_REAL_FAILURE);//-2==real failures
}
float           
NeuroHassle::Statistics::getGlidingWindowAborts(int winSize)
{  
  return getGlidingWindowSuccessFailuresAborts(winSize, IDENTIFIER_ABORT);//2==aborts
}
float           
NeuroHassle::Statistics::getGlidingWindowCosts(int winSize)
{
  if (ivSequenceCounter==0) return 0.0;
  int counter = 0;
  float costSum = 0.0;
  for (int i=0; 
       i < winSize && (ivSequenceCounter-1-i)>=0;
       i++ )
  {
    counter++;
    costSum += ivVSequenceCosts[ivSequenceCounter-1-i];
  }
  return (float)costSum/(float)counter;
}
float 
NeuroHassle::Statistics::getAverageSuccess()
{ return getGlidingWindowSuccess(ivSequenceCounter); }
float 
NeuroHassle::Statistics::getAverageFailures()
{ return getGlidingWindowFailures(ivSequenceCounter); }
float 
NeuroHassle::Statistics::getAverageRealFailures()
{ return getGlidingWindowRealFailures(ivSequenceCounter); }
float 
NeuroHassle::Statistics::getAverageAborts()
{ return getGlidingWindowAborts(ivSequenceCounter); }
float 
NeuroHassle::Statistics::getAverageCosts()
{ return getGlidingWindowCosts(ivSequenceCounter); }
void            
NeuroHassle::Statistics::addEntry(int success, float costs)
{
  ivSequenceCounter++;
  ivVSequenceSuccesses.push_back(success);
  ivVSequenceCosts.push_back(costs);
}
int 
NeuroHassle::Statistics::getSize()
{  return ivSequenceCounter; }
void
NeuroHassle::Statistics::writeOut(ostream &stream)
{  
  bool isFileHandle = false;
  if (    (&stream != &(std::cout))
       && (&stream != &(std::cerr))
       && (&stream != &(std::clog)) )
    isFileHandle = true;
  if (isFileHandle)
  {
    //Wir haben einen ofstream uebergeben bekommen.
    ofstream *localOutStream = (ofstream*) &stream;
    if ( localOutStream->is_open()==false)
    {
      if ( cvEvaluationMode )
        localOutStream->open(cvEvaluationProtocolFileName, 
                             ofstream::out | ofstream::app);
      else
        localOutStream->open(cvLearningProtocolFileName,
                             ofstream::out | ofstream::app);
    }
  }
  
  if (   cvEvaluationMode  )
  {
    char netName[100];
    sprintf(netName,"SZENARIOS/SZENARIO_%d_%d_%f.net",
                                     0,//NeuroHassle::cvOverallRepetitionCounter,
                                     NeuroHassle::cvLearnSequenceCounter,
                                     this->getAverageCosts());
    cvpNeuralNetwork->save_net(netName);
  }
  
  stream.precision(4);
  stream<<NeuroHassle::cvLearnSequenceCounter;
  stream<<"\t";
  stream<<this->getSize();
  stream<<"\t";
  stream<<this->getAverageSuccess();
  stream<<"\t";
  stream<<this->getAverageFailures();
  stream<<"\t";
  stream<<this->getAverageAborts();
  stream<<"\t";
  stream<<this->getAverageRealFailures();
  stream<<"\t";
  stream<<this->getAverageCosts();
  stream<<"\t";
  stream<<this->getGlidingWindowSuccess(100);
  stream<<"\t";
  stream<<this->getGlidingWindowFailures(100);
  stream<<"\t";
  stream<<this->getGlidingWindowAborts(100);
  stream<<"\t";
  stream<<this->getGlidingWindowRealFailures(100);
  stream<<"\t";
  stream<<this->getGlidingWindowCosts(100);
  stream<<"\t";
  stream<<this->getGlidingWindowSuccess(500);
  stream<<"\t";
  stream<<this->getGlidingWindowFailures(500);
  stream<<"\t";
  stream<<this->getGlidingWindowAborts(500);
  stream<<"\t";
  stream<<this->getGlidingWindowRealFailures(500);
  stream<<"\t";
  stream<<this->getGlidingWindowCosts(500);
  stream<<"\t";
  stream<<endl;
  stream<<flush;
  if (isFileHandle)
  {
    ofstream *localOutStream = (ofstream*) &stream;
    localOutStream->close();
  }
}
void
NeuroHassle::Statistics::clear()
{
  ivSequenceCounter = 0;  
  ivVSequenceSuccesses.erase( ivVSequenceSuccesses.begin(), 
                              ivVSequenceSuccesses.end() );
  ivVSequenceCosts.erase( ivVSequenceCosts.begin(), 
                          ivVSequenceCosts.end() );
}
//==end of inner class Statistics=============================================
//////////////////////////////////////////////////////////////////////////////


//////////////////////////////////////////////////////////////////////////////
//IMPLEMENTIERUNG DER INNEREN KLASSE Action
//////////////////////////////////////////////////////////////////////////////
#define ACTION_NO_ACTION_VALUE 100.0

//--Initialisierung von Klassenvariablen--------------------------------------

//..Schusskraftdiskretisierung................................................
const int 
NeuroHassle::Action::cvpDashPowerDiscretization[] 
//  = {100,70,50,35,25,15,10,5};
  = {-100, -50, 0, 25, 50, 100 };
vector<int> 
NeuroHassle::Action::cvVDashPowerDiscretization( 
       NeuroHassle::Action::cvpDashPowerDiscretization,
       NeuroHassle::Action::cvpDashPowerDiscretization 
         + (sizeof(NeuroHassle::Action::cvpDashPowerDiscretization) 
             / sizeof(int)) );
             
//..Schussrichtungdiskretisierung................................................
const int 
NeuroHassle::Action::cvpTurnAngleDiscretization[] 
//  = {-175,-170,-165,-160,-155,-150,-145,-140,-135,-130,-125,-120,-115,-110,-105,-100,-95,-90,-85,-80,-75,-70,-65,-60,-55,-50,-45,-40,-35,-30,-25,-20,-15,-10,-5,
//     0,5,10,15,20,25,30,35,40,45,50,55,60,65,70,75,80,85,90,95,100,105,110,115,120,125,130,135,140,145,150,155,160,165,170,175,180};
  = {-179, -150, -120, -90, -60, -30, 0, 30, 60, 90, 120, 150, 179};
vector<int> 
NeuroHassle::Action::cvVTurnAngleDiscretization( 
       NeuroHassle::Action::cvpTurnAngleDiscretization,
       NeuroHassle::Action::cvpTurnAngleDiscretization 
         + (sizeof(NeuroHassle::Action::cvpTurnAngleDiscretization) 
             / sizeof(int)) );
   
//--Default Constructor-------------------------------------------------------
NeuroHassle::Action::Action()
{
  ivType = NIL;
  ivDashPowerIndex = -1;
  ivRealDashPower  = 0.0;
  ivTurnAngleIndex = -1;
  ivRealTurnAngle  = 0.0;
  ivExplorationFlag = false;
  ivIsSuccessAction = false;
  ivValue = ACTION_NO_ACTION_VALUE;
}


 
//--incorporateMeIntoCommand()------------------------------------------------
void 
NeuroHassle::Action::incorporateMeIntoCommand(Cmd * cmd)
{
  switch (ivType)
  {
    case TURN: 
      {
          float value;
          if (ivTurnAngleIndex == -1)
            value = ivRealTurnAngle;
          else
            value = DEG2RAD( cvVTurnAngleDiscretization[ivTurnAngleIndex] );
          cmd->cmd_body.set_turn( value );
      }
      break;
    case DASH: 
      {
        float value;
        if (ivDashPowerIndex == -1)
          value = ivRealDashPower;
        else
          value = cvVDashPowerDiscretization[ivDashPowerIndex];
        cmd->cmd_body.set_dash( value );
      }
      break;
    case KICK:
      {
        exit(1);
      }
    case NIL:
      break;
  }
}

//--performMeOnThisState()----------------------------------------------------
void
NeuroHassle::Action::performMeOnThisState(
                                  NeuroHassle::State current, 
                                  NeuroHassle::State * successor,
                                  bool debug)
{
  //calculate my current state
  Vector old_my_pos( ANGLE( current.ivOpponentInterceptAngle) );
  old_my_pos.normalize( - current.ivOpponentInterceptDistance );   
  Vector old_my_vel( current.ivMyVelocityX,
                     current.ivMyVelocityY );
  ANGLE old_my_ang(  current.ivMyAngleToOpponentPosition
                   + current.ivOpponentInterceptAngle );
  Vector old_ball_pos( current.ivBallPositionX,
                       current.ivBallPositionY );
  Vector old_ball_vel( 0.0, 0.0 );
  
  Cmd testCmd;
  this->incorporateMeIntoCommand(&testCmd);
  
//TGLOGnhPOL(0,<<"NeuroHassle: performMeOnThisState: given state = "
//  <<current.toShortString());
//TGLOGnhPOL(0,<<"  NeuroHassle: performMeOnThisState: oldPos="<<old_my_pos
//  <<" oldVel="<<old_my_vel<<" oldANG="<<RAD2DEG(old_my_ang.get_value_0_p2PI())
//  <<" oldBallPos="<<old_ball_pos);
  
  Vector  new_my_pos;
  Vector  new_my_vel;
  ANGLE   new_my_ang;
  Vector  new_ball_pos;
  Vector  new_ball_vel;

  Tools::model_cmd_main
              (  old_my_pos, 
                 old_my_vel, 
                 old_my_ang, 
                 //old_my_stamina, 
                 old_ball_pos, 
                 old_ball_vel,
                 testCmd.cmd_body,
                 new_my_pos, 
                 new_my_vel,
                 new_my_ang, 
                 //new_my_stamina,
                 new_ball_pos, 
                 new_ball_vel, 
                 false); //do_random

//TGLOGnhPOL(0,<<"  NeuroHassle: performMeOnThisState: newPos="<<new_my_pos
//  <<" newVel="<<new_my_vel<<" newANG="<<RAD2DEG(new_my_ang.get_value_0_p2PI())
//  <<" newBallPos="<<new_ball_pos);

  successor->setMeAccordingToThisWorldInformation
                               (  
                                  0.0, //oppPosX
                                  0.0, //oppPosY
                                  current.ivOpponentAbsoluteVelocity,//oppVel
                                  ANGLE(current.ivOpponentBodyAngle), //oppANGLE
                                  new_my_pos.getX(),   //myPosX
                                  new_my_pos.getY(),   //myPosY
                                  new_my_vel.getX(),   //myVelX
                                  new_my_vel.getY(),   //myVelY
                                  new_my_ang,          //myANGLE
                                  new_ball_pos.getX(), //ballPosX
                                  new_ball_pos.getY(), //ballPosY
                                  new_ball_vel.getX(),
                                  new_ball_vel.getY()
                                );
  
}

//--setMeFromCommand()-----------------------------------------------------
void 
NeuroHassle::Action::setMeFromCommand(Cmd &cmd, bool neuro)
{
  switch (cmd.cmd_body.get_type())
  {
    case Cmd_Body::TYPE_TURN:
    {
      Angle a;
      cmd.cmd_body.get_turn(a);
      float degrees = RAD2DEG( a );
      while (degrees<-180.0) degrees+=180.0;
      while (degrees>180.0) degrees-=180.0;
      int bestIndex = 0;
      float bestDelta = fabs(cvVTurnAngleDiscretization[0] - degrees);
      for (unsigned int i=0; i<cvVTurnAngleDiscretization.size(); i++)
      {
        if ( fabs(cvVTurnAngleDiscretization[i] - degrees ) < bestDelta )
        {
          bestDelta = fabs(cvVTurnAngleDiscretization[i] - degrees );
          bestIndex = i;
        }
      }
      this->ivTurnAngleIndex = bestIndex;
      this->ivType = TURN;
      if (bestDelta > 0.1)
      {
        this->ivRealTurnAngle = degrees;
        this->ivTurnAngleIndex = -1;
      }
      break;
    }
    case Cmd_Body::TYPE_DASH:
    {
      double power;
      cmd.cmd_body.get_dash(power);
      int bestIndex = 0;
      float bestDelta = fabs(cvVDashPowerDiscretization[0] - power);
      for (unsigned int i=0; i<cvVTurnAngleDiscretization.size(); i++)
      {
        if ( fabs(cvVDashPowerDiscretization[i] - power ) < bestDelta )
        {
          bestDelta = fabs(cvVDashPowerDiscretization[i] - power );
          bestIndex = i;
        }
      }
      this->ivDashPowerIndex = bestIndex;
      this->ivType = DASH;
      if (bestDelta > 0.1)
      {
        this->ivRealDashPower = power;
        this->ivDashPowerIndex = -1;
      }
      break;
    }
  }
}

//--setMeRandomly()--------------------------------------------------------
void
NeuroHassle::Action::setMeRandomly()
{
  int numOfActions = cvVTurnAngleDiscretization.size()
                     + cvVDashPowerDiscretization.size();
    float r = (int)((float)numOfActions * ((float)rand() / (float)RAND_MAX));
    if (r < cvVTurnAngleDiscretization.size() )
    {
      this->ivType = TURN;
      this->ivTurnAngleIndex = rand()%cvVTurnAngleDiscretization.size();
    }
    else
    {
      this->ivType = DASH;
      this->ivDashPowerIndex = rand()%cvVDashPowerDiscretization.size();
    }
    this->ivValue = ACTION_NO_ACTION_VALUE;
    this->ivIsSuccessAction = false;
}
        
//--setMeFromAction()---------------------------------------------------------
void       
NeuroHassle::Action::setMeFromAction( NeuroHassle::Action a )
{
    ivType = a.ivType;
    ivDashPowerIndex = a.ivDashPowerIndex;
    ivTurnAngleIndex = a.ivTurnAngleIndex;
    ivExplorationFlag = a.ivExplorationFlag;
    ivIsSuccessAction = a.ivIsSuccessAction;
    ivValue = a.ivValue;
}
              
//--setNullAction()---------------------------------------------------------
void       
NeuroHassle::Action::setNullAction()
{
    ivType = TURN;
    ivDashPowerIndex = 0;
    ivTurnAngleIndex = 0;
    ivExplorationFlag = false;
    ivIsSuccessAction = false;
    ivValue = ACTION_NO_ACTION_VALUE;
}
              
//--getDashPower()------------------------------------------------------------
int
NeuroHassle::Action::getDashPower()
{
  return cvVDashPowerDiscretization[ivDashPowerIndex];
}

//--getTurnAngle()------------------------------------------------------------
int
NeuroHassle::Action::getTurnAngle()
{
  return cvVTurnAngleDiscretization[ivTurnAngleIndex];
}

//--setTurnAngle()------------------------------------------------------------
void 
NeuroHassle::Action::setTurnAngle(float a)
{
  float minDelta = 360.0;
  int minIndex = 0;
  for (unsigned int i=0; i<cvVTurnAngleDiscretization.size(); i++)
  {
    if ( fabs(a - cvVTurnAngleDiscretization[i]) < minDelta)
    {
      minDelta = fabs(a - cvVTurnAngleDiscretization[i]);
      minIndex = i;
    }
  }
  ivTurnAngleIndex = minIndex;
}

//--setExplorationFlag()------------------------------------------------------
void
NeuroHassle::Action::setExplorationFlag(bool flag)
{
  ivExplorationFlag = flag;
}

//--getExplorationFlag()------------------------------------------------------
bool
NeuroHassle::Action::getExplorationFlag()
{
  return ivExplorationFlag;
}

//--setIsSuccessAction()------------------------------------------------------
void 
NeuroHassle::Action::setIsSuccessAction(bool flag)
{
  ivIsSuccessAction = flag;
}

//--getIsSuccessAction()------------------------------------------------------
bool
NeuroHassle::Action::getIsSuccessAction()
{
  return ivIsSuccessAction;
}

//--setType()-----------------------------------------------------------------       
void 
NeuroHassle::Action::setType(LearnActionTypes type)
{
  this->ivType = type;
}
//--getType()-----------------------------------------------------------------       
NeuroHassle::Action::LearnActionTypes
NeuroHassle::Action::getType()
{
  return this->ivType;
}

//--setValue()----------------------------------------------------------------
void
NeuroHassle::Action::setValue(float aValue)
{
  ivValue = aValue;
}

//--getValue()----------------------------------------------------------------
float
NeuroHassle::Action::getValue()
{
  return ivValue;
}

//--toString()----------------------------------------------------------------
string
NeuroHassle::Action::toString()
{
  string returnValue;
  char dummy[100];
  returnValue += "NeuroHassle::Action: ";
  switch (ivType)
  {
    case TURN: 
      returnValue += "TURN ( "; 
      sprintf(dummy, "%d", this->getTurnAngle() );
      returnValue += dummy;
      break;
    case DASH:
      returnValue += "DASH ( ";
      sprintf(dummy, "%d", this->getDashPower() );
      returnValue += dummy;
      break;
    case KICK:
      returnValue += "KICK (ERROR)";
      returnValue += dummy;
      break;
    case NIL:
      returnValue += "NIL (ERROR)";
      returnValue += dummy;
      break;
  }
    returnValue += " ) --- V=";
    sprintf(dummy, "%f", this->getValue() );
    returnValue += dummy;
    if (ivExplorationFlag==true) returnValue+=" [EXPLO]";
    if (ivIsSuccessAction)
      sprintf(dummy, "%s", " [SUCCESS]");
    else 
      sprintf(dummy, "%s", "");
    returnValue += dummy;
  return returnValue; 
}

//Ende der inneren Klasse Action//////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////

//////////////////////////////////////////////////////////////////////////////
// Beginn der inneren Klasse State ///////////////////////////////////////////

//--Konstruktor---------------------------------------------------------------
NeuroHassle::State::State()
{
        ivOpponentInterceptAngle = 0.0; 
        ivOpponentInterceptDistance = 0.0;
        ivMyVelocityX = 0.0;
        ivMyVelocityY = 0.0;
        ivMyAngleToOpponentPosition = 0.0;
        ivBallPositionX = 0.0;
        ivBallPositionY = 0.0;
        ivOpponentBodyAngle = 0.0;
        ivOpponentAbsoluteVelocity = 0.0;
        
        ivRealBallVelocityX = 0.0;
        ivRealBallVelocityY = 0.0;
        ivMyPositionX = 0.0;
        ivMyPositionY = 0.0;
        ivOpponentHeadStart = 0.0;
}


//--Copy Constructor----------------------------------------------------------
NeuroHassle::State::State(const State & s)
{
        ivOpponentInterceptAngle = s.ivOpponentInterceptAngle;
        ivOpponentInterceptDistance = s.ivOpponentInterceptDistance;
        ivMyVelocityX = s.ivMyVelocityX;
        ivMyVelocityY = s.ivMyVelocityY;
        ivMyAngleToOpponentPosition = s.ivMyAngleToOpponentPosition;
        ivBallPositionX = s.ivBallPositionX;
        ivBallPositionY = s.ivBallPositionY;
        ivOpponentBodyAngle = s.ivOpponentBodyAngle;
        ivOpponentAbsoluteVelocity = s.ivOpponentAbsoluteVelocity;
        
        ivRealBallVelocityX = s.ivRealBallVelocityX;
        ivRealBallVelocityY = s.ivRealBallVelocityY;
        ivMyPositionX = s.ivMyPositionX;
        ivMyPositionY = s.ivMyPositionY;
        ivOpponentHeadStart = s.ivOpponentHeadStart;
}

//--setMeAccordingToThisWorldInformation()------------------------------------
void
NeuroHassle::State::setMeAccordingToThisWorldInformation
                    ( 
                      float  oppPosX,
                      float  oppPosY,
                      float  oppVelAbs,
                      ANGLE  oppANGLE,
                      float  myPosX,
                      float  myPosY,
                      float  myVelX,
                      float  myVelY,
                      ANGLE  myANGLE,
                      float  ballPosX,
                      float  ballPosY,
                      float  ballVelX,
                      float  ballVelY
                    )
{
  ANGLE opponentAngleToMyGoal( oppPosX + FIELD_BORDER_X,
                               oppPosY );
  
  //consider opponent
  ANGLE myAngleToOpponent( oppPosX - myPosX,
                           oppPosY - myPosY );
  ivOpponentInterceptAngle = (myAngleToOpponent - opponentAngleToMyGoal)
                             .get_value_mPI_pPI(); 
  
  ivOpponentInterceptDistance = sqrt( (myPosX-oppPosX)*(myPosX-oppPosX)
                                     +(myPosY-oppPosY)*(myPosY-oppPosY) );
                                     
  ivOpponentAbsoluteVelocity = oppVelAbs;

  //consider me
  Vector myVel( myVelX, myVelY );
  myVel.rotate( - opponentAngleToMyGoal.get_value_mPI_pPI() );
  
  ivMyVelocityX = myVel.getX();
  ivMyVelocityY = myVel.getY();

  ANGLE myAngleToOppFromMyPerspective( oppPosX - myPosX,
                                       oppPosY - myPosY );
  ivMyAngleToOpponentPosition = (myANGLE - myAngleToOppFromMyPerspective)
                                .get_value_mPI_pPI();

  //consider ball
  Vector ballPos( ballPosX - oppPosX, ballPosY - oppPosY );
  ballPos.rotate( - opponentAngleToMyGoal.get_value_mPI_pPI() );
  ivBallPositionX = ballPos.getX();
  ivBallPositionY = ballPos.getY();
  
  //consider opponent angle
  ANGLE oppANGLEInHisCoordinateSystem( oppANGLE - opponentAngleToMyGoal );
  ivOpponentBodyAngle = oppANGLEInHisCoordinateSystem.get_value_mPI_pPI(); 
  
  //ball velocity => not used for learning
  ivRealBallVelocityX = ballVelX;
  ivRealBallVelocityY = ballVelY;
  //my position
  Vector myPos( myPosX - oppPosX, myPosY - oppPosY);
  myPos.rotate( - opponentAngleToMyGoal.get_value_mPI_pPI() );
  ivMyPositionX = myPos.getX();
  ivMyPositionY = myPos.getY();
  
  
  //debug
  /*
  TGLOGnhPOL(0,<<"NeuroHassle: setMeForWorldInfos: oppPos="<<oppPosX<<","
    <<oppPosY<<" oppANG="
    <<RAD2DEG(oppANGLE.get_value_mPI_pPI())<<" myPos="<<myPosX<<","<<myPosY
    <<" myVel="<<myVelX<<","<<myVelY<<" myANG="<<RAD2DEG(myANGLE.get_value_mPI_pPI())
    <<" ballPos="<<ballPosX<<","<<ballPosY);
  TGLOGnhPOL(0,"NeuroHassle: setMeForWorldInfos: INFO: opponentAngleToMyGoal="
    <<RAD2DEG(opponentAngleToMyGoal.get_value_mPI_pPI())
    <<" myAngleToOpponent="<<RAD2DEG(myAngleToOpponent.get_value_mPI_pPI()));
  TGLOGnhPOL(0,<<"NeuroHassle: setMeForWorldInfos: RESULT: oppIcptAng="
    <<RAD2DEG(ivOpponentInterceptAngle)
    <<" oppIcptDist="<<ivOpponentInterceptDistance<<" myVel="
    <<ivMyVelocityX<<","<<ivMyVelocityY<<" myAngToOppPos=["
    <<ivMyAngleToOpponentPosition<<"|"<<RAD2DEG(ivMyAngleToOpponentPosition)
    <<"] ballPos="<<ivBallPositionX
    <<","<<ivBallPositionY);
  TGLOGnhPOL(0,"NeuroHassle: setMeForWorldInfos: RESULT: "<<this->toShortString());
  */
  
}

//--setMeFromAnotherState()---------------------------------------------------
void
NeuroHassle::State::setMeFromAnotherState(const State & s )
{
        ivOpponentInterceptAngle = s.ivOpponentInterceptAngle;
        ivOpponentInterceptDistance = s.ivOpponentInterceptDistance;
        ivMyVelocityX = s.ivMyVelocityX;
        ivMyVelocityY = s.ivMyVelocityY;
        ivMyAngleToOpponentPosition = s.ivMyAngleToOpponentPosition;
        ivBallPositionX = s.ivBallPositionX;
        ivBallPositionY = s.ivBallPositionY;
        ivOpponentBodyAngle = s.ivOpponentBodyAngle;
        ivOpponentAbsoluteVelocity = s.ivOpponentAbsoluteVelocity;
        
        ivRealBallVelocityX = s.ivRealBallVelocityX;
        ivRealBallVelocityY = s.ivRealBallVelocityY;
        ivMyPositionX = s.ivMyPositionX;
        ivMyPositionY = s.ivMyPositionY;
        ivOpponentHeadStart = s.ivOpponentHeadStart;
}

//--toShortString()-----------------------------------------------------------
string
NeuroHassle::State::toShortString()
{
    char dummy[100];
    string returnValue;
    returnValue += "State(";
    sprintf(dummy,"##[%.2f|%.2f], ", ivOpponentInterceptAngle, 
                                   RAD2DEG(ivOpponentInterceptAngle));
    returnValue += dummy;
    sprintf(dummy,"%.2f## ", ivOpponentInterceptDistance);
    returnValue += dummy;
    sprintf(dummy,"%.2f, ", ivMyPositionX);
    returnValue += dummy;
    sprintf(dummy,"%.2f, ", ivMyPositionY);
    returnValue += dummy;
    sprintf(dummy,"%.2f, ", ivMyVelocityX);
    returnValue += dummy;
    sprintf(dummy,"%.2f, ", ivMyVelocityY);
    returnValue += dummy;
    sprintf(dummy,"[%.2f|%.2f], ", ivMyAngleToOpponentPosition, 
                                   RAD2DEG(ivMyAngleToOpponentPosition));
    returnValue += dummy;
    sprintf(dummy,"%.2f, ", ivBallPositionX);
    returnValue += dummy;
    sprintf(dummy,"%.2f,", ivBallPositionY);
    returnValue += dummy;
    sprintf(dummy,"[%.2f|%.2f],", ivOpponentBodyAngle,
                                  RAD2DEG(ivOpponentBodyAngle) );
    returnValue += dummy;
    sprintf(dummy,"%.2f)", ivOpponentAbsoluteVelocity);
    returnValue += dummy;
    return returnValue;
}



//Ende der inneren Klasse State //////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////


//////////////////////////////////////////////////////////////////////////////
// Beginn der Klasse NeuroHassle /////////////////////////////////////////////
int              NeuroHassle::cvOperationMode = OP_MODE_EXPLOIT;
bool             NeuroHassle::cvEvaluationMode = false;
float            NeuroHassle::cvRewardForIntermediateState = 0.0;
Net            * NeuroHassle::cvpNeuralNetwork = NULL;
char             NeuroHassle::cvNeuralNetworkFilename[IDENTIFIER_LENGTH];
int              NeuroHassle::cvNumberOfTrainingExamplesForNetworkTraining=500;
NeuroHassle::Statistics NeuroHassle::cvLearnStatistics;
NeuroHassle::Statistics NeuroHassle::cvEvaluationStatistics;
char            NeuroHassle::cvLearningProtocolFileName[IDENTIFIER_LENGTH];
ofstream        NeuroHassle::cvLearningProtocolFileHandle;
char            NeuroHassle::cvEvaluationProtocolFileName[IDENTIFIER_LENGTH];
ofstream        NeuroHassle::cvEvaluationProtocolFileHandle;
int             NeuroHassle::cvLearnSequenceCounter = 0;
int             NeuroHassle::cvFlushLearnStatisticsAfterAsManySequences = 100;
float           NeuroHassle::cvBestTimeAborts = 1.0;

//--Konstruktor---------------------------------------------------------------
NeuroHassle::NeuroHassle()
{
  ivDiscountFactor = 0.9;
  ivStarted = false;
  ivUnExploitedEpisodeCounter = 0;
  ivTrainNeuralNetworkCounter = 0;
  ivEpisodeInitialGoalDist = 50.0;
  ivHaveCollectedEnoughTrainingExamplesForNextTrainingCycle = false;
  if (WSinfo::ball)
    ivBallVector = WSinfo::ball->pos;
  else 
    ivBallVector = Vector(0.0,0.0);
  initNeuralNetwork();
}

//--destruktor----------------------------------------------------------------
NeuroHassle::~NeuroHassle()
{
}

//--analyseCompletedEvaluationSequence()------------------------------------
void
NeuroHassle::analyseCompletedEvaluationSequence()
{
    if (ivVStateSequence.size() <= 1) return;
    for ( vector<State>::iterator i = ivVStateSequence.begin();
          i != ivVStateSequence.end();
          i++ )
    {
      //Schauen wir mal, ob im aktuellen Zustand (*i) bereits der
      //Ball kickbar war.
      if ( this->classifyStateAsSuccessOrFailure( (*i), false ) == true)
      {
        //Ist das aktuelle Element noch _nicht_ das Ende der Sequenz?
        if ( (i+1) != ivVStateSequence.end() )
        {
          //Alles ab Element Nachfolgerelement i+1 wird gel?scht!   
          ivVStateSequence.erase( i+1 );
          break; //Abbruch der for-Schleife;
        }
      }
    }
    State finalState =  ivVStateSequence[ ivVStateSequence.size() - 1 ];
    //bool success =
      this->classifyStateAsSuccessOrFailure( finalState, 
                                                          false );
//    this->handleLearningStatistics(success);
}

//--classifySequenceAsSuccessOrFailure()------------------------------------
int
NeuroHassle::classifySequenceAsSuccessOrFailure()
{
    State finalState =  ivVStateSequence[ ivVStateSequence.size() - 1 ];
    int success = this->classifyStateAsSuccessOrFailure( finalState, 
                                                         false );
                                                         
    //consider head startings
    if ( success >= 0 )
    {
      int oppHeadStartCounter = 0;
      float previousOppHeadStart = 10.0;
      for (unsigned int i=0; i<ivVStateSequence.size(); i++)
      {
        State & s = ivVStateSequence[i];
        Vector ballPos(s.ivBallPositionX,s.ivBallPositionY);
        if (
               s.ivOpponentHeadStart >= 1.0
            && ballPos.norm() < 2.0 
            && s.ivOpponentHeadStart > previousOppHeadStart 
           )
        {
          TGLOGnhPOL(0,<<"NeuroHassle: warning head start at "<<i<<" is: "<<s.ivOpponentHeadStart);
          oppHeadStartCounter ++ ;
        }
        else
          oppHeadStartCounter = 0;
        if (oppHeadStartCounter > 0)
        {
          TGLOGnhPOL(0,<<"NeuroHassle: HEAD START FAILURE SEQUENCE: erase from"
            <<i<<" to "<<ivVStateSequence.size()<<"!");
          ivVStateSequence.erase( ivVStateSequence.begin() + i,
                                  ivVStateSequence.end() );
          success = IDENTIFIER_FAILURE;
          break;
        }
        previousOppHeadStart = s.ivOpponentHeadStart;
      }
    }
                                                         
    this->handleLearningStatistics(success);
    //this->showResultsForCompletedSequence(success);
    return success;
}

//--classifyStateAsSuccessOrFailure()-----------------------------------------
int    
NeuroHassle::classifyStateAsSuccessOrFailure(State s, 
                                             bool debug )
{
  int successIndicator = IDENTIFIER_FAILURE;
  
  Vector ballPos( s.ivBallPositionX, s.ivBallPositionY );
  Vector myPos( ANGLE( PI + s.ivOpponentInterceptAngle ) );
  myPos.normalize( s.ivOpponentInterceptDistance );   
  
  if (      ballPos.distance( myPos )
          < WSinfo::me->kick_radius * 1.2 )
  {
    TGLOGnhPOL(0,<<"NeuroHassle: classifyStateAsSuccessOrFailure: SUCCESS STATE, because ball in kickrange ("
      <<s.toShortString()<<").");
    return IDENTIFIER_SUCCESS;
  }
  else
  {
    TGLOGnhPOL(0,<<"NeuroHassle: classifyStateAsSuccessOrFailure ball NOT in kickrange ("
      <<s.toShortString()<<"). Considered myPos="<<myPos
      <<", considered ballPos="<<ballPos);
  }
  
  //my angle in the coordinate system with the opponent at its center
  //turned by its angle to my goal center
  ANGLE myTackleAngle(   s.ivOpponentInterceptAngle
                       + s.ivMyAngleToOpponentPosition );
  float prob 
    = Tools::get_tackle_success_probability
                   ( myPos,
                     ballPos,
                     myTackleAngle.get_value_mPI_pPI() );
  if (       prob
          >= REQUIRED_TACKLE_SUCCESS_PROBABILITY_FOR_SUCCESS  )
  {
    TGLOGnhPOL(0,<<"NeuroHassle: SUCCESS STATE, because ball tackleable "
      <<"with probability "<<prob<<" ("
      <<s.toShortString()<<"). Considered myPos="<<myPos
      <<" considered ballPos="<<ballPos
      <<" considered myTackleAngle = "<<RAD2DEG(myTackleAngle.get_value_0_p2PI()));
    return IDENTIFIER_SUCCESS;
  }
  else
  {
    TGLOGnhPOL(0,<<"NeuroHassle: Tackle success probability is low: "
      <<prob<<" ("
      <<s.toShortString()<<"). Considered myPos="<<myPos
      <<" considered ballPos="<<ballPos
      <<" considered myTackleAngle = "<<RAD2DEG(myTackleAngle.get_value_0_p2PI()));
  }

  PlayerSet opps = WSinfo::alive_opponents;
  opps.keep_and_sort_closest_players_to_point(1, WSinfo::me->pos);

  if (    ivBallVector.distance( MY_GOAL_CENTER ) < 10.0 
       || ivBallVector.distance( WSinfo::me->pos) > 5.0 )
  {
    successIndicator = IDENTIFIER_REAL_FAILURE;
    if ( opps.num > 0 )
    {
      //panick kick by opponent
     int stateIndex = (int)ivVStateSequence.size() - 1;
     while (stateIndex >= 0)
     {
       Vector ballPos( ivVStateSequence[stateIndex].ivBallPositionX,
                       ivVStateSequence[stateIndex].ivBallPositionY );
       if (ballPos.norm() < 1.0)
         break;
       stateIndex -- ;
     }
     TGLOGnhPOL(0,<<"NeuroHassle: Panick Kick state indes ="<<stateIndex);
     State & panickKickState = ivVStateSequence[stateIndex];
     if (   (
                panickKickState.ivOpponentInterceptDistance < 1.5
             || (    panickKickState.ivOpponentInterceptDistance < 3.0
                  && panickKickState.ivMyPositionX < 0.0 )
            )
         && opps[0]->pos.distance(MY_GOAL_CENTER) > 0.6*ivEpisodeInitialGoalDist)
        successIndicator = IDENTIFIER_ABORT;
    }
    TGLOGnhPOL(0,<<"NeuroHassle: classifyStateAsSuccessOrFailure: "
      <<successIndicator);
    return successIndicator;    
  }
  
  if ( opps.num > 0 )
    if (  opps[0]->pos.distance(MY_GOAL_CENTER)
        < WSinfo::me->pos.distance(MY_GOAL_CENTER) - 1.0 )
    {
      TGLOGnhPOL(0,<<"NeuroHassle: classifyStateAsSuccessOrFailure: "
        <<IDENTIFIER_REAL_FAILURE);
      return IDENTIFIER_REAL_FAILURE;
    }
  
  return successIndicator;
}

//-decideForAnAction()--------------------------------------------------------
void
NeuroHassle::decideForAnAction(Action *chosenAction)
{
  State  successorState;
  Action currentAction;
  float maxValueOfAState = -INT_MAX;
  bool actionFound = false;
  
  //[A] Betrachtung der DASH-Aktionen
  currentAction.setType(NeuroHassle::Action::DASH);  
  for ( unsigned int i=0; i<Action::cvVDashPowerDiscretization.size(); i++)
  {
    currentAction.ivDashPowerIndex = i;
//TGLOGnhPOL(0,"NeuroHassle: decideForAnAction: consider "<<currentAction.toString());
    currentAction.performMeOnThisState( ivCurrentState, 
                                        &successorState, 
                                        false);
    float valueOfSuccessorState
      =    cvRewardForIntermediateState
        +   ivDiscountFactor
          * getValueFromFunctionApproximator(successorState, NULL);
//TGLOGnhPOL(0,<<"NeuroHassle: succ state for action "<<currentAction.toString()
//  <<": "<<successorState.toShortString()<<"  ==> V = "<<valueOfSuccessorState);
//TGLOGnhPOL(0,<<"--------------------------------------------------------");
    if (valueOfSuccessorState >= maxValueOfAState)
    {
      maxValueOfAState = valueOfSuccessorState;
      chosenAction->setMeFromAction( currentAction );
      actionFound = true;
    }
  }
  //[B] Betrachtung der TURN-Aktion
  currentAction.setType(NeuroHassle::Action::TURN);
  for ( unsigned int i=0; i<Action::cvVTurnAngleDiscretization.size(); i++)
  { 
    currentAction.ivTurnAngleIndex = i;
//TGLOGnhPOL(0,"NeuroHassle: decideForAnAction: consider "<<currentAction.toString());
    currentAction.performMeOnThisState( ivCurrentState, 
                                        &successorState, 
                                        false);
    float valueOfSuccessorState
      =   cvRewardForIntermediateState
        +   ivDiscountFactor
          * getValueFromFunctionApproximator(successorState, NULL);
//TGLOGnhPOL(0,<<"NeuroHassle: succ state for action "<<currentAction.toString()
//  <<": "<<successorState.toShortString()<<"  ==> V = "<<valueOfSuccessorState);
//TGLOGnhPOL(0,<<"--------------------------------------------------------");
    if (valueOfSuccessorState >= maxValueOfAState)
    {
      maxValueOfAState = valueOfSuccessorState;
      chosenAction->setMeFromAction( currentAction );
      actionFound = true;
    }
  }
  //avoid too many backwards dashs
  if (    chosenAction->getType() == NeuroHassle::Action::DASH
       && chosenAction->getDashPower() < 0 )
  {
    TGLOGnhPOL(0,<<"NeuroHassle: Best action is negative dash.");
    if (    WSinfo::me->pos.distance( ivBallVector ) > 3.0
         || (      ivBallVector.distance( MY_GOAL_CENTER )
                 < WSinfo::me->pos.distance( MY_GOAL_CENTER ) 
              && WSinfo::me->pos.distance( ivBallVector ) > 2.0 )
       )
    {
      TGLOGnhPOL(0,<<"NeuroHassle: I am overwriting the -dash-action.");
      chosenAction->setType(NeuroHassle::Action::TURN);
      chosenAction->setTurnAngle(180.0);
    }
  }
  
  //Ende
  if (actionFound == false)
  {
    cout<<"No action found. Abort."<<endl;
    exit(3);
  }
}

//--determineCurrentStateFromWorldModel()-------------------------------------
bool
NeuroHassle::determineCurrentStateFromWorldModel()
{
  if ( ivpHassleOpponent == NULL )
  {
    TGLOGnhPOL(0,<<"NeuroHassle: Cannot determine current state from world "
      <<"model because of a NULL opponent.");
    return false;
  }
    
  ivCurrentState.setMeAccordingToThisWorldInformation
                 (  
                    ivpHassleOpponent->pos.getX(),
                    ivpHassleOpponent->pos.getY(),
                    ivpHassleOpponent->vel.norm(),
                    ivpHassleOpponent->ang,
                    WSinfo::me->pos.getX(),
                    WSinfo::me->pos.getY(),
                    WSinfo::me->vel.getX(),
                    WSinfo::me->vel.getY(),
                    WSinfo::me->ang,
                    ivBallVector.getX(),
                    ivBallVector.getY(),
                    WSinfo::ball->vel.getX(),
                    WSinfo::ball->vel.getY()
                 );
  ivCurrentState.ivOpponentHeadStart
    =   WSinfo::me->pos.distance( MY_GOAL_CENTER )
      - ivpHassleOpponent->pos.distance( MY_GOAL_CENTER );
  return true;
}


//--eraseTrainingExamples()---------------------------------------------------
void 
NeuroHassle::eraseTrainingExamples(bool remainTopTrainingExamples)
{
  int numOfTrainingExamples=0, numberOfTopExamples=0, numberOfBottomExamples=0;
  float shareOfTopExamplesToRemain = TOP_SHARE_REMAIN,
        shareOfBottomExamplesToRemain = BOTTOM_SHARE_REMAIN;
        
  if (remainTopTrainingExamples)
  {
    numOfTrainingExamples = ivVCurrentNeuralNetTrainingExamples.size();
    numberOfTopExamples   
      =   ivVTopNeuralNetTrainingExamples.size()
        + (int)(shareOfTopExamplesToRemain * (float)(numOfTrainingExamples));
    TGLOGnhPOL(0,<<"LOESCHUNG der TEs, behalte aber die besten "<<numberOfTopExamples);
    if (numberOfTopExamples > 0)
      for (int i=0; i<numOfTrainingExamples; i++)
      {
        if ((int)ivVTopNeuralNetTrainingExamples.size() < numberOfTopExamples)
          ivVTopNeuralNetTrainingExamples.push_back( ivVCurrentNeuralNetTrainingExamples[i] );
        else
        {
          int worstTopIndex = 0;
          float worstTop = ivVTopNeuralNetTrainingExamples[ worstTopIndex ].second;
          for (int t=0; t<numberOfTopExamples; t++)
          {
            if ( worstTop > ivVTopNeuralNetTrainingExamples[ t ].second )
            { 
              worstTop = ivVTopNeuralNetTrainingExamples[ t ].second;
              worstTopIndex = t;
            }
          }
          bool shallIAddATrainingExample = ivVCurrentNeuralNetTrainingExamples[i].second >= worstTop;
          //bool shallIAddATrainingExample = ((float)rand() / (float)RAND_MAX) < 0.5 ? true : false;
          if ( shallIAddATrainingExample )
          {
            ivVTopNeuralNetTrainingExamples.erase( 
              ivVTopNeuralNetTrainingExamples.begin() + worstTopIndex,
              ivVTopNeuralNetTrainingExamples.begin() + worstTopIndex + 1 );
            ivVTopNeuralNetTrainingExamples.push_back( ivVCurrentNeuralNetTrainingExamples[i] );
          }
        }
      }
    numberOfBottomExamples   
      =   ivVBottomNeuralNetTrainingExamples.size()
        + (int)(shareOfBottomExamplesToRemain * (float)(numOfTrainingExamples));
    if (numberOfBottomExamples > 0)
      for (int i=0; i<numOfTrainingExamples; i++)
      {
        if ((int)ivVBottomNeuralNetTrainingExamples.size() < numberOfBottomExamples)
          ivVBottomNeuralNetTrainingExamples.push_back( ivVCurrentNeuralNetTrainingExamples[i] );
        else
        {
          int bestBottomIndex = 0;
          float bestBottom = ivVBottomNeuralNetTrainingExamples[ bestBottomIndex ].second;
          for (int t=0; t<numberOfBottomExamples; t++)
          {
            if ( bestBottom < ivVBottomNeuralNetTrainingExamples[ t ].second )
            { 
              bestBottom = ivVBottomNeuralNetTrainingExamples[ t ].second;
              bestBottomIndex = t;
            }
          }
          bool shallIAddATrainingExample = ivVCurrentNeuralNetTrainingExamples[i].second <= bestBottom;
          //bool shallIAddATrainingExample = ((float)rand() / (float)RAND_MAX) < 0.5 ? true : false;
          if ( shallIAddATrainingExample )
          {
            ivVBottomNeuralNetTrainingExamples.erase( 
              ivVBottomNeuralNetTrainingExamples.begin() + bestBottomIndex,
              ivVBottomNeuralNetTrainingExamples.begin() + bestBottomIndex + 1 );
            ivVBottomNeuralNetTrainingExamples.push_back( ivVCurrentNeuralNetTrainingExamples[i] );
          }
        }
      }
  }
  
  ivVCurrentNeuralNetTrainingExamples.erase( ivVCurrentNeuralNetTrainingExamples.begin(),
                                             ivVCurrentNeuralNetTrainingExamples.end() );
  ivVNeuralNetTrainingExamples.erase( ivVNeuralNetTrainingExamples.begin(),
                                      ivVNeuralNetTrainingExamples.end() );
  ivHaveCollectedEnoughTrainingExamplesForNextTrainingCycle = false;   
    
  if (remainTopTrainingExamples)
  { 
    cout<<"Behalte: ";
    for (int i=0; i<numberOfTopExamples; i++)
    { 
      cout<<ivVTopNeuralNetTrainingExamples[i].second<<" ";
      //ivVNeuralNetTrainingExamples.push_back( ivVTopNeuralNetTrainingExamples[i] ); 
    }
    for (int i=0; i<numberOfBottomExamples; i++)
    { 
      cout<<ivVBottomNeuralNetTrainingExamples[i].second<<" ";
      //ivVNeuralNetTrainingExamples.push_back( ivVBottomNeuralNetTrainingExamples[i] ); 
    }
    cout<<endl;
  }
}

//--get_cmd()-----------------------------------------------------------------
bool
NeuroHassle::get_cmd_exploit(Cmd & cmd)
{
  //determine error conditions
  if (WSinfo::ws->play_mode != PM_PlayOn)
  {
    TGLOGnhPOL(0,<<"NeuroHassle: ERROR: Behavior should be invoked only"
      <<"during play-on situations!");
    return false;
  }
  if (this->determineCurrentStateFromWorldModel() == false)
  {
    TGLOGnhPOL(0,<<"NeuroHassle: ERROR: CANNOT DETERMINE WORLD STATE!");
    return false;
  }
  if (ivpHassleOpponent == NULL)
  {
    TGLOGnhPOL(0,<<"NeuroHassle: ERROR: No hassle opponent has been set!");
    return false;
  }
  //taking an action 
  Action chosenAction;
  this->decideForAnAction(&chosenAction);
  chosenAction.incorporateMeIntoCommand( &cmd );
  TGLOGnhPOL(0,<<"NeuroHassle: Obtained a NEURO HASSLE command! (bpDist="
    <<WSinfo::me->pos.distance(ivpHassleOpponent->pos)<<"): "<<cmd.cmd_body);
  return true;
}

//--get_cmd()-----------------------------------------------------------------
bool
NeuroHassle::get_cmd(Cmd & cmd)
{
  if (cvOperationMode == OP_MODE_EXPLOIT)
    return get_cmd_exploit( cmd );
  
  if (WSinfo::ws->time == ivLastTimeInvoked)
    return false;
  ivLastTimeInvoked = WSinfo::ws->time;
    
  float beginTimeInMilliSeconds = Tools::get_current_ms_time();
  
  //Gucken wir uns erst mal an, was die letzte Aktion so bewirkt hat. ;-)
  //Was ist der aktuelle Zustand (wird in ivCurrentState abgelegt)?
  if (this->determineCurrentStateFromWorldModel() == false)
  {
    cout<<"CANNOT DETERMINE WORLD STATE!"<<endl;
    return false;
  }
  //Na dann koennen wir ja ivCurrentState zur aktuellen Sequenz
  //hinzufuegen!
  if (ivVStateSequence.size() == 0)
    ivEpisodeInitialGoalDist = ivBallVector.distance(MY_GOAL_CENTER);
  if (ivStarted && WSinfo::ws->play_mode == PM_PlayOn)
    ivVStateSequence.push_back( State(ivCurrentState) );
    
  
  //Diese Variable wird weiter unten mit Daten belegt werden.
  Action chosenAction;

  switch (WSinfo::ws->play_mode)
  {
    //########################################################################
    case PM_my_KickIn:
    //########################################################################
      cvEvaluationMode = true;
      //Kein Break hier!!!

    //########################################################################
    case PM_my_FreeKick:  
    //########################################################################
      //Erst mal eine Auswertung der Sequenz!
      if ( !cvEvaluationMode )
      {
        this->learningUpdate();
      } 
      if ( cvEvaluationMode )
      {
        this->analyseCompletedEvaluationSequence();
      }
      //Wir muessen die Sequenz der Zustaende loeschen.
      ivVStateSequence .erase( ivVStateSequence.begin(), 
                               ivVStateSequence.end() );
      ivVActionSequence.erase( ivVActionSequence.begin(), 
                               ivVActionSequence.end() );
      //Was fuer einen Spielzustand haben wird denn eigentlich?
      if (WSinfo::ws->play_mode == PM_my_KickIn)
      {
        //Dies zeigt an, dass als naechstes eine Evaluationssequenz folgt!
        cvEvaluationMode = true;
      }
      if (WSinfo::ws->play_mode == PM_my_FreeKick)
      {
        //Dies zeigt an, dass eine Sequenz vorueber ist. ==> Auswertung erfolgte.
        //Statistiken
        cvEvaluationMode = false;
        cvLearnSequenceCounter ++ ;
      }

      cout<<endl<<"SEQUENZENDE"<<endl;
      break;
      
    //########################################################################
    case PM_PlayOn:   
    //########################################################################
      //Weiter ... die naechste Aktion bitte!
      //cout<<"PlayMode = PM_PlayOn"<<endl;
        
      if (!ivStarted) 
      { 
        ivStarted = true;
        ivVStateSequence.push_back( State(ivCurrentState) );
      }
      this->decideForAnAction(&chosenAction);
      chosenAction.performMeOnThisState( 
                                         ivCurrentState, 
                                         &ivPredictedSuccessorState,
                                         false);
      cout<<chosenAction.toString()<<endl;
      cout<<"CURR STATE = "<<ivCurrentState.toShortString()<<endl;
      cout<<"PRED SUCC STATE = "<<ivPredictedSuccessorState.toShortString()<<endl;

      ivVActionSequence.push_back( chosenAction );
      chosenAction.incorporateMeIntoCommand( &cmd );
      if (ivpHassleOpponent) {
        TGLOGnhPOL(0,<<"NeuroHassle: Obtained a NEURO HASSLE command! (bpDist="
          <<WSinfo::me->pos.distance(ivpHassleOpponent->pos)<<")");
      }
      break;
    //########################################################################
    case PM_TimeOver:
    //########################################################################
      //absolutes Ende des Lernvorganges
      cout<<"PlayMode = PM_TimeOver"<<endl;
      //Statistiken ... TBD
      exit(0);
      break;
    default:
      //cout<<"PlayMode = "<<WSinfo::ws->play_mode<<endl;
      break;
  } 
  float endTimeInMilliSeconds = Tools::get_current_ms_time();
  cout<<"get_cmd needed "<<(endTimeInMilliSeconds-beginTimeInMilliSeconds)<<" ms."<<endl;
  return true;
}


//--getValueFromFunctionApproximator()----------------------------------------
float
NeuroHassle::getValueFromFunctionApproximator( State & state, 
                                               float * netInputs)
{
  float returnValue;
  //Anfrage ans NN
  this->setNeuralNetInputFromState(state);
  cvpNeuralNetwork->forward_pass( cvpNeuralNetwork->in_vec,
                                  cvpNeuralNetwork->out_vec );
  returnValue = cvpNeuralNetwork->out_vec[0];
  return returnValue;
}


//--handleLearningStatistics()------------------------------------------------
void
NeuroHassle::handleLearningStatistics(int successCode)
{
    Statistics &statRef = cvEvaluationMode ? cvEvaluationStatistics
                                           : cvLearnStatistics;
    statRef.addEntry(successCode, 0.0/*this->calculateSequenceCosts(success)*/);

    if (      !cvEvaluationMode
          &&
              cvLearnStatistics.getSize() 
              % cvFlushLearnStatisticsAfterAsManySequences
              == 0         )
      cvLearnStatistics.writeOut(cvLearningProtocolFileHandle);
      
}

//--init()--------------------------------------------------------------------
bool
NeuroHassle::init(char const * conf_file, int argc, char const* const* argv)
{
  ValueParser vp(conf_file, "NeuroHassle");
  vp.get("op_mode", cvOperationMode);
  vp.get("neuralNetworkFilename", cvNeuralNetworkFilename,
         IDENTIFIER_LENGTH, "neuroHassleSpecification.net");
  vp.get("learningProtocolFile", cvLearningProtocolFileName,
         IDENTIFIER_LENGTH, "learnProtocol.txt");
  vp.get("evaluationProtocolFile", cvEvaluationProtocolFileName,
         IDENTIFIER_LENGTH, "evaluationProtocol.txt");
  vp.get("flushLearnStatisticsAfterAsManyEpisodes", 
         cvFlushLearnStatisticsAfterAsManySequences);
  vp.get("numberOfTrainingExamplesForNetworkTraining", 
         cvNumberOfTrainingExamplesForNetworkTraining);
         
  return true;
}

//--Initialisierung eines neuronalen Netzes als Funktionsapproximator---------
bool 
NeuroHassle::initNeuralNetwork()
{
  cvpNeuralNetwork = new Net;
  if ( cvpNeuralNetwork->load_net(cvNeuralNetworkFilename) == FILE_ERROR)
  {
    if (cvOperationMode == OP_MODE_EXPLOIT)
    {
      TGLOGnhPOL(0,<<"NeuroHassle: ERROR: Could not load net "
        <<cvNeuralNetworkFilename<<".");
      return false;
    }
    
    //Wir muessen das Netz selbst neu aufbauen ...
    int numberOfFeatures = 5;
    int numberOfNeuronsInHiddenLayer = 10;
    int numberOfOutputNeurons = 1;
    int nodesPerLayer[3];
    nodesPerLayer[0] = numberOfFeatures;
    nodesPerLayer[1] = numberOfNeuronsInHiddenLayer;
    nodesPerLayer[2] = numberOfOutputNeurons;
    float learnParams[5];
    learnParams[0] = 0.1; //delta null     //Lernrate
    learnParams[1] = 0.0; //delta max      //Momentum
    learnParams[2] = 0.0; //weight decay   //Wegwerfen minderer Gewichte
    learnParams[3] = 0.0;
    learnParams[4] = 0.0;
    cvpNeuralNetwork->create_layers(3, nodesPerLayer);
    cvpNeuralNetwork->connect_layers();
    cvpNeuralNetwork->init_weights(0, 0.5); //alle Gewichte initial zufaellig zwischen
                               //-0.5 und 0.5
    cvpNeuralNetwork->set_update_f(0, learnParams); //0==BP, 1==RPROP
    cvpNeuralNetwork->save_net(cvNeuralNetworkFilename);
    cout<<"Ein komplett neues Netz wurde generiert."<<endl;
  }
  else
  {
//    cout<<"Netz wurde erfolgreich geladen!"<<endl;
//    cout<<"  Update-Funktion: \t"<<cvpNeuralNetwork->update_id<<endl;
    if (cvOperationMode != OP_MODE_EXPLOIT)
    {
      cvpNeuralNetwork->save_net("neuroHassleTrained.net");
    }
    //for (int i=0; i<MAX_PARAMS; i++)
    //  cout<<"  Lernparameter "<<i<<": \t"<<cvpNeuralNetwork->update_params[i]<<endl;
  }
  return true;
}

//--isUnusableSequence()------------------------------------------------------
bool
NeuroHassle::isUnusableSequence()
{
  bool returnValue = false;
  
  if (ivVStateSequence.size() == 0) 
    return true;
  
  State & finalState = ivVStateSequence[ ivVStateSequence.size()-1 ];
  Vector ballVel( finalState.ivRealBallVelocityX,
                  finalState.ivRealBallVelocityY ),
         ballPos( finalState.ivBallPositionX,
                  finalState.ivBallPositionY );
  if (    ballPos.norm() > 2.0      //ausserhalb des opp-Kickrange
       && (   ballVel.norm() < 0.75 //kein richtiger schuss
           || ballVel.getX() > 0.0 )     //==ruasrutscher beim gegner
       && finalState.ivOpponentInterceptDistance > 2.0
     )
    returnValue = true;
  if (    ivVStateSequence.size() < 5 //kick aus der alten sequenz vom 
       && ballPos.norm() > 3.0)       //gegner wurde in der neuen 
    returnValue = true;               //sequenz umgesetzt

  TGLOGnhPOL(0,<<"NeuroHassle: Is Sequence UNUSABLE? (SSSize="
    <<ivVStateSequence.size()<<",bDist="
    <<ballPos.norm()<<", ballVel="<<ballVel<<") -> "<<returnValue);
  return returnValue;
}

//--neuralNetworkError()------------------------------------------------------
float
NeuroHassle::neuralNetworkError()
{
  int numOfTrainingExamples = ivVNeuralNetTrainingExamples.size();
  float error = 0.0;
  for (int e=0; e<numOfTrainingExamples; e++)
  {
    State state  = ivVNeuralNetTrainingExamples[e].first;
    float target = ivVNeuralNetTrainingExamples[e].second;
    this->setNeuralNetInputFromState(state);
    cvpNeuralNetwork->forward_pass( cvpNeuralNetwork->in_vec,
                                    cvpNeuralNetwork->out_vec );
if (e%1000==0) {
TGLOGnhPOL(0,<<"DISKREPANZ: target="<<target<<"\tnetOut="
  <<cvpNeuralNetwork->out_vec[0]<<"\tdelta="
  <<fabs(target - cvpNeuralNetwork->out_vec[0])<<"\t\t"
  <<state.toShortString());
}
    error += (target - cvpNeuralNetwork->out_vec[0])
             * (target - cvpNeuralNetwork->out_vec[0]);
  }
  return error;
}

//--learningUpdate()----------------------------------------------------------
void
NeuroHassle::learningUpdate()
{
  //verwerfen gaenzlich irrelevanter episoden
  if (isUnusableSequence()) return;


  TGLOGnhPOL(0,<<"NeuroHassle: Call method learningUpdate!");
  int successOrFailure 
    = this->classifySequenceAsSuccessOrFailure();
  TGLOGnhPOL(0,<<"NeuroHassle: Last sequence success or failure? -> "
    <<successOrFailure);
  TGLOGnhPOL(0,<<"NeuroHassle: Process tuples in ivVActionSequence: "
   <<ivVActionSequence.size()<<" and in ivVStateSequence: "
   <<ivVStateSequence.size()<<" (currently in ivVNNTE:"
   <<ivVNeuralNetTrainingExamples.size()<<")");
   
   //remove irritating episode ends
   if (   successOrFailure == IDENTIFIER_ABORT  //panic kick by opp
       && ivVStateSequence.size() > 1 )
   {
     int stateIndex = (int)ivVStateSequence.size() - 1;
     while (stateIndex >= 0)
     {
       Vector ballPos( ivVStateSequence[stateIndex].ivBallPositionX,
                       ivVStateSequence[stateIndex].ivBallPositionY );
       if (ballPos.norm() < 1.0)
         break;
       stateIndex -- ;
     }
     TGLOGnhPOL(0,<<"NeuroHassle: ERASE with stateIndex="<<stateIndex);
     ivVStateSequence.erase( ivVStateSequence.begin() + stateIndex + 1,
                             ivVStateSequence.end() );
     TGLOGnhPOL(0,<<"NeuroHassle: resultant size = "<<ivVStateSequence.size());
   }

  //defaults
  float netMin = 0.1, netMax = 0.9,
        targetMin = -1.0, targetMax = 1.0;

  //calculate target values
  float finalValue;
  if (successOrFailure == 1) finalValue = targetMax;
  if (successOrFailure == 0) finalValue = -targetMax * (2.0/3.0);
  if (successOrFailure <  0) finalValue = targetMin;
  
  //V1
  vector<float> vectorOfTargetValues;
  for (unsigned int i=0; i<ivVStateSequence.size(); i++)
  {
    State & currentState = ivVStateSequence[i];
    float currentValue
      =   finalValue
        * pow( ivDiscountFactor,
               (float)(ivVStateSequence.size()-1-i) );
    //scale value
    float scaledCurrentValue
      = (((currentValue-targetMin) / (targetMax-targetMin))
         * (netMax-netMin)) + netMin;
    //apply learning rate
    scaledCurrentValue
      =   1.0 * scaledCurrentValue
        + 0.0 * getValueFromFunctionApproximator(currentState, NULL);
    //add example
    vectorOfTargetValues.push_back( scaledCurrentValue );
  }
  /*//V2
  vector<float> vectorOfTargetValues;
  if (successOrFailure >= 0)
  {
    for (unsigned int i=0; i<ivVStateSequence.size(); i++)
    {
      float currentValue 
        //=   targetMax
        //  * pow( ivDiscountFactor, (float)(ivVStateSequence.size()-1-i) );
        = targetMax - (float)(ivVStateSequence.size()-1-i)*0.04;
      if (currentValue < 0.0) currentValue = 0.0;
      float scaledCurrentValue
        = (((currentValue) / (targetMax))
           * (netMax-netMin)) + netMin;
      vectorOfTargetValues.push_back( scaledCurrentValue );
    }
  }
  else //unsuccessful episode
  {
    for ( unsigned int i=0; i<ivVStateSequence.size(); i++ )
    {
      State & currentState = ivVStateSequence[i];
      Vector ballPos( currentState.ivBallPositionX,
                      currentState.ivBallPositionY );
      if ( ballPos.norm() < 1.5 )
      {
        float bestUnsuccessfulValue = 0.4;
        float currentValue 
          =   bestUnsuccessfulValue
          //  * pow( ivDiscountFactor, (float)i);

          //  * (1.0 - (fabs(currentState.ivOpponentInterceptAngle)/PI));
          //  - (currentState.ivOpponentInterceptDistance / 14.0);

            - (float)(ivVStateSequence.size()-i)*0.04;

        if (currentValue < 0.0) currentValue = 0.0;
        if (currentValue > bestUnsuccessfulValue) 
          currentValue = bestUnsuccessfulValue;  

        float scaledCurrentValue
          = (((currentValue) / (targetMax))
             * (netMax-netMin)) + netMin;
        vectorOfTargetValues.push_back( scaledCurrentValue );
      }
      else
        vectorOfTargetValues.push_back( -1000.0 );
      }
  }*/

  //print target values
  
  for (unsigned int i=0; i<ivVStateSequence.size(); i++)
  {
    TGLOGnhPOL(0,<<i<<": "<<ivVStateSequence[i].toShortString()
      <<" => V="<<vectorOfTargetValues[i] );
    if (i<ivVActionSequence.size())
    {
      TGLOGnhPOL(0,<<i<<": "<<ivVActionSequence[i].toString());
    }
  }
  
  //calculate average targets
  float targetSumNN = 0.0, targetSumSS = 0.0;
  for (unsigned int i=0; i<ivVNeuralNetTrainingExamples.size(); i++)
    targetSumNN += ivVNeuralNetTrainingExamples[i].second;
  for (unsigned int i=0; i<vectorOfTargetValues.size(); i++)
    targetSumSS += vectorOfTargetValues[i];
  float averageTargetNN
    =   targetSumNN / (float) ( ivVNeuralNetTrainingExamples.size());
  float averageTargetSS
    =   targetSumSS / (float) ( vectorOfTargetValues.size());

  //output target values
  
  for (unsigned int i=0; i<ivVNeuralNetTrainingExamples.size(); i++)
    TGLOGnhPOL(0,<<"NeuroHassle: target in ivVN.: "
      <<ivVNeuralNetTrainingExamples[i].first.toShortString()
      <<" -> "<<ivVNeuralNetTrainingExamples[i].second);
  for (unsigned int i=0; i<ivVStateSequence.size(); i++)
    TGLOGnhPOL(0,<<"NeuroHassle: target in ivVSS.: "
      <<ivVStateSequence[i].toShortString()
      <<" -> "<<vectorOfTargetValues[i]);
  
  //decision what to do now
  bool storePatterns = false;
  float balanceThresholdMin = 0.3, balanceThresholdMax = 0.7,
        balanceThresholdDelta = 0.01;
  if (ivVNeuralNetTrainingExamples.size() == 0)
  {
    //no patterns stored, yet
    storePatterns = true;
  }
  else
  {
    //some patterns already stored
    if ( averageTargetSS > balanceThresholdMin )
      storePatterns = true;
    else //averageTargetSS <= 0.3
      if ( averageTargetNN > balanceThresholdMin )
        storePatterns = true;
    
  }
  //store patterns
  if (1 || storePatterns)
  //if (successOrFailure >= 0.0)
  {
    TGLOGnhPOL(0,<<"NeuroHassle: store patterns");
    int numberOfNegativeExamples = 0;
    for (int i = (int)ivVStateSequence.size() - 1; i>= 0; i--)
    {
      if (   vectorOfTargetValues[i] > -100.0
          && numberOfNegativeExamples < 3000 )
      {
        setValueForFunctionApproximator( vectorOfTargetValues[i],
                                         ivVStateSequence[i],
                                         NULL );
        TGLOGnhPOL(0,<<"NeuroHassle: stored pattern : "
          <<ivVStateSequence[i].toShortString()<<" -> "
          <<vectorOfTargetValues[i]);
        if (successOrFailure < 0) numberOfNegativeExamples ++ ;
      }
    }
  }
  //recalculate average
  targetSumNN = 0.0;
  for (unsigned int i=0; i<ivVNeuralNetTrainingExamples.size(); i++)
    targetSumNN += ivVNeuralNetTrainingExamples[i].second;
  averageTargetNN 
    = targetSumNN / (float) ( ivVNeuralNetTrainingExamples.size());

  //decide whether to train the net
if (0)
  if (      averageTargetNN 
          < balanceThresholdMin - ivUnExploitedEpisodeCounter*balanceThresholdDelta
       ||   averageTargetNN 
          > balanceThresholdMax + ivUnExploitedEpisodeCounter*balanceThresholdDelta)
  {
    TGLOGnhPOL(0,<<"NeuroHassle: Don't train the net, because of unbalanced"
      <<" average target = "<<averageTargetNN
      <<", MIN="<<balanceThresholdMin - ivUnExploitedEpisodeCounter*balanceThresholdDelta);
    ivUnExploitedEpisodeCounter ++ ;
    return;
  }
//  else  
  {
    if (ivHaveCollectedEnoughTrainingExamplesForNextTrainingCycle)
    {
      trainNeuralNetwork();
      ivTrainNeuralNetworkCounter ++;
      ivHaveCollectedEnoughTrainingExamplesForNextTrainingCycle = false;
    }
    ivUnExploitedEpisodeCounter = 0;
  }       
}

//--setNeuralNetInputFromState()----------------------------------------------
void 
NeuroHassle::setNeuralNetInputFromState(State & s)
{
      ////////////////////////////////////////////////////////////////////////
      // BLOCK ZUR NETZEINGABE-GENERIERUNG
      //Da der Winkel in [-180;180] liegt, sollte eine Normalisierung
      //erfolgen!
      cvpNeuralNetwork->in_vec[0] = s.ivOpponentInterceptAngle;
      cvpNeuralNetwork->in_vec[1] = s.ivOpponentInterceptDistance - 2.0;
//      cvpNeuralNetwork->in_vec[0] = s.ivMyPositionX;
//      cvpNeuralNetwork->in_vec[1] = s.ivMyPositionY;
      cvpNeuralNetwork->in_vec[2] = s.ivMyVelocityX * 5.0;
      cvpNeuralNetwork->in_vec[3] = s.ivMyVelocityY * 5.0;
      cvpNeuralNetwork->in_vec[4] = s.ivMyAngleToOpponentPosition;
      cvpNeuralNetwork->in_vec[5] = s.ivBallPositionX * 2.0;
      cvpNeuralNetwork->in_vec[6] = s.ivBallPositionY * 2.0;
      cvpNeuralNetwork->in_vec[7] = s.ivOpponentBodyAngle;
      cvpNeuralNetwork->in_vec[8] = s.ivOpponentAbsoluteVelocity * 5.0;
      ////////////////////////////////////////////////////////////////////////
}

//--setOpponent()-------------------------------------------------------------
void
NeuroHassle::setOpponent( PPlayer opp )
{
  if ( opp == NULL )
  {
    TGLOGnhPOL(0,<<"NeuroHassle: Warning, opponent player is NULL.");
  }
  else
  {
    TGLOGnhPOL(0,<<_2D<<L2D(WSinfo::me->pos.x,WSinfo::me->pos.y,
                            opp->pos.x,opp->pos.y,"ffff66"));
  }
  ivBallVector = WSinfo::ball->pos;
  ivpHassleOpponent = opp;
}


//--setVirtualBall()----------------------------------------------------------
void
NeuroHassle::setVirtualBall( Vector * ballPos )
{
  if ( ballPos == NULL )
  {
    TGLOGnhPOL(0,<<"NeuroHassle: Use normal ball.");
    ivBallVector = WSinfo::ball->pos;    
  }
  else
  {
    ivBallVector = *ballPos;
  }
}



//--setValueForFunctionApproximator()----------------------------------------
void
NeuroHassle::setValueForFunctionApproximator(float v,
                                            State &state, 
                                            float *netInputs)
{
    pair<State, float> trainingExample(state, v);
    ivVNeuralNetTrainingExamples.push_back(trainingExample);
    if (   (int)ivVNeuralNetTrainingExamples.size() 
        >= (int)((float)cvNumberOfTrainingExamplesForNetworkTraining 
                 * (float)(ivTrainNeuralNetworkCounter+1)
                 * pow( 1.05, (double)ivTrainNeuralNetworkCounter )  
                )     
       )
      ivHaveCollectedEnoughTrainingExamplesForNextTrainingCycle = true; 
}


//--trainNeuralNetwork()------------------------------------------------------
void
NeuroHassle::trainNeuralNetwork()
{
  char bestName[100];
  sprintf(bestName,
          "best500Net_%d_%d_%.2f.net", cvLearnSequenceCounter,
                                       cvLearnStatistics.getSize(),
                                       cvLearnStatistics.getGlidingWindowFailures(500));
 cvpNeuralNetwork->save_net(bestName);

  for (unsigned int i=0; i<ivVNeuralNetTrainingExamples.size(); i++)
    ivVCurrentNeuralNetTrainingExamples.push_back( ivVNeuralNetTrainingExamples[i] );
  for (unsigned int i=0; i<ivVTopNeuralNetTrainingExamples.size(); i++)
    ivVNeuralNetTrainingExamples.push_back( ivVTopNeuralNetTrainingExamples[i] );
  for (unsigned int i=0; i<ivVBottomNeuralNetTrainingExamples.size(); i++)
    ivVNeuralNetTrainingExamples.push_back( ivVBottomNeuralNetTrainingExamples[i] );
  
  float e1 = this->neuralNetworkError();
  TGLOGnhPOL(0,<<"NeuroHassle: trainNN: FEHLER auf TD vorher: "<<e1);
  //lokale Variablen
  //int numOfLearningEpochs   = (int)sqrt((float)ivVNeuralNetTrainingExamples.size()); 
  int numOfLearningEpochs   = 500;//50;// * (ivTrainNeuralNetworkCounter+1);
cvpNeuralNetwork->init_weights(0, 0.5);
  int numOfTrainingExamples = ivVNeuralNetTrainingExamples.size();
  float summedSquaredError  = 0.0,
        error               = 0.0;
  //(Re-)Initialisierung des Netzes ==> NEIN!
    //cvpNeuralNetwork->init_weights(0, 0.5);
    //cvpNeuralNetwork->set_update_f(1,uparams);
  //Hauptschleife
  for (int n=0; n<numOfLearningEpochs; n++)
  {
    cout<<n<<flush;
    summedSquaredError = 0.0;
    for (int e=0; e<numOfTrainingExamples; e++)
    {
      State state  = ivVNeuralNetTrainingExamples[e].first;
      float target = ivVNeuralNetTrainingExamples[e].second;
      this->setNeuralNetInputFromState(state);
      cvpNeuralNetwork->forward_pass( cvpNeuralNetwork->in_vec,
                                      cvpNeuralNetwork->out_vec );
      for (int o=0; o < cvpNeuralNetwork->topo_data.out_count; o++)
      {
        error = cvpNeuralNetwork->out_vec[o] - target;
        cvpNeuralNetwork->out_vec[o] = error;
        summedSquaredError += error * error;
      }
      cvpNeuralNetwork->backward_pass( cvpNeuralNetwork->out_vec, 
                                       cvpNeuralNetwork->in_vec);
    
    }
    cvpNeuralNetwork->update_weights();
    //float dummyError = this->neuralNetworkError();
    TGLOGnhPOL(0,<<"TSS:"<<summedSquaredError);
    //TGLOGnhPOL(0,<<"FEHLER auf TD mittendrin: "<<dummyError);
  }
  cvpNeuralNetwork->save_net("neuroHassleTrained.net");
    TGLOGnhPOL(0,<<"NeuroHassle: trainNN: Fertig, neuronales Netz wurde "
      <<"trainiert (und als "<<"neuroHassleTrained.net abgelegt).");
  float e2=this->neuralNetworkError();
    TGLOGnhPOL(0,<<"NeuroHassle: trainNN: FEHLER auf TD nachher: "
      <<e2<<"   (vs. vorher: "<<e1<<")"<<endl);
  //Kurze Log-Ausgabe
  cvLearningProtocolFileHandle.open(cvLearningProtocolFileName, 
                                    ofstream::out | ofstream::app);
  cvLearningProtocolFileHandle<<"# train nn at t="
    <<cvLearnStatistics.getSize()
    <<" with #te="<<ivVNeuralNetTrainingExamples.size()
    <<" (>="<<(cvNumberOfTrainingExamplesForNetworkTraining 
           * (ivTrainNeuralNetworkCounter+1))<<")"
    <<" FEHLER auf TD nachher: "<<e2<<"   (vs. vorher: "<<e1<<")"<<endl;
  cvLearningProtocolFileHandle.close();  


  //this->eraseTrainingExamples( true ); //true=behalte top-examples
}
