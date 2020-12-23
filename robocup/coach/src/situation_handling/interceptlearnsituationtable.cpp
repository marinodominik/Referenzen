#include "interceptlearnsituationtable.h"
#include "situationcollection.h"
#include "param.h"


#define 	MAXIMAL_BALL_PLAYER_DISTANCE 28.0
#define 	MINIMAL_BALL_PLAYER_DISTANCE 	2.0

//============================================================================
//  KONSTRUCTOR
//============================================================================
InterceptLearnSituationTable::InterceptLearnSituationTable ()
{
}

//============================================================================
//  HAUPTMETHODEN
//============================================================================

//---------------------------------------------------------------------------
//  createRandomSituationTable
//---------------------------------------------------------------------------
bool 
InterceptLearnSituationTable::
  createRandomSituationTable(int rows, bool ball, int numPlayers)
{

  ivNumRows = 0;
  ivNumCols = 0;
  if (ball) ivNumCols += 4;
  ivNumCols += 5 * numPlayers;
  if (rows > 50)
    this->setMaxRows(rows);
  else
    this->setMaxRows(50);

  float *tmp = 0;

  for (int i=0; i<rows; i++)
  {
/*if(i%100==0)*/cout<<i+1<<" von "<<rows<<endl<<flush;
    this->createSingleRandomSituation( *(&tmp), ball );
    ivppTableArray[ivNumRows] = tmp;
    ivNumRows++;
    tmp = 0;
  }

  if (tmp)
    delete[] tmp;
  return true;
}

//---------------------------------------------------------------------------
// METHODE createSingleRandomSituation
//---------------------------------------------------------------------------
void
InterceptLearnSituationTable::
  createSingleRandomSituation( float *(&tmp), bool ball )
{
    if (ivNumCols == 0)
    {
        int numPlayers = 22;
    	if (ball) ivNumCols += 4;
    	ivNumCols += numPlayers * 5;
    }
    if (tmp==0) 
      tmp = new float[ivNumCols];

    //SPIELER -> in der Mitte des Spielfeldes positionieren
    tmp[4] = 0; // x pos
    tmp[5] = 0; // y pos
    tmp[6] = this->getRandomPlayerAngle(); // angle
    bool velOk = false;
	while (!velOk)
	{
      tmp[7] = this->getRandomPlayerVelX(); // x vel
      tmp[8] = this->getRandomPlayerVelY(); // y vel
      velOk = this->checkPlayerVelocity(tmp[7], tmp[8]);
	}
    //RESTLICHE SPIELER
    for (int j=2; j<NUMBER_OF_OBJECTS_IN_SITUATION; j++)
      this->setOffObject(j);
  	//BALL -> zuf?llig im Kickrange des Spielers positionieren
  	if (ball)
    {
      float kickableArea = ServerParam::kickable_margin;
      bool ballKickable   = true,
           ballTooNear    = true,
           ballTooDistant = true;
      while (ballKickable || ballTooDistant || ballTooNear)
      {
      	ballKickable   = true;
      	ballTooNear    = true;
      	ballTooDistant = true;
	    tmp[0] = this->getRandomPlayerPosWithinRange(-MAXIMAL_BALL_PLAYER_DISTANCE, MAXIMAL_BALL_PLAYER_DISTANCE);
	    tmp[1] = this->getRandomPlayerPosWithinRange(-MAXIMAL_BALL_PLAYER_DISTANCE, MAXIMAL_BALL_PLAYER_DISTANCE);
	    ballKickable = this->checkIsBallKickable(
	                         tmp[4],tmp[5],tmp[0],tmp[1],kickableArea);
        float deltaX = tmp[0] - tmp[4];
        float deltaY = tmp[1] - tmp[5];
        float ballDistance = sqrt( deltaX*deltaX + deltaY*deltaY );
cout<<ballDistance<<endl;
        if (ballDistance <= MAXIMAL_BALL_PLAYER_DISTANCE)
          ballTooDistant = false;
        if (ballDistance > MINIMAL_BALL_PLAYER_DISTANCE)
          ballTooNear = false;
      }
	  bool velOk = false;
	  while (!velOk)
	  {
	  	float ballSpeedDivisor = 1.0;
  	    tmp[2] = this->getRandomBallVelX();
	    tmp[3] = this->getRandomBallVelY();
	    velOk = this->checkBallVelocity(tmp[2], tmp[3], ballSpeedDivisor);
        if (0&&velOk)
        {
          Vector playerToBall = Vector(tmp[0] - tmp[4], tmp[1] - tmp[5]);
          Vector ballVel = Vector(tmp[2], tmp[3]);
          Angle angleDiff = playerToBall.arg() - ballVel.arg(); //arg liefer in 0..2pi
          if (angleDiff < 0.0) angleDiff += 2*PI; //vorher in -2pi..2pi, jetzt in 0..2pi
          if (angleDiff > PI)  angleDiff -= PI; //jetzt in 0..pi
          if (angleDiff < ((85.0/180.0)*PI) ) velOk = false;
          if (angleDiff > ((95.0/180.0)*PI) ) velOk = false;
cout<<"playerToBall=("<<playerToBall.getX()<<","<<playerToBall.getY()<<")  ballVel=("<<ballVel.getX()<<","<<ballVel.getY()<<")  angleDiff="<<angleDiff<<"   OK? "<<velOk<<endl;
        }
	  }
	  //Zu setzende Positionen in X- wie Y-Richtung muessen um die 
	  //Ballgeschwindigkeit dekrementiert werden, da im ersten
	  //fuer den Spieler wahrnehmbaren Zyklus die Geschwindigkeiten
	  //auf die Positionen durch den Server addiert worden sind!
	  tmp[0] -= tmp[2];
	  tmp[1] -= tmp[3];
    }
}

//===========================================================================
// PRIVATE METHODEN (Hilfsfunktionen)
//===========================================================================

//---------------------------------------------------------------------------
// METHODE checkIsBallKickable
//---------------------------------------------------------------------------
bool 
InterceptLearnSituationTable::
  checkIsBallKickable
    (float plX, float plY, float bX, float bY, float kickRadius)
{
	float delta = sqrt( (plX-bX)*(plX-bX) + (plY-bY)*(plY-bY) );
	delta = delta - (ServerParam::ball_size) - (ServerParam::player_size);
	if (delta < 0) //ball too narrow to player
	  return false;
	return delta < kickRadius;
}





