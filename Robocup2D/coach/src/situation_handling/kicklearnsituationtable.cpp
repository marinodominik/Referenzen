#include "kicklearnsituationtable.h"
#include "situationcollection.h"
#include "param.h"


//============================================================================
//  KONSTRUCTOR
//============================================================================
KickLearnSituationTable::KickLearnSituationTable ()
{
	
}

//============================================================================
//  HAUPTMETHODEN
//============================================================================

//---------------------------------------------------------------------------
//  createRandomSituationTable
//---------------------------------------------------------------------------
bool 
KickLearnSituationTable::
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
KickLearnSituationTable::
  createSingleRandomSituation( float *(&tmp), bool ball )
{
   if (tmp==0) 
      tmp = new float[ivNumCols];

  bool allright = false;
  while (allright == false)
  {
    //SPIELER -> in der Mitte des Spielfeldes still stehend positionieren
    tmp[4] = getRandomPlayerPosWithinRange(-47.5, 12.5);// 0; // x pos
    tmp[5] = getRandomPlayerPosWithinRange(-30.0, 30.0); //0; // y pos
    tmp[6] = this->getRandomPlayerAngle(); // angle
    tmp[7] = this->getRandomPlayerVelX(); // x vel
    tmp[8] = this->getRandomPlayerVelY(); // y vel
    //RESTLICHE SPIELER
    for (int j=2; j<NUMBER_OF_OBJECTS_IN_SITUATION; j++)
      this->setOffObject(j);
  	//BALL -> zuf?llig im Kickrange des Spielers positionieren
  	if (ball)
    {
      float kickableArea = ServerParam::kickable_margin;
      bool ballKickable = false;
      while (!ballKickable)
      {

	    tmp[0] = this->getRandomPlayerPosWithinRange(-2*kickableArea, 2*kickableArea) + tmp[4];
	    tmp[1] = this->getRandomPlayerPosWithinRange(-2*kickableArea, 2*kickableArea) + tmp[5];
	    ballKickable = this->checkIsBallKickable(
	                         tmp[4],tmp[5],tmp[0],tmp[1],kickableArea);
      }
  	  bool velOk = false;
	    while (!velOk)
	    {
  	    tmp[2] = this->getRandomBallVelX() ;
  	    tmp[3] = this->getRandomBallVelY() ;
	      float ballSpeedDivisor = 1.0;
	      velOk = this->checkBallVelocity(tmp[2], tmp[3], ballSpeedDivisor);
        velOk =    velOk 
                && this->checkBallPositionInNextCycle(tmp[0], tmp[1], tmp[2], tmp[3], 
                                                      tmp[4], tmp[5], tmp[7], tmp[8], tmp[6]);
  	  }
	    //Zu setzende Positionen in X- wie Y-Richtung muessen um die 
	    //Ballgeschwindigkeit dekrementiert werden, da im ersten
  	  //fuer den Spieler wahrnehmbaren Zyklus die Geschwindigkeiten
	    //auf die Positionen durch den Server addiert worden sind!
	    tmp[0] -= tmp[2];
  	  tmp[1] -= tmp[3];
      //Und das gilt nicht nur fuer den Ball, sondern auch fuer den Spieler!
      tmp[4] -= tmp[7];
      tmp[5] -= tmp[8];
    }
    //check for pre-collision
    float delta = sqrt( (tmp[0]-tmp[4])*(tmp[0]-tmp[4]) + (tmp[1]-tmp[5])*(tmp[1]-tmp[5]) );
    delta = delta - (ServerParam::ball_size) - (ServerParam::player_size);
    if (delta > 0) //ball too narrow to player
      allright = true;
  }
}

//===========================================================================
// PRIVATE METHODEN (Hilfsfunktionen)
//===========================================================================

//---------------------------------------------------------------------------
// METHODE checkIsBallKickable
//---------------------------------------------------------------------------
bool 
KickLearnSituationTable::
  checkIsBallKickable
    (float plX, float plY, float bX, float bY, float kickRadius)
{
	float delta = sqrt( (plX-bX)*(plX-bX) + (plY-bY)*(plY-bY) );
	delta = delta - (ServerParam::ball_size) - (ServerParam::player_size);
	if (delta < 0) //ball too narrow to player
	  return false;
	return delta < kickRadius;
}

//---------------------------------------------------------------------------
// METHODE checkBallPositionInNextCycle
//---------------------------------------------------------------------------
bool 
KickLearnSituationTable::
  checkBallPositionInNextCycle( float ballPosX, float ballPosY,
                                float ballVelX, float ballVelY,
                                float playerPosX, float playerPosY,
                                float playerVelX, float playerVelY,
                                float playerAng )//in 0..2PI
{
  float kickPowerRate = ServerParam::kick_power_rate;
  float maxKickPower = 100;
  float distToBall =   sqrt(ballPosX*ballPosX+ballPosY*ballPosY)
                     - (ServerParam::ball_size) - (ServerParam::player_size);
  //compute the player's next position
  float nextPlayerPosX = playerPosX + playerVelX;
  float nextPlayerPosY = playerPosY + playerVelY;
  //compute whether the ball will leave my kickable area
  float nextBallPosX = ballPosX + ballVelX;
  float nextBallPosY = ballPosY + ballVelY;
  if ( checkIsBallKickable( nextPlayerPosX, nextPlayerPosY, 
                            nextBallPosX, nextBallPosY, ServerParam::kickable_margin) )
    return true;
  //ok, what we know at this point, is that the ball will leave the kickable area
  //let us compute how much it will go outside the kickable area
  float ballPlayerXDelta = nextBallPosX - nextPlayerPosX;
  float ballPlayerYDelta = nextBallPosY - nextPlayerPosY;
  float outSideKickableArea
    =   sqrt(ballPlayerXDelta*ballPlayerXDelta+ballPlayerYDelta*ballPlayerYDelta)
      - (ServerParam::ball_size) - (ServerParam::player_size)
      - ServerParam::kickable_margin;
  //now let us determine the effektive kick power usable at this moment
  float ballDirection = atan2( ballPosY, ballPosX );
  if (ballDirection<0.0) ballDirection += 2*PI;
  float directionDifference = fabs(ballDirection - playerAng);
  if (directionDifference > PI) directionDifference -= PI;
  float kickImpairment1 = 0.25 * (directionDifference / PI);
  float kickImpairment2 = 0.25 * (distToBall/ServerParam::kickable_margin);
  float normalEffectiveKickPower = maxKickPower * kickPowerRate; //2.7
  float effectiveKickPower =   normalEffectiveKickPower 
                             * (1.0 - kickImpairment1 - kickImpairment2 );
  //the effective kick power computed so far sorresponds to a vector that
  //is added (durnig kicking) to the ball's current velocity vector
  //therefore: the length of effective kick power must be at least as long
  //as the value of outSideKickableArea in order for the player to be able
  //to keep the ball in its kickable area
  if ( effectiveKickPower > outSideKickableArea )
    return true;
  else
    return false;
}
                                




