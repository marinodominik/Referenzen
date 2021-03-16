#include "hasslelearnsituationtable.h"
#include "situationcollection.h"
#include "param.h"


//============================================================================
//  KONSTRUCTOR
//============================================================================
HassleLearnSituationTable::HassleLearnSituationTable ()
{
	
}

//============================================================================
//  HAUPTMETHODEN
//============================================================================

//---------------------------------------------------------------------------
//  createRandomSituationTable
//---------------------------------------------------------------------------
bool 
HassleLearnSituationTable::
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

  float *tmp = (float*) malloc( sizeof(float) * ivNumCols );

  for (int i=0; i<rows; i++)
  {
//if(i%100==0)
    cout<<i+1<<" von "<<rows<<endl<<flush;
    this->createSingleRandomSituation( *(&tmp), ball );
    ivppTableArray[ivNumRows] 
      = (float*) malloc( sizeof(float) * ivNumCols);
    float *certainRow = ivppTableArray[ivNumRows]; 
    for (int j=0; j<ivNumCols; j++)
      certainRow[j] = tmp[j]; 
    ivNumRows++;
    //RESTLICHE SPIELER
    for (int j=1; j<NUMBER_OF_OBJECTS_IN_SITUATION; j++)
      if ( j != 1 && j != 2 && j != 12 )
        this->setOffObject(j);
  }
  if (tmp) free(tmp);
  return true;
}

//---------------------------------------------------------------------------
// METHODE createSingleRandomSituation
//---------------------------------------------------------------------------
void
HassleLearnSituationTable::
  createSingleRandomSituation( float *(&tmp), bool ball )
{

  bool allright = false;
  while (allright == false)
  {
    //verschiebung
    int modus = 0;
    float xShift, yShift;
    switch (modus)
    {
      case 0: //default: am mittelpunkt
      {
        xShift = 0.0;     yShift = 0.0;
        break;
      }
      case 1: //ausgangspunkt: P(-25.0,0.0)
      {
        xShift = -25.0;     yShift = 0.0;
        break;
      }
      case 2: //ausgangspunkt: P(-40.0,25.0)
      {
        xShift = -40.0;     yShift = 25.0;
        break;
      }
      case 3: //ausgangspunkt: P(-40.0,25.0)
      {
        xShift = -40.0;     yShift = -25.0;
        break;
      }
      default: break;
    }
    //SPIELER -> in der Mitte des Spielfeldes still stehend positionieren
    tmp[4] = 0.0 + xShift; // x pos
    tmp[5] = 0.0 + yShift; // y pos
    tmp[6] = this->getRandomPlayerAngle(); // angle
    tmp[7] = this->getRandomPlayerVelX(); // x vel
    tmp[8] = this->getRandomPlayerVelY(); // y vel
    //TORMANN -> im Tor
    tmp[ 9] = -52.0;
    tmp[10] = 0.0;
    tmp[11] = 0.0;
    tmp[12] = 0.0;
    tmp[13] = 0.0;
    //GEGENSPIELER: 59,60,61,62,63
    float relativeAngleToMe = this->getRandomPlayerAngle();// / 2.0 - (PI/2.0);
    float standardPlayerDistance = 5.0;
    switch (modus)
    {
      case 0:
      case 1:
      {
        if (   relativeAngleToMe > 0.5*PI
            && relativeAngleToMe < 1.5*PI ) 
          standardPlayerDistance = 3.0;
        break;
      }
      case 2:
      {
        if (   relativeAngleToMe > 2.678
            && relativeAngleToMe < 5.820 ) 
          standardPlayerDistance = 3.0;
        break;
      }
      case 3:
      {
        if (   relativeAngleToMe > 0.464
            && relativeAngleToMe < 3.605 ) 
          standardPlayerDistance = 3.0;
        break;
      }
      default: break;
    }
    tmp[59] = cos(relativeAngleToMe) * standardPlayerDistance + xShift;
    tmp[60] = sin(relativeAngleToMe) * standardPlayerDistance + yShift;
    tmp[61] = this->getRandomPlayerAngle(); // angle 
    tmp[62] = 0.0;//this->getRandomPlayerVelX(); // x vel
    tmp[63] = 0.0;//this->getRandomPlayerVelY(); // y vel
  	//BALL -> zufaellig in der Naehe des Gegenspielers
  	if (ball)
    {
      float ballAngleFromOpponent = this->getRandomPlayerAngle();
      float standardBallDistance = 1.2; 
      tmp[0] = tmp[59] + cos(ballAngleFromOpponent) * standardBallDistance;
      tmp[1] = tmp[60] + sin(ballAngleFromOpponent) * standardBallDistance;
      tmp[2] = 0.0;
      tmp[3] = 0.0;
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
    tmp[59] -= tmp[62];
    tmp[60] -= tmp[63];
    
    allright = true;
    if (1)
    {
      float myDistToMyGoal 
        = sqrt(   fabs( tmp[4] - (-52.5) ) * fabs( tmp[4] - (-52.5) )
                + fabs( tmp[5] -  0.0 ) * fabs( tmp[5] -  0.0 ) );
      float oppDistToMyGoal
        = sqrt(   fabs( tmp[59] - (-52.5) ) * fabs( tmp[59] - (-52.5) )
                + fabs( tmp[60] -  0.0 ) * fabs( tmp[60] -  0.0 ) );
      if (oppDistToMyGoal<myDistToMyGoal) 
        allright = false;
    }
  }
}

//===========================================================================
// PRIVATE METHODEN (Hilfsfunktionen)
//===========================================================================

//---------------------------------------------------------------------------
// METHODE checkIsBallKickable
//---------------------------------------------------------------------------
bool 
HassleLearnSituationTable::
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
HassleLearnSituationTable::
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
                                




