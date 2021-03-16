#include "interceptlearnevaluationsituationtable.h"
#include "situationcollection.h"
#include "param.h"


#define 	MAXIMAL_BALL_PLAYER_DISTANCE 	8.0
#define 	MINIMAL_BALL_PLAYER_DISTANCE 	3.0

//============================================================================
//  KONSTRUCTOR
//============================================================================
InterceptLearnEvaluationSituationTable
  ::InterceptLearnEvaluationSituationTable ()
{
}

//============================================================================
//  HAUPTMETHODEN
//============================================================================

//---------------------------------------------------------------------------
//  createRandomSituationTable
//---------------------------------------------------------------------------
bool 
InterceptLearnEvaluationSituationTable::
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

  float ballSpeedDivisor = 1.0;
  float speedDelta = 0.1;
  float xSpeedMin = - ServerParam::ball_speed_max / ballSpeedDivisor,
        xSpeedMax =   ServerParam::ball_speed_max / ballSpeedDivisor;
  float ySpeedMin = - ServerParam::ball_speed_max / ballSpeedDivisor,
        ySpeedMax =   ServerParam::ball_speed_max / ballSpeedDivisor;

  this->setMaxRows(   (xSpeedMax-xSpeedMin)/speedDelta
                    * (ySpeedMax-ySpeedMin)/speedDelta );

  for ( int i=0; 
        (xSpeedMin + i*speedDelta) <= xSpeedMax;
        i++ )
  {
    for ( int j=0; 
          (ySpeedMin + j*speedDelta) <= ySpeedMax;
          j++ )
    {
      float xSpeed = (xSpeedMin + i*speedDelta);
      float ySpeed = (ySpeedMin + j*speedDelta);
      cout<<(xSpeedMin + i*speedDelta)<<" und "<<(ySpeedMin + j*speedDelta)<<endl<<flush;
      if ( this->checkBallVelocity(0.99*xSpeed, 0.99*ySpeed, ballSpeedDivisor) )
      {
        this->createSingleRandomSituation( *(&tmp), ball,
                                           xSpeed,
                                           ySpeed );
        ivppTableArray[ivNumRows] = tmp;
        ivNumRows++;
        tmp = 0;
      }
    }
  }

  if (tmp)
    delete[] tmp;
  return true;
}

//---------------------------------------------------------------------------
// METHODE createSingleRandomSituation
//---------------------------------------------------------------------------
void
InterceptLearnEvaluationSituationTable::
  createSingleRandomSituation( float *(&tmp), bool ball, 
                               float ballVelocityX,
                               float ballVelocityY )
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
    tmp[6] = 0; // angle
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
      tmp[0] = 5.0;
	  tmp[1] = 0.0;
	  bool velOk = false;
	  while (!velOk)
	  {
//	  	float ballSpeedDivisor = 2.0;
  	    tmp[2] = ballVelocityX;
	    tmp[3] = ballVelocityY;
	    velOk = true;
	    //velOk = this->checkBallVelocity(0.99*tmp[2], 0.99*tmp[3], ballSpeedDivisor);
	  }
	  //Zu setzende Positionen in X- wie Y-Richtung muessen um die 
	  //Ballgeschwindigkeit dekrementiert werden, da im ersten
	  //fuer den Spieler wahrnehmbaren Zyklus die Geschwindigkeiten
	  //auf die Positionen durch den Server addiert worden sind!
	  tmp[0] -= tmp[2];
	  tmp[1] -= tmp[3];
	  tmp[2] /= 0.94;
	  tmp[3] /= 0.94;
    }
}

//===========================================================================
// PRIVATE METHODEN (Hilfsfunktionen)
//===========================================================================

//---------------------------------------------------------------------------
// METHODE checkIsBallKickable
//---------------------------------------------------------------------------
bool 
InterceptLearnEvaluationSituationTable::
  checkIsBallKickable
    (float plX, float plY, float bX, float bY, float kickRadius)
{
	float delta = sqrt( (plX-bX)*(plX-bX) + (plY-bY)*(plY-bY) );
	delta = delta - (ServerParam::ball_size) - (ServerParam::player_size);
	if (delta < 0) //ball too narrow to player
	  return false;
	return delta < kickRadius;
}





