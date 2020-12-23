#include "attackpositioninglearnsituationtable.h"
#include "situationcollection.h"

#include "logger.h"
#include "param.h"

#include "ConfigReader.h"

#include <string>

//============================================================================
//  KONSTRUCTOR
//============================================================================
AttackPositioningLearnSituationTable::AttackPositioningLearnSituationTable ()
{
}

//============================================================================
//  HAUPTMETHODEN
//============================================================================

//---------------------------------------------------------------------------
// METHODE checkIsBallKickable
//---------------------------------------------------------------------------
bool 
AttackPositioningLearnSituationTable::
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
//  createRandomSituationTable
//---------------------------------------------------------------------------
bool 
AttackPositioningLearnSituationTable::
  createRandomSituationTable(int rows, bool ball, int numPlayers)
{
  //Initialisierung ist noetig, um aus der Coach-Konfigurationsdatei die
  //Parameter einzulesen, nach denen die Spielerpositionen zu setzen sind.
  bool returnValue = this->init();
  if (!returnValue) 
  {
    LOG_FLD(1, "AttackPositioningLearnSituationTable: ERROR, I could not"
      <<" initialize this object.");
    return false;
  }
  
  //Start der zufaelligen Generierung
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
    this->createSingleRandomSituation( *(&tmp), numPlayers, ball );
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
/**
 * Eine einzelne zufaellige Startposition wird mittels dieser Methode 
 * erstellt.
 * 
 * Bezueglich der Spalten in der hiesigen Tabelle 
 * AttackPositioningLearnSituationTable ist folgendes zu bemerken:
 * 
 * Eintrag              Semantik
 * 0..1                 Ballposition
 * 2..3                 Ballgeschwindigkeit
 * 4..5                 Position Spieler 0
 * 6                    Koerperausrichtung (Winkel) Spieler 0
 * 7..8                 Geschwindigkeit Spieler 0
 * 9..13                Position/Winkel/Geschwindigkeit Spieler 1
 * 14..18               Position/Winkel/Geschwindigkeit Spieler 2
 * 19..23               Position/Winkel/Geschwindigkeit Spieler 3
 * 24..28               Position/Winkel/Geschwindigkeit Spieler 4
 * 29..33               Position/Winkel/Geschwindigkeit Spieler 5
 * 34..38               Position/Winkel/Geschwindigkeit Spieler 6
 * 39..43               Position/Winkel/Geschwindigkeit Spieler 7
 * 44..48               Position/Winkel/Geschwindigkeit Spieler 8
 * 49..53               Position/Winkel/Geschwindigkeit Spieler 9
 * 54..58               Position/Winkel/Geschwindigkeit Spieler 10
 * 59..63               Position/Winkel/Geschwindigkeit Spieler 11
 * 64..68               Position/Winkel/Geschwindigkeit Spieler 12
 * 69..73               Position/Winkel/Geschwindigkeit Spieler 13
 * 74..78               Position/Winkel/Geschwindigkeit Spieler 14
 * 79..83               Position/Winkel/Geschwindigkeit Spieler 15
 * 84..88               Position/Winkel/Geschwindigkeit Spieler 16
 * 89..93               Position/Winkel/Geschwindigkeit Spieler 18
 * 94..98               Position/Winkel/Geschwindigkeit Spieler 18
 * 99..103              Position/Winkel/Geschwindigkeit Spieler 19
 * 104..108             Position/Winkel/Geschwindigkeit Spieler 20
 * 109..113             Position/Winkel/Geschwindigkeit Spieler 21
 */
void
AttackPositioningLearnSituationTable::
  createSingleRandomSituation( float *(&tmp), int numPlayers, bool ball )
{
  if (ivNumCols == 0)
  {
    int numPlayers = 22;
  	if (ball) ivNumCols += 4;
  	ivNumCols += numPlayers * 5;
  }
  if (tmp==0) 
    tmp = new float[ivNumCols];
    
    
  //CONSIDER ALL PLAYERS
  for (int spieler=0; spieler < numPlayers; spieler++ )
  {
cout<<"spieler "<<spieler<<" ..."<<endl;
    playerPositionAreas * ppa = (spieler < 11) ? ivpTeam1PositionAreas
                                               : ivpTeam2PositionAreas;
    int spIdx = (spieler < 11) ? spieler : spieler - 11;

    float avgX = ppa[spIdx].ivAverageX * FIELD_BORDER_X,
          avgY = ppa[spIdx].ivAverageY * FIELD_BORDER_Y,
          dX   = ppa[spIdx].ivMaximalXDeviation * FIELD_BORDER_X,
          dY   = ppa[spIdx].ivMaximalYDeviation * FIELD_BORDER_Y;
cout<<avgX<<" - "<<avgY<<" - "<<dX<<" - "<<dY<<endl;
    int baseIndex = 4 + (spieler * 5);

    //player position
    tmp[ baseIndex + 0 ] = this->getRandomPlayerPosWithinRange
                                 ( avgX - dX, avgX + dX );
    tmp[ baseIndex + 1 ] = this->getRandomPlayerPosWithinRange
                                 ( avgY - dY, avgY + dY ); 
    
    //player angle
    tmp[ baseIndex + 2 ] = this->getRandomPlayerAngle();
    
    //player velocity => no veloicties!
    tmp[ baseIndex + 3 ] = 0.0;
    tmp[ baseIndex + 4 ] = 0.0;
    
  }

  //CONSIDER BALL
  //determine ball holder
  int ballPossessingPlayer = (int)(TG_MY_RANDOM_NUMBER * 10.0) + 1;
  int ballPossessingPlayerBaseIndex = 4 + (ballPossessingPlayer * 5);
  //ball position
  tmp[ 0 ] = tmp[ ballPossessingPlayerBaseIndex + 0 ] + 0.5;
  tmp[ 1 ] = tmp[ ballPossessingPlayerBaseIndex + 1 ] + 0.0;
  //ball veloicty
  tmp[ 2 ] = 0.0;
  tmp[ 3 ] = 0.0;

  //DETERMINE NEAREST OPPONENT TO BALL and set him away!
  for (int spieler=11; spieler < numPlayers; spieler++ )
  {
    int baseIndex = 4 + (spieler * 5);
    while (   pow( tmp[0] - tmp[baseIndex+0], 2) 
            + pow( tmp[1] - tmp[baseIndex+1], 2) < 4.0*4.0 )
    {
      tmp[baseIndex+0] += 1.0;
    }
  }

  //ERINNERUNG:
  //Zu setzende Positionen in X- wie Y-Richtung muessen um die 
  //Ball-/Spielergeschwindigkeit dekrementiert werden, da im ersten
  //fuer den Spieler wahrnehmbaren Zyklus die Geschwindigkeiten
  //auf die Positionen durch den Server addiert worden sind!
  //Da hier alle Geschwindigkeiten 0.0 sind, braucht jene Subtraktion
  //nicht zu erfolgen.
}

//---------------------------------------------------------------------------
// METHODE init()
//---------------------------------------------------------------------------
/**
 * Diese Method liest aus der Konfigurationsdatei coach.conf die 
 * Informationen ein, nach welchen Kriterien die zu erstellenden
 * Ausgangssituationen zu kreieren sind.
 * 
 * Insbesondere wird fuer jeden der 22 Spieler auf dem Platz dessen 
 * Durchschnittsposition eingelesen sowie ein (rechteckiger) Bereich
 * um jene Position. Innerhalb jenes Vierecks wird letztlich die zu
 * ermittelnde Startposition des Spielers sein.
 * Die zugehoerigen Daten werden in den Instanzenvariablen 
 * ivpTeam1PositionAreas und ivpTeam2PositionAreas gespeichert.
 */
bool
AttackPositioningLearnSituationTable::init()
{
  Tribots::ConfigReader vp(4);
  vp.append_from_file( "coach.conf" );
  char team1String[100], team2String[100];
  team1String[0] = '\0'; team2String[0] = '\0'; 
  string specString;
  vp.get(
    "AttackPositioningLearnSituationTable::usedRandomSituationSpecification",
    specString );
  sprintf(team1String, "%s::team1", specString.c_str() );
  sprintf(team2String, "%s::team2", specString.c_str() );
  vector<float> randomPositionAreasTeam1, randomPositionAreasTeam2;
  vp.get( team1String, randomPositionAreasTeam1 );
  vp.get( team2String, randomPositionAreasTeam2 );
  if (   specString.size() == 0 || team1String == NULL || team2String == NULL
      || randomPositionAreasTeam1.size() < 11*4
      || randomPositionAreasTeam2.size() < 11*4 )
  {
    LOG_FLD(0, "AttackPositioningLearnSituationTable: ERROR during "
      <<"reading of coach.conf. Wrong specification of random situations."
      <<flush);
    exit(0);
  }
  for (int i=0; i<11; i++)
  {
    ivpTeam1PositionAreas[i].ivAverageX = randomPositionAreasTeam1[4*i + 0];
    ivpTeam1PositionAreas[i].ivAverageY = randomPositionAreasTeam1[4*i + 1];
    ivpTeam1PositionAreas[i]
      .ivMaximalXDeviation = randomPositionAreasTeam1[4*i + 2];
    ivpTeam1PositionAreas[i]
      .ivMaximalYDeviation = randomPositionAreasTeam1[4*i + 3];

    ivpTeam2PositionAreas[i].ivAverageX = randomPositionAreasTeam2[4*i + 0];
    ivpTeam2PositionAreas[i].ivAverageY = randomPositionAreasTeam2[4*i + 1];
    ivpTeam2PositionAreas[i]
      .ivMaximalXDeviation = randomPositionAreasTeam2[4*i + 2];
    ivpTeam2PositionAreas[i]
      .ivMaximalYDeviation = randomPositionAreasTeam2[4*i + 3];
  }
  return true;
}







