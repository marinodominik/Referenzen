#include "state.h"

#include "../modules/mod_direct_opponent_assignment.h"

#include <sstream>

const int 
State::cvcNumberOfStateComponents = 8;
const int 
State::cvcStateComponents[] = { BALL_POSITION,
                                BALL_VELOCITY,
                                MY_DEFENDERS_POSITIONS_FLTR,
                                MY_MIDFIELDERS_POSITIONS_FLTR,
                                MY_ATTACKERS_POSITIONS_FLTR,
                                HIS_GOALIE_POSITION,
                                HIS_DEFENDERS_POSITIONS_FLTR,
                                HIS_MIDFIELDERS_POSITIONS_FLTR };

//============================================================================
// KONSTRUKTOREN
//============================================================================

State::State()
{
  ivSize = 0;
}

State::State( const State & state )
{
  ivSize = state.ivSize;
  if (ivSize > MAX_NUMBER_OF_STATE_VARIABLES)
  {
    LOG_FLD(0, "ERROR: State: Feature "<<"vector size exceeds maximum.");
    ivSize = MAX_NUMBER_OF_STATE_VARIABLES;
  }
  for (int i=0; i<ivSize; i++)
    ivFeatures[i] = state.ivFeatures[i];
}

State::State( const vector<float> & vectorOfFloats )
{
  ivSize = (int)vectorOfFloats.size();
  if (ivSize > MAX_NUMBER_OF_STATE_VARIABLES)
  {
    LOG_FLD(0, "ERROR: State: Feature "<<"vector size exceeds maximum.");
    ivSize = MAX_NUMBER_OF_STATE_VARIABLES;
  }
  for (int i=0; i<ivSize; i++)
    ivFeatures[i] = vectorOfFloats[i];
}


//============================================================================
// STATISCHE METHODE: getCurrentState
//============================================================================
/**
 * Diese Methode ermittelt den aktuellen Zustand und speichert ihn im per
 * Referenz uebergebenen Objekt.
 * 
 * Die Spezifikation, was einen Zustand umfasst erfolgt in state.h.
 *
 */
void
State::getCurrentState(State & s)
{
  s.ivSize = 0;
  vector<int> localHisDefenders;
  for (int i=0; i<cvcNumberOfStateComponents; i++)
  {
    switch ( cvcStateComponents[i] )
    {
      case BALL_POSITION:
      {
        s.ivFeatures[s.ivSize] = fld.ball.pos_x;
        s.ivSize ++ ;
        s.ivFeatures[s.ivSize] = fld.ball.pos_y;
        s.ivSize ++ ;
        break;
      }
      case BALL_VELOCITY:
      {
        s.ivFeatures[s.ivSize] = fld.ball.vel_x;
        s.ivSize ++ ;
        s.ivFeatures[s.ivSize] = fld.ball.vel_y;
        s.ivSize ++ ;
        break;
      }
      case MY_ATTACKERS_POSITIONS_FLTR:
      {
        vector<int> v = getMyAttackerIndizesFLTR();
        for (unsigned int j=0; j<v.size(); j++)
        {
          s.ivFeatures[s.ivSize] = fld.myTeam[ v[j] ].pos_x;
          s.ivSize ++ ;
          s.ivFeatures[s.ivSize] = fld.myTeam[ v[j] ].pos_y;
          s.ivSize ++ ;
        }
        break;
      }
      case MY_MIDFIELDERS_POSITIONS_FLTR:
      {
        vector<int> v = getMyMidfielderIndizesFLTR();
        for (unsigned int j=0; j<v.size(); j++)
        {
          s.ivFeatures[s.ivSize] = fld.myTeam[ v[j] ].pos_x;
          s.ivSize ++ ;
          s.ivFeatures[s.ivSize] = fld.myTeam[ v[j] ].pos_y;
          s.ivSize ++ ;
        }
        break;
      }
      case MY_DEFENDERS_POSITIONS_FLTR:
      {
        vector<int> v = getMyDefenderIndizesFLTR();
        for (unsigned int j=0; j<v.size(); j++)
        {
          s.ivFeatures[s.ivSize] = fld.myTeam[ v[j] ].pos_x;
          s.ivSize ++ ;
          s.ivFeatures[s.ivSize] = fld.myTeam[ v[j] ].pos_y;
          s.ivSize ++ ;
        }
        break;
      }
      case HIS_MIDFIELDERS_POSITIONS_FLTR:
      {
        vector<int> v = getHisMidfielderIndizesFLTR( localHisDefenders );
        for (unsigned int j=0; j<v.size(); j++)
        {
          s.ivFeatures[s.ivSize] = fld.hisTeam[ v[j] ].pos_x;
          s.ivSize ++ ;
          s.ivFeatures[s.ivSize] = fld.hisTeam[ v[j] ].pos_y;
          s.ivSize ++ ;
        }
        break;
      }
      case HIS_DEFENDERS_POSITIONS_FLTR:
      {
        vector<int> v = getHisDefenderIndizesFLTR();
        for (unsigned int j=0; j<v.size(); j++)
        {
          s.ivFeatures[s.ivSize] = fld.hisTeam[ v[j] ].pos_x;
          s.ivSize ++ ;
          s.ivFeatures[s.ivSize] = fld.hisTeam[ v[j] ].pos_y;
          s.ivSize ++ ;
          localHisDefenders.push_back( v[j] );
        }
        break;
      }
      case HIS_GOALIE_POSITION:
      {
        int idx = getHisGoalieIndex();
        s.ivFeatures[s.ivSize] = fld.hisTeam[ idx ].pos_x;
        s.ivSize ++ ;
        s.ivFeatures[s.ivSize] = fld.hisTeam[ idx ].pos_y;
        s.ivSize ++ ;
        break;
      }
    }    
  }
}

//============================================================================
// STATISCHE METHODE: getCurrentFullState
//============================================================================
/**
 * Diese Methode ermittelt den aktuellen kompletten Zustand und speichert ihn 
 * im per
 * Referenz uebergebenen Objekt.
 * 
 * Ein kompletter Zustand umfasst Ballposition und Geschwindigkeit (4) 
 * sowie die Positionen aller Mitspieler (22) und Gegenspieler (22); also
 * insgesamt 48 Werte.
 *
 */
void
State::getCurrentFullState( State & s)
{
  s.ivSize = 0;
  s.ivFeatures[s.ivSize] = fld.ball.pos_x;
  s.ivSize ++ ;
  s.ivFeatures[s.ivSize] = fld.ball.pos_y;
  s.ivSize ++ ;
  s.ivFeatures[s.ivSize] = fld.ball.vel_x;
  s.ivSize ++ ;
  s.ivFeatures[s.ivSize] = fld.ball.vel_y;
  s.ivSize ++ ;
  //my team
  for (int j=1; j<=TEAM_SIZE; j++)
  {
    Player * player;
    fld.getPlayerByNumber(j, my_TEAM, player);
    s.ivFeatures[s.ivSize] = player->pos_x;
    s.ivSize ++ ;
    s.ivFeatures[s.ivSize] = player->pos_y;
    s.ivSize ++ ;
  }
  //his team
  for (int j=1; j<=TEAM_SIZE; j++)
  {
    Player * player;
    fld.getPlayerByNumber(j, his_TEAM, player);
    s.ivFeatures[s.ivSize] = player->pos_x;
    s.ivSize ++ ;
    s.ivFeatures[s.ivSize] = player->pos_y;
    s.ivSize ++ ;
  }
}

void
State::getFeatureIndex(int id, int & min, int & max)
{

  max = -1;
  for (int i=0; i<cvcNumberOfStateComponents; i++)
  {
    switch (cvcStateComponents[id])
    {
      case BALL_POSITION:
      {
        min = max + 1;
        max = min + 1;
        break;
      }
      case BALL_VELOCITY:
      {
        min = max + 1;
        max = min + 1;
        break;
      }
      case MY_DEFENDERS_POSITIONS_FLTR:
      {
        min = max + 1;
        max = min + 5;
        break;
      }
      case MY_MIDFIELDERS_POSITIONS_FLTR:
      {
        min = max + 1;
        max = min + 5;
        break;
      }
      case MY_ATTACKERS_POSITIONS_FLTR:
      {
        min = max + 1;
        max = min + 5;
        break;
      }
      case HIS_GOALIE_POSITION:
      {
        min = max + 1;
        max = min + 1;
        break;
      }
      case HIS_DEFENDERS_POSITIONS_FLTR:
      {
        min = max + 1;
        max = min + 7;
        break;
      }
      case HIS_MIDFIELDERS_POSITIONS_FLTR:
      {
        min = max + 1;
        max = min + 5;
        break;
      }
    }
  }
}

//============================================================================
// STATISCHE METHODE: getMyAttackerIndizesFLTR
//============================================================================
/**
 * Diese Methode die Indizes der eigenen Angriffsspieler im fld-Objekt.
 * Zurueckgegeben wird ein Vektor, der diese Indizes enthaelt.
 *
 * Die Stuermer der Brainstormers tragen stets die Rueckennummern 9, 10 
 * und 11 (von links nach rechts, aus eigener Sicht).
 */
vector<int>
State::getMyAttackerIndizesFLTR()
{
  int playerIndex9, playerIndex10, playerIndex11;
  vector<int> returnValue;
  if (  fld.getPlayerIndexByNumber(  9, my_TEAM, playerIndex9 )
      & fld.getPlayerIndexByNumber( 10, my_TEAM, playerIndex10)
      & fld.getPlayerIndexByNumber( 11, my_TEAM, playerIndex11) )
  {
    returnValue.push_back( playerIndex9  );
    returnValue.push_back( playerIndex10 );
    returnValue.push_back( playerIndex11 );
  }
  else {
    LOG_FLD(0, "ERROR: Could not retrieve the indizes of my attackers.");
  }
  return returnValue;
}

//============================================================================
// STATISCHE METHODE: getMyMidfielderIndizesFLTR()
//============================================================================
/**
 * Diese Methode die Indizes der eigenen Mittelfeldspieler im fld-Objekt.
 * Zurueckgegeben wird ein Vektor, der diese Indizes enthaelt.
 *
 * Die Mittelfeldspieler der Brainstormers tragen stets die Rueckennummern 6, 
 * 7 und 11 (von links nach rechts, aus eigener Sicht).
 */
vector<int>
State::getMyMidfielderIndizesFLTR()
{
  int playerIndex6, playerIndex7, playerIndex8;
  vector<int> returnValue;
  if (  fld.getPlayerIndexByNumber(  6, my_TEAM, playerIndex6 )
      & fld.getPlayerIndexByNumber(  7, my_TEAM, playerIndex7 )
      & fld.getPlayerIndexByNumber(  8, my_TEAM, playerIndex8 ) )
  {
    returnValue.push_back( playerIndex6  );
    returnValue.push_back( playerIndex7  );
    returnValue.push_back( playerIndex8  );
  }
  else {
    LOG_FLD(0, "ERROR: Could not retrieve the indizes of my midfielders.");
  }
  return returnValue;
}

//============================================================================
// STATISCHE METHODE: getMyDefenderIndizesFLTR()
//============================================================================
/**
 * Diese Methode die Indizes der eigenen Abwehrspieler im fld-Objekt.
 * Zurueckgegeben wird ein Vektor, der diese Indizes enthaelt.
 *
 * Die Abwehrspieler der Brainstormers tragen stets die Rueckennummern 2, 
 * 3 und 5 (von links nach rechts, aus eigener Sicht).
 * 
 * Zur Information: Der Libero der Brainstormers traegt die Rueckennummern 4.
 */
vector<int>
State::getMyDefenderIndizesFLTR()
{
  int playerIndex2, playerIndex3, playerIndex5;
  vector<int> returnValue;
  if (  fld.getPlayerIndexByNumber(  2, my_TEAM, playerIndex2 )
      & fld.getPlayerIndexByNumber(  3, my_TEAM, playerIndex3 )
      & fld.getPlayerIndexByNumber(  5, my_TEAM, playerIndex5 ) )
  {
    returnValue.push_back( playerIndex2  );
    returnValue.push_back( playerIndex3  );
    returnValue.push_back( playerIndex5  );
  }
  else
  {
    LOG_FLD(0, "ERROR: Could not retrieve the indizes of my defenders.");
  }
  return returnValue;
}

//============================================================================
// STATISCHE METHODE: getHisMidfielderIndizesFLTR
//============================================================================
/**
 * Diese Methode ermittelt die Indizes der gegnerischen Mittelfeldspieler
 * im fld-Objekt.
 * Zurueckgegeben wird ein Vektor, der diese Indizes enthaelt.
 *
 * Es wird STETS angenommen, dass die gegnerische Mannschaft ueber GENAU 3
 * Mittelfeldspieler verfuegt.
 */
vector<int>
State::getHisMidfielderIndizesFLTR( vector<int> & hisDefenders )
{
  bool hisConsideredPlayers[TEAM_SIZE];
  for (int i=0; i<TEAM_SIZE; i++) 
    hisConsideredPlayers[i] = true;
  if (hisDefenders.size() == 0)
    hisDefenders = getHisDefenderIndizesFLTR();
  for (int i=0; i<TEAM_SIZE; i++) 
  {
    if ( fld.hisTeam[i].goalie )
      hisConsideredPlayers[i] = false;
    for (unsigned int j=0; j<hisDefenders.size(); j++)
      if ( i == hisDefenders[j])
        hisConsideredPlayers[i] = false;
  }  
  //determine his midfielders
  vector<double> highest3XValues;
  highest3XValues.push_back(-100);  highest3XValues.push_back(-100);
  highest3XValues.push_back(-100);  
  int smallestOf3HighestAt = 0;
  for (int i=0; i<TEAM_SIZE; i++) 
  {
    if ( hisConsideredPlayers[i] )
    {
      if ( fld.hisTeam[i].pos_x > highest3XValues[smallestOf3HighestAt] )
      {
        highest3XValues[smallestOf3HighestAt] = fld.hisTeam[i].pos_x;
        smallestOf3HighestAt = 0;
        for (int j=1; j<3; j++)
          if ( highest3XValues[j] < highest3XValues[smallestOf3HighestAt] )
            smallestOf3HighestAt = j;
      }
    }
  }
  for (int i=0; i<TEAM_SIZE; i++) 
  {
    if (hisConsideredPlayers[i])
    {
      if ( fld.hisTeam[i].pos_x < highest3XValues[smallestOf3HighestAt] )
        hisConsideredPlayers[i] = false;
    }
  }  
  
  //sort his midfielders from left to right (from his point of view)
  // -> at the beginning of the vector returned there must be the defender
  //    with the most negative y position
  vector<int> consideredIndizes;
  for (int i=0; i<TEAM_SIZE; i++) 
    if (hisConsideredPlayers[i])
      consideredIndizes.push_back( i );
  if (consideredIndizes.size() != 3)
  {
    LOG_FLD(0,"ERROR: Could not determine 3 opponent midfielders.");
    exit(0);
  }
  sortHisPlayerIndizesFromLeftToRight( consideredIndizes );

  stringstream s;
  for (unsigned int i=0; i<consideredIndizes.size(); i++)
    s <<" "<< consideredIndizes[i] << "("<<fld.hisTeam[consideredIndizes[i]].number<<")";
  LOG_FLD(0,"State:getHisMidfielderIndizesFLTR: His midfielders are"<<s.str());

  return consideredIndizes;
}

//============================================================================
// STATISCHE METHODE: getHisDefenderIndizesFLTR
//============================================================================
/**
 * Diese Methode ermittelt die Indizes der gegnerischen Verteidiger 
 * im fld-Objekt.
 * Zurueckgegeben wird ein Vektor, der diese Indizes enthaelt.
 *
 * Es wird STETS angenommen, dass die gegnerische Mannschaft ueber GENAU 4
 * Abwehrspieler verfuegt.
 */
vector<int>
State::getHisDefenderIndizesFLTR()
{
  bool hisConsideredPlayers[TEAM_SIZE];
  for (int i=0; i<TEAM_SIZE; i++) 
    if ( fld.hisTeam[i].goalie )
      hisConsideredPlayers[i] = false;
    else
      hisConsideredPlayers[i] = true;
  
  //determine his defenders
  vector<double> highest4XValues;
  highest4XValues.push_back(-100);  highest4XValues.push_back(-100);
  highest4XValues.push_back(-100);  highest4XValues.push_back(-100);
  int smallestOf4HighestAt = 0;
  for (int i=0; i<TEAM_SIZE; i++) 
  {
    if ( hisConsideredPlayers[i] )
    {
      if ( fld.hisTeam[i].pos_x > highest4XValues[smallestOf4HighestAt] )
      {
        highest4XValues[smallestOf4HighestAt] = fld.hisTeam[i].pos_x;
        smallestOf4HighestAt = 0;
        for (int j=1; j<4; j++)
          if ( highest4XValues[j] < highest4XValues[smallestOf4HighestAt] )
            smallestOf4HighestAt = j;
      }
    }
  }
  for (int i=0; i<TEAM_SIZE; i++) 
  {
    if (hisConsideredPlayers[i])
    {
      if ( fld.hisTeam[i].pos_x < highest4XValues[smallestOf4HighestAt] )
        hisConsideredPlayers[i] = false;
    }
  }  
  
  //sort his defenders from left to right (from his point of view)
  // -> at the beginning of the vector returned there must be the defender
  //    with the most negative y position
  vector<int> consideredIndizes;
  for (int i=0; i<TEAM_SIZE; i++) 
    if (hisConsideredPlayers[i])
      consideredIndizes.push_back( i );
  if (consideredIndizes.size() != 4)
  {
    LOG_FLD(0,"ERROR: Could not determine 4 opponent defenders.");
    exit(0);
  }
  sortHisPlayerIndizesFromLeftToRight( consideredIndizes );

  stringstream s;
  for (unsigned int i=0; i<consideredIndizes.size(); i++)
    s <<" "<< consideredIndizes[i] << "("<<fld.hisTeam[consideredIndizes[i]].number<<")";
  LOG_FLD(0,"State:getHisDefenderIndizesFLTR: His defenders are"<<s.str());

  return consideredIndizes;
}

//============================================================================
// STATISCHE METHODE: getHisGoalieIndex
//============================================================================
/**
 * Diese Methode ermittelt den Index des gegnerischen Torwarts im fld-Objekt.
 * Zurueckgegeben wird ein Integer, der diesem Index entspricht.
 *
 */
int
State::getHisGoalieIndex()
{
  for (int i=0; i<TEAM_SIZE; i++)
    if ( fld.hisTeam[i].goalie )
      return i;

  LOG_FLD(0,"ERROR: Could not determine his goalie's index.");
  return 0;
}

//============================================================================
// STATISCHE METHODE: sortHisPlayerIndizesFromLeftToRight
//============================================================================
/**
 * Diese Methode macht, was ihr Name verheisst: Sie sortiert die in einem
 * Vektor uebergebene Menge Spielerindizes (indizieren Elemente in
 * fld.hisTeam) aufsteigende nach y-Positionen.
 */
void
State::sortHisPlayerIndizesFromLeftToRight( vector<int> & plIndizes )
{
  for (unsigned int i=0; i<plIndizes.size()-1; i++)
    for ( unsigned int j=0; j<((plIndizes.size()-1)-i); j++)
      if (   fld.hisTeam[ plIndizes[ j ] ].pos_y
           > fld.hisTeam[ plIndizes[j+1] ].pos_y )
      {  //swap
         int tmp = plIndizes[j];
         plIndizes[j] = plIndizes[j+1];
         plIndizes[j+1] = tmp;
      }
}


//############################################################################
// CLASS   E P I S O D E
//############################################################################

void
Episode::addNextState()
{
  State s, fs;
  State::getCurrentState( s );
  ivVStates.push_back( s );
  State::getCurrentFullState( fs );
  ivVFullStates.push_back( fs );
}

void
Episode::addSuggestedImmediateReward(float r)
{
  ivVRewards.push_back( r );
}

bool
Episode::ballKickableForOpponent(int t)
{
  if (t<0 || t>=(int)ivVStates.size()) {
    LOG_FLD(0,"ERROR: Index exceeds bounds of state vector.");
  }
//  Vector ballPos = getBallPosition( t );

  //check if ball was kickable for a team player
  Vector ballPosition = getBallPosition( t );
  int minIndex = 26,
      maxIndex = 47;
  for (int index = minIndex; index <= maxIndex; index += 2)
  {
    Vector playerPosition;
    playerPosition.setX( ivVFullStates[t].ivFeatures[index] );
    playerPosition.setY( ivVFullStates[t].ivFeatures[index+1] );
    if ( playerPosition.distance( ballPosition ) <= KICK_RANGE )
      return true;
  }
    
  //sonst-fall: ball im kickbereich keines mitspielers
  return false;
}

bool
Episode::ballKickableForTeammate(int t)
{
  if (t<0 || t>=(int)ivVStates.size()) {
    LOG_FLD(0,"ERROR: Index exceeds bounds of state vector.");
  }
//  Vector ballPos = getBallPosition( t );

  //check if ball was kickable for a team player
  Vector ballPosition = getBallPosition( t );
  int minIndex = 4,
      maxIndex = 25;
  for (int index = minIndex; index <= maxIndex; index += 2)
  {
    Vector playerPosition;
    playerPosition.setX( ivVFullStates[t].ivFeatures[index] );
    playerPosition.setY( ivVFullStates[t].ivFeatures[index+1] );
    if ( playerPosition.distance( ballPosition ) <= KICK_RANGE )
      return true;
  }
    
  //sonst-fall: ball im kickbereich keines mitspielers
  return false;
}

double
Episode::ballVelocityAtTime(int t)
{
  if (t<1 || t>=(int)ivVFullStates.size()) {
    LOG_FLD(0,"ERROR: Index exceeds bounds of state vector.");
  }
  Vector ballVel;
  ballVel.setX( ivVFullStates[t].ivFeatures[ 0 ] );
  ballVel.setY( ivVFullStates[t].ivFeatures[ 1 ] );
  return ballVel.norm();
}

double
Episode::getAverageDistanceToBall( int t )
{
  if (t<1 || t>=(int)ivVFullStates.size()) {
    LOG_FLD(0,"ERROR: Index exceeds bounds of state vector.");
  }
  int baseIndexMyPlayers  = 4,
      distToBallCnt = 0;
  double distToBallSum = 0.0;
  for (int myPlayerNumber = 0; myPlayerNumber < 11; myPlayerNumber++)
  {
    Vector myPos;
    myPos.setX( ivVFullStates[t]
               .ivFeatures[ baseIndexMyPlayers + 2*myPlayerNumber ] );
    myPos.setY( ivVFullStates[t]
               .ivFeatures[ baseIndexMyPlayers + 2*myPlayerNumber + 1 ] );
    //achtung: hier wird implizit angenommen, dass unsere spieler in fld.myTeam
    //entsprechend ihrer rueckennummern sortiert sind
    Vector ballPos;
    ballPos.setX( ivVFullStates[t].ivFeatures[ 0 ] );
    ballPos.setY( ivVFullStates[t].ivFeatures[ 1 ] );
    //distance
    distToBallSum += myPos.distance( ballPos );
  }
  if (distToBallCnt>0) 
    return distToBallSum / (float)distToBallCnt;
  return 0.0;
}

double
Episode::getAverageDistanceToDirectOpponents( int t )
{
  if (t<1 || t>=(int)ivVFullStates.size()) {
    LOG_FLD(0,"ERROR: Index exceeds bounds of state vector.");
  }
  int baseIndexMyPlayers  = 4,
      baseIndexHisPlayers = 26,
      distToDOCnt = 0;
  double distToDOSum = 0.0;
  for (int myPlayerNumber = 0; myPlayerNumber < 11; myPlayerNumber++)
  {
    Vector myPos;
    myPos.setX( ivVFullStates[t]
               .ivFeatures[ baseIndexMyPlayers + 2*myPlayerNumber ] );
    myPos.setY( ivVFullStates[t]
               .ivFeatures[ baseIndexMyPlayers + 2*myPlayerNumber + 1 ] );
    //achtung: hier wird implizit angenommen, dass unsere spieler in fld.myTeam
    //entsprechend ihrer rueckennummern sortiert sind
    int directOppNumber 
      = ModDirectOpponentAssignment::cvDirectOpponentAssignment[myPlayerNumber];
    directOppNumber -- ; //zaehlung von 0..10
    Vector dirOppPos;
    dirOppPos.setX( ivVFullStates[t]
                   .ivFeatures[ baseIndexHisPlayers + 2*directOppNumber ] );
    dirOppPos.setY( ivVFullStates[t]
                   .ivFeatures[ baseIndexHisPlayers + 2*directOppNumber + 1 ] );
    //distance
    distToDOSum += myPos.distance( dirOppPos );
  }
  if (distToDOCnt>0) 
    return distToDOSum / (float)distToDOCnt;
  return 0.0;
}

double
Episode::getAverageWayGoneByTeammates( int t )
{
  if (t<1 || t>=(int)ivVFullStates.size()) {
    LOG_FLD(0,"ERROR: Index exceeds bounds of state vector.");
  }
  double waySum = 0.0;
  int wayCnt = 0;
  //consider own defenders
  int minIndex = 4,
      maxIndex = 25;
  for (int index = minIndex; index <= maxIndex; index+=2 )
  {
    Vector p0, p1;
    p0.setX( ivVStates[t-1].ivFeatures[ index ] );
    p0.setY( ivVStates[t-1].ivFeatures[ index + 1 ] );
    p1.setX( ivVStates[t].ivFeatures[ index ] );
    p1.setY( ivVStates[t].ivFeatures[ index + 1 ] );
    double dist = p0.distance( p1 );
    if (dist > 1.2) waySum += 0.6;
    else            waySum += dist;
    wayCnt ++ ;
  }
  if (wayCnt==0) return 0.0;
  return waySum / (float)wayCnt;
}

Vector
Episode::getBallPosition(int t)
{
  if (t<0 || t>=(int)ivVFullStates.size()) {
    LOG_FLD(0,"ERROR: Index exceeds bounds of full-state vector.");
  }
  Vector v;
  v.setX( ivVFullStates[t].ivFeatures[0] );
  v.setY( ivVFullStates[t].ivFeatures[1] );
  return v;
}

int
Episode::getLength()
{
  return (int)ivVStates.size();
}

int
Episode::getNumberOfOpponentPlayersAroundPoint( int t,
                                                Vector & pos,
                                                double    radius )
{
  if (t<0 || t>=(int)ivVFullStates.size()) {
    LOG_FLD(0,"ERROR: Index exceeds bounds of full-state vector.");
  }
  int playersNearPos = 0;
  for (int i=26; i<48; i+=2)
  {
    Vector plPos( ivVFullStates[t].ivFeatures[i],
                  ivVFullStates[t].ivFeatures[i+1] );
    double dist = plPos.distance( pos );
    if ( dist < radius )
      playersNearPos ++ ;
  }  
  return playersNearPos;  
}

int
Episode::getNumberOfTeammatePlayersAroundPoint( int            t,
                                                const Vector & pos,
                                                double          radius )
{
  if (t<0 || t>=(int)ivVFullStates.size()) {
    LOG_FLD(0,"ERROR: Index exceeds bounds of full-state vector.");
  }
  int playersNearPos = 0;
  for (int i=4; i<26; i+=2)
  {
    Vector plPos( ivVFullStates[t].ivFeatures[i],
                  ivVFullStates[t].ivFeatures[i+1] );
    double dist = plPos.distance( pos );
    if ( dist < radius )
      playersNearPos ++ ;
  }  
  return playersNearPos;  
}

double
Episode::getDistanceOfNearestPlayerToPos(int t, Vector & pos)
{
  if (t<0 || t>=(int)ivVFullStates.size()) {
    LOG_FLD(0,"ERROR: Index exceeds bounds of full-state vector.");
  }
  double v = INT_MAX;
  for (int i=4; i<26; i+=2)
  {
    Vector plPos( ivVFullStates[t].ivFeatures[i],
                  ivVFullStates[t].ivFeatures[i+1] );
    double dist = plPos.distance( pos );
    if ( dist < v )
      v = dist;
  }
  return v;
}

bool
Episode::getTwoTeammatesControlBall( int t )
{
  if (t<0 || t>=(int)ivVStates.size()) {
    LOG_FLD(0,"ERROR: Index exceeds bounds of state vector.");
  }
//  Vector ballPos = getBallPosition( t );
  //check if ball was kickable for a team player
  Vector ballPosition = getBallPosition( t );
  int numberOfPlayersWithBallInKickRange = 0;
  int minIndex = 4,
      maxIndex = 25;
  for (int index = minIndex; index <= maxIndex; index += 2)
  {
    Vector playerPosition;
    playerPosition.setX( ivVFullStates[t].ivFeatures[index] );
    playerPosition.setY( ivVFullStates[t].ivFeatures[index+1] );
    if ( playerPosition.distance( ballPosition ) <= KICK_RANGE )
      numberOfPlayersWithBallInKickRange ++ ;
  }
  if ( numberOfPlayersWithBallInKickRange > 1 )
    return true;
  else
    return false;
}

void
Episode::reset()
{
  ivVStates.erase( ivVStates.begin(), ivVStates.end());
  ivVFullStates.erase( ivVFullStates.begin(), ivVFullStates.end());
  ivVRewards.erase( ivVRewards.begin(), ivVRewards.end());
}

string
Episode::toStringReward(int t)
{
  if (t<0 || t>=(int)ivVRewards.size())
  {
    LOG_FLD(0,"ERROR: Index exceeds bounds of full-state vector.");
    return string("ERROR: index t beyond range");
  }
  stringstream s;
  s << ivVRewards[t];
  return s.str();
}

string
Episode::toStringState(int t)
{
  if (t<0 || t>=(int)ivVStates.size())
  {
    LOG_FLD(0,"ERROR: Index exceeds bounds of full-state vector.");
    return string("ERROR: index t beyond range");
  }
  State & state = ivVStates[t];
  stringstream s;
  for (int i=0; i<state.ivSize; i++)
    s << state.ivFeatures[i] << " ";
  return s.str();
}
