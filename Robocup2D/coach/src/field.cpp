/* Author: Manuel "Sputnick" Nickschas, 11/2001
 *
 * See field.h for description.
 *
 */

#include "field.h"
#include "logger.h"
#include <stdlib.h>
#include <stdio.h>
#include <sstream>

/*********************************************
 * class Object
 ********************************************/



/********************************************
 * class Player
 *******************************************/

Player::Player(double px,double py,double vx,double vy,ANGLE bang,ANGLE nang,int nr,ANGLE pt_dir, bool pt_flag) {
  pos_x=px;pos_y=py;vel_x=vx;vel_y=vy;
  bodyAng=bang;neckAng=nang;pointtoDir=pt_dir;
  alive=false;type=0;pt_flag= pt_flag;
  staminaCapacityBound = 130600;
  for(int i=0;i<MAX_PLAYER_TYPES;i++) possibleTypes[i]=0;
}

/********************************************
 * class Field
 *******************************************/

Field::Field() {
  currentTime=0;
  currentPM=0;
  myTeamName[0]=0;hisTeamName[1]=0;
  for(int i=0;i<MAX_SAY_TYPES;i++) {
    sayInfo.last[i]=-1;sayInfo.lastInPlayOn[i]=-1;
    sayInfo.ack[i]=true;
  }  
}

void Field::setTime(long time) {currentTime=time;}

long Field::getTime() {return currentTime;}

void Field::setPM(int PM) {currentPM=PM;}

int Field::getPM() {return currentPM;}

const char *Field::getMyTeamName() {
  return myTeamName;
}

const char *Field::getHisTeamName() {
  return hisTeamName;
}

//============================================================================
// getPlayerByNumber
//============================================================================
/**
 * Zu einer gegebenen Rueckennummer gibt diese Methode den zugehoerigen
 * Spieler zurueck.
 */
bool  
Field::getPlayerByNumber( int number, int team, Player *& player )
{
  Player (*consideredTeam)[TEAM_SIZE];
  if (team == his_TEAM) consideredTeam = & fld.hisTeam;
  else 
    if (team == my_TEAM) consideredTeam = & fld.myTeam;
    else 
      return false;
  if (number < 1 || number > 11)
    return false;
  for (int i=0; i<TEAM_SIZE; i++)
    if ( (*consideredTeam)[i].number == number )
    {
      player =  &  (*consideredTeam)[i];
      break;
    }
  return true;
}

//============================================================================
// getPlayerIndexByNumber
//============================================================================
/**
 * Zu einer gegebenen Rueckennummer gibt diese Methode den zugehoerigen
 * Index des betreffenden Spielers innerhalb des objektinternen Feldes
 * alles Spieler zurueck.
 */
bool
Field::getPlayerIndexByNumber( int number, int team, int & playerIndex )
{
  Player (*consideredTeam)[TEAM_SIZE];
  if (team == his_TEAM) consideredTeam = & fld.hisTeam;
  else 
    if (team == my_TEAM) consideredTeam = & fld.myTeam;
    else 
      return false;
  if (number < 1 || number > 11)
    return false;
  for (int i=0; i<TEAM_SIZE; i++)
    if ( (*consideredTeam)[i].number == number )
    {
      playerIndex =  i;
      break;
    }
  return true;
}

	      
void Field::visState() {
return;
  std::ostringstream oss;
  Vector pos,vel;
  double bang;//,nang;

  if(ball.alive) {
    oss << C2D(ball.pos_x,ball.pos_y,0.9,"#550055");
    oss << L2D(ball.pos_x,ball.pos_y,ball.pos_x+5*ball.vel_x,ball.pos_y+5*ball.vel_y,"#550055");
  }
  
  for(int i=0;i<11;i++) {
    if(myTeam[i].alive) {
      pos=myTeam[i].pos();
      vel=myTeam[i].vel();
      bang=myTeam[i].bodyAng.get_value();
//      nang=myTeam[i].neckAng.get_value();
      oss << VC2D(pos,.7,"#0000ff");
      oss << L2D(pos.getX(),pos.getY(),pos.getX()+.7*cos(bang),pos.getY()+.7*sin(bang),"#0000ff");
      oss << VL2D(pos,pos+5*vel,"#0000ff");
    }
    if(hisTeam[i].alive) {
      pos=hisTeam[i].pos();
      vel=hisTeam[i].vel();
      bang=hisTeam[i].bodyAng.get_value();
//      nang=hisTeam[i].neckAng.get_value();
      oss << VC2D(pos,.7,"#ff0000");
      oss << L2D(pos.getX(),pos.getY(),pos.getX()+.7*cos(bang),pos.getY()+.7*sin(bang),"#ff0000");
      oss << VL2D(pos,pos+5*vel,"#ff0000");
    }
    if(oss.str().length()>1800) {
      oss << '\0';
      LOG_VIS(0,<< _2D << oss.str());
      oss.seekp(0);
    }
  }

  oss << '\0';
  LOG_VIS(0,<< _2D << oss.str());
}
