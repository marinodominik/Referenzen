/* Author: Manuel "Sputnick" Nickschas, 11/2001
 *
 * The class Field and the other classes declared in this file
 * hold all data describing the field for the coach.
 * (e.g. the position of objects on the field, current playmode
 * and similar stuff).
 *
 */

#ifndef _FIELD_H_
#define _FIELD_H_

#include "defs.h"
#include "param.h"
#include "angle.h"
#include "messages.h"

class Object {
 public:
  bool alive;
  double pos_x,pos_y,vel_x,vel_y;
  
  Vector pos() const { return Vector(pos_x,pos_y);};
  Vector vel() const { return Vector(vel_x,vel_y);};
  void setPos(double px,double py) { pos_x=px;pos_y=py;};
  void setVel(double vx,double vy) { vel_x=vx;vel_y=vy;};

 protected:
  Object() {alive=false;}
};

class Player : public Object {
 public:
  ANGLE bodyAng,neckAng,pointtoDir;
  bool goalie,pointtoFlag;
  int staminaCapacityBound;
  int number,type;
  int possibleTypes[MAX_PLAYER_TYPES];

  Player(double px=0,double py=0,double vx=0,double vy=0,
	 ANGLE bodyAng=ANGLE(0),ANGLE neckAng=ANGLE(0),int number=0,ANGLE pointtoDir=ANGLE(0),bool pointtoFlag=false);
};
  
class Ball : public Object {

};

class Field {
 private:
  int currentTime;
  int currentPM;
  char myTeamName[40];
  char hisTeamName[40];
  friend class MsgTeamNames;
 public:
  PlayerType plType[MAX_PLAYER_TYPES];
  Player myTeam[TEAM_SIZE],hisTeam[TEAM_SIZE];
  
  Ball ball;
  
  Field();

  struct {
    int last[MAX_SAY_TYPES];
    int lastInPlayOn[MAX_SAY_TYPES];
    bool ack[MAX_SAY_TYPES];
  } sayInfo;
  
  long getTime();
  void setTime(long time);
  int getPM();
  void setPM(int PM);
  const char *getMyTeamName();
  const char *getHisTeamName();
  bool  getPlayerByNumber( int number, int team, Player *& player );
  bool  getPlayerIndexByNumber( int number, int team, int & playerIndex );

  void visState();
};







#endif
