#ifndef _MDPSTATE_H_
#define _MDPSTATE_H_

#include <iostream>
/** 
MDP = Markov Decision Process

Die Schnittstelle verwendet bei allen Angaben und unabhaengig von der
wirklichen Spielseite des Spielers das unten angegeben Koordinatensystem. Der
Koordinatenursprung befindet sich in der Mitte des Feldes. Die tatsaechliche
Position der Tore und Linien koennen aus den  Angaben ueber Feldmasse abgeleitet
werden. Sie sind aber insofern irrelevant, als dass es bei dieser Schnittstelle
nur um Positionen der beweglichen Objekte geht, unabhaengig von der
tatsaechlichen Feldgroesse. Das gegnerische Tor befinden sich immer rechts, also
in Richtung der X Achse.

      /------------------------------------------------\
      |                       |                        |
      |                       |                        |
      |                       |                        |
      |-------\               |                /-------|
      |       |               |                |       |
      |       |               |                |       |
      |       |               O <- (0,0)       |       | Goal of the opponent
      |       |               |                |       |
      |       |               |                |       |
      |-------/               |                \-------|
      |                       |                        |
      |                       |                        |
      |                       |                        |
      \------------------------------------------------/


            ^ Y axis
           /|\
	    |
            |
     -------O-------->  X axis
	    |
  	    |
	    |



Winkel werden im Bogenmass angegeben, d.h. jeder Winkel ist aus dem Intervall
[0,2*Pi) 

            
            | 0.5 Pi
	    |
   Pi       |          0
     -------O--------> 
	    |
  	    |
	    | 1.5 Pi
*/

#include "globaldef.h"
#include "Vector.h"


class FVal { //Fuzzy Value
public:
   double v;  // value
   double p;  // probability of correctness
   double t;  // time of last not simulated update
   FVal();
   FVal(double value);
   FVal(double value,double pp);
   FVal(double value,double pp, double time);
   void operator =(const FVal &fval); 
   void set_zero_probability();
};


class FBall {
public:
   FVal pos_x;  // x position
   FVal pos_y;  // y position
   FVal vel_x;  // angle
   FVal vel_y;  // velocity
   FBall();
   void init(const FVal &_pos_x,const FVal &_pos_y);
   void init(const FVal &_pos_x,const FVal &_pos_y,const FVal &_vel_x,const FVal &_vel_y);
   Vector pos() const { return Vector(pos_x.v,pos_y.v);};
   Vector vel() const { return Vector(vel_x.v,vel_y.v);};
   int pos_seen_at() const { return (int)pos_x.t;};
   int vel_seen_at() const { return (int)vel_x.t;};
};

class FPlayer {
public:
   int alive;
   int number;   // player number
   FVal pos_x;  // x position
   FVal pos_y;  // y position
   FVal vel_x;  // vel_x
   FVal vel_y;  // velocity
   FVal ang;
 
   FVal neck_angle;  //gibt den neck_angle absolut an, also nicht relativ zum FPlayer.angle
   FVal stamina;
   FVal effort;
   FVal recovery;
   FPlayer();
   void init(int n,const FVal &_pos_x,const FVal &_pos_y);
   void init(int n,const FVal &_pos_x,const FVal &_pos_y,const FVal &_vel_x,const FVal &_vel_y);
   void init(int n,const FVal &_pos_x,const FVal &_pos_y,const FVal &_vel_x,const FVal &_vel_y,
	const FVal& _ang,const FVal &_stamina, const FVal& _effort, const FVal & _recovery, const FVal &_neck_angle);
   Vector pos() const { return Vector(pos_x.v,pos_y.v);};
   Vector vel() const { return Vector(vel_x.v,vel_y.v);};
   double neck_angle_rel() const ;
   double get_stamina() const {return stamina.v;};
   double stamina_valid() const {return (stamina.p>.5 ) ? stamina.p :0 ;};
   int pos_seen_at() const { return (int)(pos_x.t);};
   int vel_seen_at() const { return (int)(vel_x.t);};
};

class MDPstate {
 public:
  //const static int NUM_PLAYERS= ::NUM_PLAYERS;

  FPlayer *me;
  long ms_time_of_sb;
  int time_current;
  int time_of_last_update;
  char StateInfoString[2000]; //compact representation of state for messaging
  
  int play_mode;
  int my_team_score;
  int his_team_score;
   
  int my_goalie_number;
  int his_goalie_number;

  struct{
    double dir;
    double dist;
    int number;
    int seen_at;
    int number_of_sensed_opponents;
  } closest_attacker; // the closest player to me; If team is known, then only opponents are considered
  
  // ... more information values
  
  int view_angle;   //takes values WIDE,NORMAL,NARROW defined in types.h
  int view_quality; //tekes values HIGH,LOW defined in types.h
  struct{
    int heart_at;
    int sent_at;
    int from;
  } last_message;
  int last_successful_say_at;   // time, when my message was broadcast to others
  
  FBall ball;
  FPlayer my_team[NUM_PLAYERS];
  FPlayer his_team[NUM_PLAYERS];
  MDPstate();
  void reset();
  
  //this definitions are inside of MDPstate, because there is a lot of legacy code using MPDstate::PM_*
  const static int  PM_Unknown            = ::PM_Unknown;
  const static int  PM_my_BeforeKickOff   = ::PM_my_BeforeKickOff;
  const static int  PM_his_BeforeKickOff  = ::PM_his_BeforeKickOff;
  const static int  PM_TimeOver           = ::PM_TimeOver;
  const static int  PM_PlayOn             = ::PM_PlayOn;
  const static int  PM_my_KickOff         = ::PM_my_KickOff;
  const static int  PM_his_KickOff        = ::PM_his_KickOff;
  const static int  PM_my_KickIn          = ::PM_my_KickIn;
  const static int  PM_his_KickIn         = ::PM_his_KickIn;
  const static int  PM_my_FreeKick        = ::PM_my_FreeKick;
  const static int  PM_his_FreeKick       = ::PM_his_FreeKick;
  const static int  PM_my_CornerKick      = ::PM_my_CornerKick;
  const static int  PM_his_CornerKick     = ::PM_his_CornerKick;
  const static int  PM_my_GoalKick        = ::PM_my_GoalKick;
  const static int  PM_his_GoalKick       = ::PM_his_GoalKick;
  const static int  PM_my_AfterGoal       = ::PM_my_AfterGoal;
  const static int  PM_his_AfterGoal      = ::PM_his_AfterGoal;
  const static int  PM_Drop_Ball          = ::PM_Drop_Ball;
  const static int  PM_my_OffSideKick     = ::PM_my_OffSideKick;
  const static int  PM_his_OffSideKick    = ::PM_his_OffSideKick;
  const static int  PM_Half_Time          = ::PM_Half_Time;
  const static int  PM_Extended_Time      = ::PM_Extended_Time;
  const static int  PM_my_GoalieFreeKick  = ::PM_my_GoalieFreeKick;
  const static int  PM_his_GoalieFreeKick = ::PM_his_GoalieFreeKick;
  const static int  PM_MAX                = ::PM_MAX;
};

std::ostream& operator<< (std::ostream& o, const MDPstate& mdp) ;
std::ostream& operator<< (std::ostream& o, const FPlayer& fp) ;
std::ostream& operator<< (std::ostream& o, const FBall& fb) ;
std::ostream& operator<< (std::ostream& o, const FVal& fv) ;



#endif
