/* Author: Manuel Nickschas
 *
 * This is a demo module
 *
 */

#include "mod_demo_train.h"
#include <stdio.h>

/** This will be called after the rest of the coach has been initialised
    and the connection to the server has been established.
    If waitForTeam is set, you may also assume that all own players are already
    on the field. Put all module specific initialisation here.
*/
bool ModDemoTrain::init(int argc,char **argv) {
  cyc_cnt=0;
  cyc_avg=0;
  diff_avg=0;
  runs=-1;
  misses=0;
  return true;
}

/** The framework will call this routine just before the connection to the server
    goes down. Up to this point, all data structures of the main coach are still intact.
*/
bool ModDemoTrain::destroy() {
  using namespace std;

  cerr << "\n\n***** STATISTIK *****"
       << "\nplayer_type:     "<<fld.myTeam[0].type
       << "\nkickable_margin: "<<fld.plType[fld.myTeam[0].type].kickable_margin
       << "\nkick_rand:       "<<fld.plType[fld.myTeam[0].type].kick_rand
       << "\nAvg Zyklen:      "<<cyc_avg
       << "\nAvg Abweichung:  "<<diff_avg
       << "\nFehlversuche:    "<<misses
       << "\n";
  
  return true;
}

/** Similar to the BS2kAgent behave routine, this one here will be called every cycle.
    It is currently synched to the see_global messages. Put code in here that
    should be called once per cycle just after the visual information has been updated.
*/
bool ModDemoTrain::behave() {
  using namespace std;
#if 1
  if(fld.getTime()<=0) {
    moveOwnPlayer(1,0,0,0,0);
    moveBall(.8,0,0,0);
    cyc_cnt=0;
    if(fld.myTeam[0].type==0) return true;
  }
  if(fld.getPM()!=PM_play_on) {
    MSG::sendMsg(MSG::MSG_CHANGE_MODE,PM_play_on);
  }
  cyc_cnt++;
  if(cyc_cnt>20) {
    misses++;
    cyc_cnt=0;
    moveBall(0.8,0,0,0);
    moveOwnPlayer(1,0,0,0,0,0);
    return true;
  }
  if(fld.ball.vel().norm()<.01) cyc_cnt=0;
  //if(fld.getTime()==1) {
  //  moveOwnPlayer(1,0,0,0,0,0);
  //  moveBall(0.8,0,0,0);
  //  cyc_cnt=0;
  //}
  //if((fld.myTeam[0].pos()-Vector(0,0)).norm()>.01) moveOwnPlayer(1,0,0,0,0);
  if((fld.myTeam[0].pos()-fld.ball.pos()).norm()>15) {
    if(++runs>0) {
      cerr << "\n-----< Run "<<runs<<">-----"
	   << "\nZyklen benoetigt: "<<cyc_cnt
	   << "\nAbweichung (y)  : "<<fabs(fld.ball.pos().getY());
      cyc_avg=(1-1.0/runs)*cyc_avg+(double)cyc_cnt/runs;
      diff_avg=(1-1.0/runs)*diff_avg+fabs(fld.ball.pos().getY())/runs;
    }
    cyc_cnt=0;
    moveBall(0.8,0,0,0);
    moveOwnPlayer(1,0,0,0,0,0);
  }
  return true;
#else
  if(fld.getTime()<=0) {
    moveOwnPlayer(1,0,10,0,0,0);
    moveOwnPlayer(2,0,-10,0,0,0);
    MSG::sendMsg(MSG::MSG_CHANGE_MODE,PM_play_on);
  }
  return true;


#endif
  //if(fld.getTime()==10) {
  //  MSG::sendMsg(MSG::MSG_CHANGE_MODE,PM_play_on);
  //  moveBall(10,10);
  //  moveOwnPlayer(1,20,20,PI,1,1);
  //}
  //return true;
}

/** Every time the referee sends a message, this function will be called. The parameter
    is true when the playmode has been changed. Note that there is a subtile difference
    between playmode changes and other referee messages...
*/
bool ModDemoTrain::onRefereeMessage(bool PMChange) {
  
  return true;
}

/** A string entered on the keyboard will be sent through this messages. If you process
    this string, you should return true. This will prevent sending the string to
    other modules as well. Return false if you don't process the message or if you want
    the string to be sent to other modules.
*/
bool ModDemoTrain::onKeyboardInput(const char *str) {

  return false;
}

/** Any hear message that does not come from the referee can be processed using this message.
    Unlike the keyboard input, a hear message will always be sent to all modules.
*/
bool ModDemoTrain::onHearMessage(const char *str) {

  return false;
}

/** This function will be called whenever a player type is changed. Note that player type
    changes can occur before the module is initialized, so you cannot be sure that you did not
    miss some changes before the beginning of the game.
    ownTeam is true, when the change happened in the own team. Remember that opponent player
    changes usually result in an unknown type (-1).

    This function will also be called, if ModAnalyse (or any other module...) makes new
    assumptions on the type of opponent players. You should then check what has changed
    in Player::possibleTypes[].
*/
bool ModDemoTrain::onChangePlayerType(bool ownTeam,int unum,int type) {

  return false;
}

void ModDemoTrain::moveBall(double x,double y,double velx,double vely) {
  MSG::sendMsg(MSG::MSG_MOVE,"ball",x,y,0,velx,vely);
}

void ModDemoTrain::moveOwnPlayer(int unum,double x,double y,double dir,double velx,double vely) {
  char buf[100];
  sprintf(buf,"player %s %d",fld.getMyTeamName(),unum);
  MSG::sendMsg(MSG::MSG_MOVE,buf,x,y,dir,velx,vely);
}

void ModDemoTrain::moveOppPlayer(int unum,double x,double y,double dir,double velx,double vely) {
  char buf[100];
  sprintf(buf,"player %s %d",fld.getHisTeamName(),unum);
  MSG::sendMsg(MSG::MSG_MOVE,buf,x,y,dir,velx,vely);
}


/* Don't forget this one... */
const char ModDemoTrain::modName[]="ModDemoTrain";
