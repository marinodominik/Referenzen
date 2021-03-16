/* Author: FluffyBunny
 *
 * This is a module template
 *
 */

#include "mod_oot.h"
#include "random.h"
#include "messages.h"

#define KICK_RANGE_DEF 1.085

  

const char ModOOT::modName[]="ModOOT";

/** This will be called after the rest of the coach has been initialised
    and the connection to the server has been established.
    If waitForTeam is set, you may also assume that all own players are already
    on the field. Put all module specific initialisation here.
*/
bool ModOOT::init(int argc,char **argv) {
  cout << "Coach gestartet!";
 
  ValueParser vp(Options::coach_conf,"ModOOT");
  vp.get("teammate1",teammate1,2);
  vp.get("teammate2",teammate2,2);
  vp.get("teammate3",teammate3,2);
  vp.get("teammate4",teammate4,2);
  vp.get("teammate5",teammate5,2);
  vp.get("teammate6",teammate6,2);
  vp.get("teammate7",teammate7,2);
  vp.get("teammate8",teammate8,2);
  vp.get("teammate9",teammate9,2);
  vp.get("teammate10",teammate10,2);
  vp.get("teammate11",teammate11,2);
  vp.get("opponent1",opponent1,2);
  vp.get("opponent2",opponent2,2);
  vp.get("opponent3",opponent3,2);
  vp.get("opponent4",opponent4,2);
  vp.get("opponent5",opponent5,2);
  vp.get("opponent6",opponent6,2);
  vp.get("opponent7",opponent7,2);
  vp.get("opponent8",opponent8,2);
  vp.get("opponent9",opponent9,2);
  vp.get("opponent10",opponent10,2);
  vp.get("opponent11",opponent11,2);
  vp.get("ball",ball,2);
  vp.get("activeTeammateNumber", activeTeammateNumber,1);
  vp.get("activeTeammateIndex", activeTeammateIndex,activeTeammateNumber[0]);
  vp.get("opponentMode", opponentMode,1);
  for (int i=0; i<11;i++){
    activeTeammate[i]=false; 
  }
  for (int i=0; i<activeTeammateNumber[0];i++){
    activeTeammate[activeTeammateIndex[i]]=true; 
  }
  return true;
}

void ModOOT::save_pos() {
  for (int i = 1 ; i < 11 ; i++){
      if(!fld.myTeam[i].alive) continue;
      mate[i].setX( fld.myTeam[i].pos_x );
      mate[i].setY( fld.myTeam[i].pos_y );
      cout << mate[i];
   }

   /*for (int i = 1 ; i < 11 ; i++){
      if (!fld.hisTeam[i].alive) continue;
      opp[i].x = fld.hisTeam[i].pos_x;
      opp[i].y = fld.hisTeam[i].pos_y;
        cout << opp[i];
  }*/
}


void ModOOT::reset() {
  /*srand( (unsigned)time( NULL ) );
  int i = (int) rand()%4+1;*/

   moveOwnPlayer(0,teammate1[0], teammate1[1] , 0,0.0,0.0);
   moveOwnPlayer(1,teammate2[0], teammate2[1] , 0,0.0,0.0);
   moveOwnPlayer(2,teammate3[0], teammate3[1] , 0,0.0,0.0);
   moveOwnPlayer(3,teammate4[0], teammate4[1] , 0,0.0,0.0);
   moveOwnPlayer(4,teammate5[0], teammate5[1] , 0,0.0,0.0);
   moveOwnPlayer(5,teammate6[0], teammate6[1] , 0,0.0,0.0);
   moveOwnPlayer(6,teammate7[0], teammate7[1] , 0,0.0,0.0);
   moveOwnPlayer(7,teammate8[0], teammate8[1] , 0,0.0,0.0);
   moveOwnPlayer(8,teammate9[0], teammate9[1] , 0,0.0,0.0);
   moveOwnPlayer(9,teammate10[0], teammate10[1] , 0,0.0,0.0);
   moveOwnPlayer(10,teammate11[0], teammate11[1] , PI,0.0,0.0);


   moveOppPlayer(0,opponent1[0], opponent1[1] , 0,0.0,0.0);
   moveOppPlayer(1,opponent2[0], opponent2[1] , 0,0.0,0.0);
   moveOppPlayer(2,opponent3[0], opponent3[1] , 0,0.0,0.0);
   moveOppPlayer(3,opponent4[0], opponent4[1] , 0,0.0,0.0);
   moveOppPlayer(4,opponent5[0], opponent5[1] , 0,0.0,0.0);
   moveOppPlayer(5,opponent6[0], opponent6[1] , 0,0.0,0.0);
   moveOppPlayer(6,opponent7[0], opponent7[1] , 0,0.0,0.0);
   moveOppPlayer(7,opponent8[0], opponent8[1] , 0,0.0,0.0);
   moveOppPlayer(8,opponent9[0], opponent9[1] , 0,0.0,0.0);
   moveOppPlayer(9,opponent10[0], opponent10[1] , 0,0.0,0.0);
   moveOppPlayer(10,opponent11[0], opponent11[1] , 0,0.0,0.0);
   moveBall(ball[0], ball[1],0.,0.);
  /*for (int i = 1 ; i < 11 ; i++){

      moveOppPlayer(i, -mate[i].x+15, -mate[i].y, 0.0,0.0,0.0);
  }*/
  MSG::sendMsg(MSG::MSG_SAY, 6, "reset int");
  MSG::sendMsg(MSG::MSG_CHANGE_MODE, PM_play_on);
 
}

/** The framework will call this routine just before the connection to the server
    goes down. Up to this point, all data structures of the main coach are still intact.
*/
bool ModOOT::destroy() {

  return true;
}

/** Similar to the BS2kAgent behave routine, this one here will be called every cycle.
    It is currently synched to the see_global messages. Put code in here that
    should be called once per cycle just after the visual information has been updated.
*/
bool ModOOT::behave() {
  if (fld.getPM()!=2&&fld.getPM()!=0) reset();
 // cout << "behave" <<  fld.getPM();
  //if (fld.getTime()==1) save_pos();
  Vector player,ball;
  
  
  if (false/*opponentMode[0]==0*/){
  bool we_have_ball=false;
  for (int i = 0 ; i < 11 ; i++){ // do consider goalie
    if (activeTeammate[i]) {
      if(!fld.myTeam[i].alive) continue;
      player.setX( fld.myTeam[i].pos_x );
      player.setY( fld.myTeam[i].pos_y );
      ball.setX( fld.ball.pos_x );
      ball.setY( fld.ball.pos_y );
      if((player-ball).norm() <= KICK_RANGE_DEF){
        we_have_ball=true;
      }
    }
  }
  if(!we_have_ball){
    ball.setX( fld.ball.pos_x );
    ball.setY( fld.ball.pos_y );
    for (int i = 0 ; i < 11 ; i++){ // do consider goalie
    if(!fld.myTeam[i].alive) continue;
    player.setX( fld.myTeam[i].pos_x );
    player.setY( fld.myTeam[i].pos_y );
    ball.setX( fld.ball.pos_x );
    ball.setY( fld.ball.pos_y );
    if((player-ball).norm() <= KICK_RANGE_DEF){
     //cout << "\nOP_HAS_BALL" << flush;
	reset();
	return true;
    }
    }
    for (int i = 1 ; i < 11 ; i++){ // do consider goalie!!!!
      if (!fld.hisTeam[i].alive) continue;
      player.setX( fld.hisTeam[i].pos_x );
      player.setY( fld.hisTeam[i].pos_y );
      if((player-ball).norm() <= KICK_RANGE_DEF){
	//cout << "\nOP_HAS_BALL" << flush;
	reset();
	return true;
      }
    }
  }
  }
  else{
  
  bool they_have_ball=false;
  
   for (int i = 0 ; i < 11 ; i++){ // do consider goalie
    if (activeTeammate[i]) {
      if(!fld.hisTeam[i].alive) continue;
      player.setX( fld.hisTeam[i].pos_x );
      player.setY( fld.hisTeam[i].pos_y );
      ball.setX( fld.ball.pos_x );
      ball.setY( fld.ball.pos_y );
      if((player-ball).norm() <= KICK_RANGE_DEF){
        they_have_ball=true;
      }
    }
  }
  
  if(!they_have_ball){
    ball.setX( fld.ball.pos_x );
    ball.setY( fld.ball.pos_y );
    for (int i = 0 ; i < 11 ; i++){ // do consider goalie
    if(!fld.hisTeam[i].alive) continue;
    player.setX( fld.hisTeam[i].pos_x );
    player.setY( fld.hisTeam[i].pos_y );
    ball.setX( fld.ball.pos_x );
    ball.setY( fld.ball.pos_y );
    if((player-ball).norm() <= KICK_RANGE_DEF){
     //cout << "\nOP_HAS_BALL" << flush;
	reset();
	return true;
    }
    }
    for (int i = 1 ; i < 11 ; i++){ // do consider goalie!!!!
      if (!fld.myTeam[i].alive) continue;
      player.setX( fld.myTeam[i].pos_x );
      player.setY( fld.myTeam[i].pos_y );
      if((player-ball).norm() <= KICK_RANGE_DEF){
	//cout << "\nOP_HAS_BALL" << flush;
	reset();
	return true;
      }
    }
  }
  
  }

  return true;
}

/** Every time the referee sends a message, this function will be called. The parameter
    is true when the playmode has been changed. Note that there is a subtile difference
    between playmode changes and other referee messages...
*/
bool ModOOT::onRefereeMessage(bool PMChange) {

  return true;
}

/** A string entered on the keyboard will be sent through this messages. If you process
    this string, you should return true. This will prevent sending the string to
    other modules as well. Return false if you don't process the message or if you want
    the string to be sent to other modules.
*/
bool ModOOT::onKeyboardInput(const char *str) {

  return false;
}

/** Any hear message that does not come from the referee can be processed using this message.
    Unlike the keyboard input, a hear message will always be sent to all modules.
*/
bool ModOOT::onHearMessage(const char *str) {

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
bool ModOOT::onChangePlayerType(bool ownTeam,int unum,int type) {

  return false;
}

void ModOOT::moveBall(double x,double y,double velx,double vely) {
  MSG::sendMsg(MSG::MSG_MOVE,"ball",x,-y,0,velx,-vely);
  // - bei y-Werten, da der Server anscheinend positive y-Koordinaten nach unten zaehlt!
}


void ModOOT::moveOwnPlayer(int unum,double x,double y,double dir,double velx,double vely) {
  if (!fld.myTeam[unum].alive) return;
  char buf[100];
  if (dir >= PI) dir -= 2*PI;
  if (dir < -PI) dir += 2*PI;
  dir = RAD2DEG(dir);
  sprintf(buf,"player %s %d",fld.getMyTeamName(),unum+1);
  MSG::sendMsg(MSG::MSG_MOVE,buf,x,-y, dir,velx,-vely);
}

void ModOOT::moveOppPlayer(int unum,double x,double y,double dir,double velx,double vely) {
  if (!fld.hisTeam[unum].alive) return;
  char buf[100];
  if (dir >= PI) dir -= 2*PI;
  if (dir < -PI) dir += 2*PI;
  dir = RAD2DEG(dir);
  sprintf(buf,"player %s %d",fld.getHisTeamName(),unum+1);
  MSG::sendMsg(MSG::MSG_MOVE,buf,x,-y,dir,velx,-vely);
}
