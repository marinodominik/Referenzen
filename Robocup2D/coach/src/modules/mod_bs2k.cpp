/* Author: Andreas Hoffmann
 *
 * This is a module template
 *
 */

#include "mod_bs2k.h"

/** This will be called after the rest of the coach has been initialised
    and the connection to the server has been established.
    If waitForTeam is set, you may also assume that all own players are already
    on the field. Put all module specific initialisation here.
*/
bool ModBS2K::init(int argc,char **argv) {
  //lade NN aus ../bs2k/conf
  coach_net = new Net();
  if(coach_net->load_net("../bs2k/conf/coach.net") == FILE_ERROR){
    cout<<"Coach: No net ../bs2k/conf/coach.net found - stop loading\n";
    exit(0);
  } 
  float value[7];
  int sort[7];
  for(int i=0;i<7;i++){
    coach_net->in_vec[0] = fld.plType[i].player_speed_max;
    coach_net->in_vec[1] = fld.plType[i].stamina_inc_max;
    coach_net->in_vec[2] = fld.plType[i].player_decay;
    coach_net->in_vec[3] = fld.plType[i].inertia_moment;
    coach_net->in_vec[4] = fld.plType[i].dash_power_rate;
    coach_net->in_vec[5] = fld.plType[i].player_size;
    coach_net->in_vec[6] = fld.plType[i].kickable_margin;
    coach_net->in_vec[7] = fld.plType[i].kick_rand;
    coach_net->in_vec[8] = fld.plType[i].extra_stamina;
    coach_net->in_vec[9] = fld.plType[i].effort_max;
    coach_net->in_vec[10] = fld.plType[i].effort_min;
    coach_net->forward_pass(coach_net->in_vec,coach_net->out_vec);
    value[i] = coach_net->out_vec[0];
    sort[i] = i;
  }
  float temp; int temp2;
  for(int i=0;i<7;i++){
    for(int j=i;j<7;j++){
      if(value[i]>value[j]){
	temp = value[i];
	value[i] = value[j];
	value[j] = temp;
	temp2 = sort[i];
	sort[i] = sort[j];
	sort[j] = temp2;
      }
    }
  }
  int counter = 11;
  for(int i=0;i<3;i++){
    MSG::sendMsg(MSG::MSG_CHANGE_PLAYER_TYPE, /*fld.getMyTeamName(),*/ counter--, (int)sort[i]);
    MSG::sendMsg(MSG::MSG_CHANGE_PLAYER_TYPE, /*fld.getMyTeamName(),*/ counter--, (int)sort[i]);
    MSG::sendMsg(MSG::MSG_CHANGE_PLAYER_TYPE, /*fld.getMyTeamName(),*/ counter--, (int)sort[i]);
  }
  return true;
}

/** The framework will call this routine just before the connection to the server
    goes down. Up to this point, all data structures of the main coach are still intact.
*/
bool ModBS2K::destroy() {
  
  return true;
}

/** Similar to the BS2kAgent behave routine, this one here will be called every cycle.
    It is currently synched to the see_global messages. Put code in here that
    should be called once per cycle just after the visual information has been updated.
*/
bool ModBS2K::behave() {

  return true;
}

/** Every time the referee sends a message, this function will be called. The parameter
    is true when the playmode has been changed. Note that there is a subtile difference
    between playmode changes and other referee messages...
*/
bool ModBS2K::onRefereeMessage(bool PMChange) {
  
  return true;
}

/** A string entered on the keyboard will be sent through this messages. If you process
    this string, you should return true. This will prevent sending the string to
    other modules as well. Return false if you don't process the message or if you want
    the string to be sent to other modules.
*/
bool ModBS2K::onKeyboardInput(const char *str) {

  return false;
}

/** Any hear message that does not come from the referee can be processed using this message.
    Unlike the keyboard input, a hear message will always be sent to all modules.
*/
bool ModBS2K::onHearMessage(const char *str) {

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
bool ModBS2K::onChangePlayerType(bool ownTeam,int unum,int type) {

  return false;
}

/* Don't forget this one... */
const char ModBS2K::modName[]="ModBS2K";
