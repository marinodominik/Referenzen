/* Author: FluffyBunny
 *
 * This is a module template
 *
 */

#include "mod_template.h"

const char ModTemplate::modName[]="ModTemplate";

/** This will be called after the rest of the coach has been initialised
    and the connection to the server has been established.
    If waitForTeam is set, you may also assume that all own players are already
    on the field. Put all module specific initialisation here.
*/
bool ModTemplate::init(int argc,char **argv) {

  return true;
}

/** The framework will call this routine just before the connection to the server
    goes down. Up to this point, all data structures of the main coach are still intact.
*/
bool ModTemplate::destroy() {
  
  return true;
}

/** Similar to the BS2kAgent behave routine, this one here will be called every cycle.
    It is currently synched to the see_global messages. Put code in here that
    should be called once per cycle just after the visual information has been updated.
*/
bool ModTemplate::behave() {
  
  return true;
}

/** Every time the referee sends a message, this function will be called. The parameter
    is true when the playmode has been changed. Note that there is a subtile difference
    between playmode changes and other referee messages...
*/
bool ModTemplate::onRefereeMessage(bool PMChange) {
  
  return true;
}

/** A string entered on the keyboard will be sent through this messages. If you process
    this string, you should return true. This will prevent sending the string to
    other modules as well. Return false if you don't process the message or if you want
    the string to be sent to other modules.
*/
bool ModTemplate::onKeyboardInput(const char *str) {

  return false;
}

/** Any hear message that does not come from the referee can be processed using this message.
    Unlike the keyboard input, a hear message will always be sent to all modules.
*/
bool ModTemplate::onHearMessage(const char *str) {

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
bool ModTemplate::onChangePlayerType(bool ownTeam,int unum,int type) {

  return false;
}
