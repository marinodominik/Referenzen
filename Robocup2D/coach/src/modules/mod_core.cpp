/* Author: Manuel "Sputnick" Nickschas, 02/2002
 *
 * This is the main coach module, offering core functionalities.
 *
 */

#include "coach.h"
#include "mod_defs.h"
#include "mod_core.h"
#include "messages.h"
//#include "modules.h"

ModCore::ModCore() {}

bool ModCore::init(int argc,char **argv) {
  //cout << "\nModCore init!";

  return true;
}

bool ModCore::destroy() {
//  cout << "\nModCore destroy!";

  return true;
}

bool ModCore::behave() {
  //cout << "\nModCore behave!";

  return true;
}

bool ModCore::onRefereeMessage(bool PMChange) {
  //cout << "\nModCore referee! " << PMChange;

  return true;
}

bool ModCore::onHearMessage(const char *str) {
  //cout << "\nHEAR: "<<str;
  return true;
}

bool ModCore::onKeyboardInput(const char *str) {
  if(str[0]=='(') {
    RUN::sock.send_msg(str,strlen(str)+1);    
    return true;
  }
  
  return false;
}

const char ModCore::modName[]="ModCore";
