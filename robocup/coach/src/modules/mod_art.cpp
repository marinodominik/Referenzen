/* Author: Art! , 02/2002
 *
 * This is a private module
 *
 */

#include "coach.h"
#include "mod_art.h"
#include "messages.h"
#include <string.h>

ModArt::ModArt() {}

bool ModArt::init(int argc,char **argv) {
  //cout << "\nModArt init!";

  return true;
}

bool ModArt::destroy() {
  cout << "\nModArt destroy!";

  return true;
}

bool ModArt::behave() {
  //cout << "\nModArt behave!";

  return true;
}

bool ModArt::onRefereeMessage(bool PMChange) {
  //cout << "\nModArt referee! " << PMChange;

  return true;
}

bool ModArt::onKeyboardInput(const char *str) {
  if(str[0]=='(') {
    RUN::sock.send_msg(str,strlen(str)+1);    
    return true;
  }
  if(!strcmp(str, "print_types")){
    RUN::printPlayerTypes();
    return true;
  }
  return false;
}

const char ModArt::modName[]="ModArt";
