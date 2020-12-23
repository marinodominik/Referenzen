/* Author: Manuel "Sputnick" Nickschas, 02/2002
 *
 * See messages.h for description..
 *
 */

#include <string.h>

#include "coach.h"
#include "modules.h"
//#include "mod_defs.h"
#include "logger.h"
#include "str2val.h"


using namespace MOD;

int MOD::numModules;
AbstractModule *MOD::coachModule[MAX_MODULES];

void MOD::initModules(int argc, char **argv) {
//  cout << "\nInitialising modules...";
  
  for(int i=0;i<numModules;i++) {
    coachModule[i]->init(argc,argv);
  }
}

void MOD::destroyModules() {
//  cout << "\nShutting down modules...";

  for(int i=0;i<numModules;i++) {
    coachModule[i]->destroy();
  }
}

void MOD::loadModules() {
  char buf[2048];
  char name[100];
  const char *str,*pnt;
  str=buf;
  numModules=0;
//  cout << "\n------------------------------------------------------------------"
//       << "\nLoading modules...\n";
  ValueParser vp(Options::coach_conf,"Modules");
  vp.get("LoadModules",buf,2048,"ModCore");
  AbstractModule *mod;
  bool flg=false;int cnt=0;
  do {
    if(!strfind(str,',',pnt)) {
      mod=getModule(str);
//      if(!mod) cout <<"<"<<str<<"> ";
      flg=true;cnt++;
    } else {
      strncpy(name,str,pnt-str);
      name[pnt-str]='\000';
      mod=getModule(name);
//      if(!mod) cout <<"<"<<name<<"> ";
      str=pnt+1;cnt++;
    }
    if(mod!=NULL) {
      coachModule[numModules++]=mod;
//      cout << "[" << mod->getModName() << "] ";
    }
  } while (!flg);
  if(cnt!=numModules) cout << "\nWARNING: One or more modules could not be found!";
}


void MOD::unloadModules() {
  for(int i=0;i<numModules;i++) delete coachModule[i];
}
