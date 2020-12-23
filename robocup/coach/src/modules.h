/* Author: Manuel "Sputnick" Nickschas, 02/2002
 *
 * This file contains stuff to deal with modules.
 *
 */

#ifndef _MODULES_H_
#define _MODULES_H_

#include "defs.h"
#include "field.h"
#include "options.h"

class AbstractModule {

 public:
  AbstractModule() {}
  virtual ~AbstractModule() {}

  virtual bool init(int argc,char **argv)=0;// {return false;}
  virtual bool destroy() {return false;}
  
  virtual bool behave() {return false;}           // called every cycle right after see_global message
  virtual bool onRefereeMessage(bool PMChange) {return false;} // called on every referee message
  virtual bool onKeyboardInput(const char*) {return false;}    // return true if processed!
  virtual bool onHearMessage(const char *) {return false;}     // called on hear messages (no referee)
  virtual bool onChangePlayerType(bool ownTeam,int unum,int type=-1) {return false;}
                                                               // called on player change
  
  static const char modName[];//="Abstract";
  virtual const char *getModName()=0;

};

//const char AbstractModule::modName[];

namespace MOD {

  const int MAX_MODULES=20;
  
  extern int numModules;
  extern AbstractModule *coachModule[MAX_MODULES];

  extern AbstractModule *getModule(const char *str); // this one is implemented in mod_defs.h!
  void loadModules();
  void initModules(int argc,char **argv);
  void destroyModules();
  
  void unloadModules();
  
}

#endif
