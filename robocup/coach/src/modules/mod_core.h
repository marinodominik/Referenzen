/* Author: Manuel "Sputnick" Nickschas, 02/2002
 *
 * This is the main coach module, offering core functionalities.
 *
 */

#ifndef _MOD_CORE_H_
#define _MOD_CORE_H_

#include "defs.h"
#include "param.h"
#include "field.h"
#include "options.h"
#include "modules.h"


class ModCore : public AbstractModule {

 public:

  ModCore();
  bool init(int argc,char **argv);
  bool destroy();
  
  bool behave();
  bool onRefereeMessage(bool PMChange);
  bool onHearMessage(const char *str);
  bool onKeyboardInput(const char *str);
  
  static const char modName[];
  const char *getModName() {return modName;}
  
};

#endif
