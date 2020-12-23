/* Author: Art! , 02/2002
 *
 * This is a module
 *
 */

#ifndef _MOD_ART_H_
#define _MOD_ART_H_

#include "defs.h"
#include "param.h"
#include "field.h"
#include "options.h"
#include "modules.h"

class ModArt : public AbstractModule {

 public:

  ModArt();
  bool init(int argc,char **argv);
  bool destroy();
  
  bool behave();
  bool onRefereeMessage(bool PMChange);
  bool onKeyboardInput(const char *str);
  
  static const char modName[];
  const char *getModName() {return modName;}
};

#endif
