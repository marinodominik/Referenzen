/* Author: FluffyBunny
 *
 * This shall demonstrate offline coach capabilities.
 *
 */

#ifndef _MOD_DEMO_TRAIN_H_
#define _MOD_DEMO_TRAIN_H_

#include "defs.h"
#include "coach.h"
#include "param.h"
#include "field.h"
#include "options.h"
#include "modules.h"
#include "messages.h"

class ModDemoTrain : public AbstractModule {

  long cyc_cnt;
  long runs;
  long misses;
  double cyc_avg;
  double diff_avg;
  
 public:

  bool init(int argc,char **argv);             /** init the module */
  bool destroy();                              /** tidy up         */
  
  bool behave();                               /** called once per cycle, after visual update   */
  bool onRefereeMessage(bool PMChange);        /** called on playmode change or referee message */
  bool onHearMessage(const char *str);         /** called on every hear message from any player */
  bool onKeyboardInput(const char *str);       /** called on keyboard input                     */
  bool onChangePlayerType(bool,int,int=-1);    /** SEE mod_template.c!                          */

  void moveBall(double x,double y,double velx=0,double vely=0);
  void moveOwnPlayer(int unum,double x,double y,double dir=0,double velx=0,double vely=0);
  void moveOppPlayer(int unum,double x,double y,double dir=0,double velx=0,double vely=0);
  
  static const char modName[];                 /** module name, should be same as class name    */
  const char *getModName() {return modName;}   /** do not change this!                          */
};

#endif
