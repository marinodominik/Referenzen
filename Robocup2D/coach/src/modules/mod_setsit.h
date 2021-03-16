/* Author: Hauke Strasdat
 * email: strasdat@gmx.de
 *
 * Training coach module to set repeatedly the same situation.
 *
 */

#ifndef _MOD_SETSIT_H_
#define _MOD_SETSIT_H_

#include "defs.h"
#include "coach.h"
#include "param.h"
#include "field.h"
#include "options.h"
#include "modules.h"
#include "messages.h"

class ModSetSit : public AbstractModule {
//  Vector *opp;
//  Vector *mate;

  void moveOwnPlayer(int unum,double x,double y,double dir,double velx,double vely);
  void moveOppPlayer(int unum,double x,double y,double dir,double velx,double vely);
  void moveBall(double x,double y,double velx,double vely);
  void reset();
  int teammate1[2];
  int teammate2[2];
  int teammate3[2];
  int teammate4[2];
  int teammate5[2];
  int teammate6[2];
  int teammate7[2];
  int teammate8[2];
  int teammate9[2];
  int teammate10[2];
  int teammate11[2];
  int opponent1[2];
  int opponent2[2];
  int opponent3[2];
  int opponent4[2];
  int opponent5[2];
  int opponent6[2];
  int opponent7[2];
  int opponent8[2];
  int opponent9[2];
  int opponent10[2];
  int opponent11[2];
  int ball[2];
  int activeTeammateIndex[11];
  bool activeTeammate[11];
  int activeTeammateNumber[1];
  int opponentMode[1];
  int noKickIn[1];

 public:


  bool init(int argc,char **argv);             /** init the module */
  bool destroy();                              /** tidy up         */
  
  bool behave();                               /** called once per cycle, after visual update   */
  bool onRefereeMessage(bool PMChange);        /** called on playmode change or referee message */
  bool onHearMessage(const char *str);         /** called on every hear message from any player */
  bool onKeyboardInput(const char *str);       /** called on keyboard input                     */
  bool onChangePlayerType(bool,int,int=-1);    /** SEE mod_template.c!                          */
  
  static const char modName[];                 /** module name, should be same as class name    */
  const char *getModName() {return modName;}   /** do not change this!                          */
};

#endif
