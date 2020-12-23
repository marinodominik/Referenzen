/* Author: Manuel "Sputnick" Nickschas, 02/2002
 *
 * This module analyses the opponent team and tries to find out the opponent
 * player types.
 *
 */

#ifndef _MOD_ANALYSE_H_
#define _MOD_ANALYSE_H_

#include "defs.h"
#include "coach.h"
#include "param.h"
#include "field.h"
#include "options.h"
#include "modules.h"
#include "messages.h"

class ModAnalyse : public AbstractModule {
 private:
  Field oldfld;
  int oldpm;

  bool changeAnnounced;
  bool goalieAnnounced;
  
  double maxKickRange;
  double kickRanges[MAX_PLAYER_TYPES];
  bool successMentioned[TEAM_SIZE*2];
  bool onlyOneLeft[TEAM_SIZE*2];
  int possTypes[TEAM_SIZE*2][MAX_PLAYER_TYPES];
  int pltypes;
  bool ballKicked;
  int kickingPlayer;
  bool canKick[TEAM_SIZE*2];
  bool mayCollide[TEAM_SIZE*2];

  Player *getPlByIndex(int i,Field *field);

  void checkCollisions();
  bool checkKick();
  bool checkPlayerDecay();
  bool checkInertiaMoment();
  bool checkPlayerSpeedMax();
  
 public:

  bool init(int argc,char **argv);             /** init the module */
  bool destroy();                              /** tidy up         */
  
  bool behave();                               /** called once per cycle, after visual update   */
  bool onRefereeMessage(bool PMChange);        /** called on playmode change or referee message */
  bool onHearMessage(const char *str);         /** called on every hear message from any player */
  bool onKeyboardInput(const char *str);       /** called on keyboard input                     */
  bool onChangePlayerType(bool,int,int);       /** SEE mod_template.c!                          */
  
  static const char modName[];                 /** module name, should be same as class name    */
  const char *getModName() {return modName;}   /** do not change this!                          */
};

#endif
