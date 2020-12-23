/* Author: Manuel "Sputnick" Nickschas, 02/2002
 *
 * This is the main coach module, offering core functionalities.
 *
 */

#ifndef _MOD_CHANGE_H_
#define _MOD_CHANGE_H_

#include "defs.h"
#include "param.h"
#include "field.h"
#include "options.h"
#include "modules.h"
#include "messages.h"

class ModChange : public AbstractModule {

  bool prepDone;
  bool waitForTeam;

  bool emerg_change;
  int emerg_type;
  int left_goals,right_goals;
  
  int fastestPlayers[3];

  int queueCnt;
  int totalSubs;
  int subsToDo;
  int typesOnFld[MAX_PLAYER_TYPES];
  struct {int unum;int type;int time;int done;} subQueue[11];  

  int cmp_real_player_speed_max(const PlayerType*,const PlayerType*);
  int cmp_stamina10m(const PlayerType*,const PlayerType*);
  
 public:

  bool init(int argc,char **argv);
  bool destroy();
  
  bool behave();
  bool onRefereeMessage(bool PMChange);
  bool onKeyboardInput(const char *);
  
  static const char modName[];
  const char *getModName() {return modName;}
  
  bool changePlayer(int unum,int type);     /** adds substitution to queue if play_on */
  bool sendQueue();
  
 protected:
  void prepareForGame();
};

#endif
