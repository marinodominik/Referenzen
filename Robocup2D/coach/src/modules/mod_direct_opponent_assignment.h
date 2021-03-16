/* Author: 
 *
 *
 */

#ifndef _MOD_DIRECT_OPPONENT_ASSIGNMENT_H_
#define _MOD_DIRECT_OPPONENT_ASSIGNMENT_H_

#include "defs.h"
#include "coach.h"
#include "param.h"
#include "field.h"
#include "options.h"
#include "modules.h"
#include "messages.h"

class ModDirectOpponentAssignment : public AbstractModule {
 private:
  Field ivOldFld;
  int ivOldPM;
  int ivLastDirectOpponentComputation;
  int ivLastChangeInComputedDirectOpponents;
  int ivLastSentDirectOpponentAssignments;
  int ivCurrentScoreLeft;
  int ivCurrentScoreRight;
  
  int ivDirectOpponentAssignment[TEAM_SIZE];
  struct
  { 
    int num; float x; float y; 
    float getAverageXPosition() {if(num==0)return 0.0; else return x/(float)num;}
    float getAverageYPosition() {if(num==0)return 0.0; else return y/(float)num;}
  } ivOpponentPositionArray[TEAM_SIZE];
  struct
  { float x; float y;
  } ivMyInitialPositions[TEAM_SIZE]; 
  
  Player *getPlByIndex(int i,Field *field);
  Player * findNearestOpponentTo( double posx,double posy, bool * assignableOpps);
  void createDirectOpponentAssignment();
  void sendDirectOpponentAssignment();
  void updateOpponentPositions();
  
 public:

  static int cvDirectOpponentAssignment[TEAM_SIZE];

  bool init(int argc,char **argv);             /** init the module */
  bool destroy();                              /** tidy up         */
  
  bool behave();                               /** called once per cycle, after visual update   */
  bool onRefereeMessage(bool PMChange);        /** called on playmode change or referee message */
  bool onHearMessage(const char *str);         /** called on every hear message from any player */
  bool onKeyboardInput(const char *str);       /** called on keyboard input                     */
  bool onChangePlayerType(bool,int,int);       /** SEE mod_template.c!                          */
  
  static const char modName[];                 /** module name, should be same as class name    */
  const char *getModName() {return modName;}   /** do not change this!                          */
  
  bool doWePlayAgainstWrightEagle();
};

#endif
