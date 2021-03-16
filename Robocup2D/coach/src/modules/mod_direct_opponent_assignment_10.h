/* Author: 
 *
 *
 */

#ifndef _MOD_DIRECT_OPPONENT_ASSIGNMENT_10_H_
#define _MOD_DIRECT_OPPONENT_ASSIGNMENT_10_H_

#include "wmtools.h"
#include "defs.h"
#include "coach.h"
#include "param.h"
#include "field.h"
#include "options.h"
#include "modules.h"
#include "messages.h"

class ModDirectOpponentAssignment10 : public AbstractModule {
 private:
  Field ivOldFld;
  int ivOldPM;
  int ivLastDirectOpponentComputation;
  int ivLastChangeInComputedDirectOpponents;
  int ivLastSentDirectOpponentAssignments;
  int ivCurrentScoreLeft;
  int ivCurrentScoreRight;
  bool ivConflictOverrideAssignment;
  double ivConflictBoundaryThresh;
  double ivMinCoveringDistance;
  
  int ivDirectOpponentAssignmentConflict[TEAM_SIZE+1];
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
  
  Player * findNearestOpponentTo( double posx,double posy, bool * assignableOpps);
  void createDirectOpponentAssignment();
  void sendDirectOpponentAssignment();
  void updateOpponentPositions();
  
  bool isPlayerBehindConflictBoundary(Player *p);
  double distanceToMyBaseline(Player *p);
  Player **get_sorted_by(int how_many, double * measured_data);
  void calculateBestReplacementAssignment(int my_p_array_pos, int my_p_num, double *distances, double max_dist);
  void checkConflicts();
  char playerNumToChar(int n);
  char ourPlayerToChar(int n);
  bool isAttacker(int num);
  double getConflictResolveBoundary();
  
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
