/* Author: 
 *         <Original Version> Manuel "Sputnick" Nickschas, 02/2002
 *         <Extensions>       Thomas Gabel, 04/2008
 *         <Extensions (*)>   Haiko Schol, 12/2008 
 *
 * This module analyses the opponent team and tries to find out the opponent
 * player types.
 * 
 * (*)
 * This module analyses the opponent team and tries to find out the opponent
 * player types. Player decay is also used for the analysis. This method is
 * inspired by the approach taken by AT Humboldt.
 * 
 */

#ifndef _MOD_ANALYSE2010_H_
#define _MOD_ANALYSE2010_H_

#include <limits.h>
#include "defs.h"
#include "coach.h"
#include "param.h"
#include "field.h"
#include "options.h"
#include "modules.h"
#include "messages.h"

class ModAnalyse2010 : public AbstractModule 
{
  //private data structures  
  private:
    struct SpeedProgressionInformation
    {
      SpeedProgressionInformation() 
      {  ivLength=-1; 
         for (int i=0;i<SPEED_PROGRESS_MAX;i++) ivSpeedProgress[i]=0.0;
         for (int i=0;i<SPEED_PROGRESS_MAX;i++) ivNumberOfUpdates[i]=0;
      };
      int   ivLength;
      double ivSpeedProgress[SPEED_PROGRESS_MAX];
      int   ivNumberOfUpdates[SPEED_PROGRESS_MAX];
    };
    struct AccelaratingPlayer
    {
      AccelaratingPlayer() 
      { ivTimeOfRecentDashSequenceStart = 0;
        for (int i=0;i<SPEED_PROGRESS_MAX;i++) ivNumberOfHistoriesForStep[i]=0;
        ivTimeOfLastLargestSpeedProgressUpdate = -1;
      };
      int                         ivTimeOfRecentDashSequenceStart;
      int                         ivNumberOfHistoriesForStep[SPEED_PROGRESS_MAX];
      SpeedProgressionInformation ivCurrentSpeedProgress;
      SpeedProgressionInformation ivLargestSpeedProgressSoFar;
      int                         ivTimeOfLastLargestSpeedProgressUpdate;
      double                      ivErrorForPlayerType[MAX_PLAYER_TYPES];
    };
  
  
 private:
  Field oldfld;
  int oldpm;
  long ivRecentTimeBehaveEntered;

  bool changeAnnounced;
  bool goalieAnnounced;
  
  double maxKickRange;
  double kickRanges[MAX_PLAYER_TYPES];
  bool successMentioned[TEAM_SIZE*2];
  bool onlyOneLeft[TEAM_SIZE*2];
  long possTypes[TEAM_SIZE*2][MAX_PLAYER_TYPES];
  int pltypes;
  bool ballKicked;
  int kickingPlayer;
  bool canKick[TEAM_SIZE*2];
  bool mayCollide[TEAM_SIZE*2];
  AccelaratingPlayer ivSpeedProgressInformationArray[2*TEAM_SIZE];
  static const double MOVEMENT_EPSILON;

  Player *getPlByIndex(int i,Field *field);
  int     getNumberOfRemainingTypesForPlayer( int plIdx );

  void checkCollisions();
  bool checkKick();
  bool checkPlayerDecay();
  bool checkPlayerDecay09();
  bool checkInertiaMoment();
  bool checkPlayerSpeedMax();
  bool checkSpeedProgress();
  bool checkMaxSpeedProgress();
  bool checkInertiaMoment08();
  bool checkForAlreadyAssignedTypes();
  bool updateSpeedProgressInformationArray();
  
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
  void showPossTypeTable();
};

#endif
