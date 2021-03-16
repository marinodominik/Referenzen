/* Author: Thomas Gabel, 03/2005
 *
 * This is the coach's module for changing player types.
 *
 */

#ifndef _MOD_CHANGE05_H_
#define _MOD_CHANGE05_H_

#include "defs.h"
#include "param.h"
#include "field.h"
#include "options.h"
#include "modules.h"
#include "messages.h"

class ModChange05 : public AbstractModule 
{

  private:
    #define NUMBER_OF_PLAYER_ROLES 5
    enum PLAYER_ROLES {GOALIE=0, SWEEPER=1, DEFENDER=2, MIDFIELDER=3, OFFENDER=4};
  
    bool ivPreparationDone;
    bool ivWaitForTeam;

    bool ivDoEmergencyChange;
    int  ivEmergencyPlayerType;
    int  ivCurrentScoreLeft,
         ivCurrentScoreRight;
  
    int  ivSubstitutionQueueCounter;
    int  ivTotalSubstitutions;
    int  ivSubstitutionsToDo;
  
    int ivTypesOnField[MAX_PLAYER_TYPES];
    struct 
    {
        int unum;
        int type;
        int time;
        int done;
    } ivSubstitutionQueue[11];  

    struct PlayerTypeUsability
    {
      PlayerTypeUsability() 
      { for (int i=0; i<MAX_PLAYER_TYPES; i++) { pt[i]=0; usability[i]=0.0; } }
      int   pt[MAX_PLAYER_TYPES];
      float usability[MAX_PLAYER_TYPES];
      void  sort();
    };
    PlayerTypeUsability ivPlayerTypeForRoleUsability[NUMBER_OF_PLAYER_ROLES];

    struct 
    {
        int          unum;
        PLAYER_ROLES role;
    } ivInitialSubstitutionOrder[NUM_PLAYERS];

    int compareRealPlayerSpeedMaximum(const PlayerType*,const PlayerType*);
    int compareStaminaDemandFor10Meters(const PlayerType*,const PlayerType*);

  protected:
    void  prepareForGame();
    bool  changePlayer(int unum,int type);     /** adds substitution to queue if play_on */
    bool  sendQueue();
    void  fillPlayerTypeForRoleUsabilityMatrix();
    float computePlayerTypeUsabilityForRole(PlayerType   * pt, 
                                            PLAYER_ROLES   role);
    float computePlayerTypeRealPlayerSpeedMaxOptimality(const PlayerType * pt);
    float computePlayerTypeStaminaDemandPerMeterOptimality(const PlayerType * pt);
    float computePlayerTypeSpeedProgressionOptimality(const PlayerType * pt);
    float computePlayerTypeKickableMarginOptimality(const PlayerType * pt);
  
  public:

    bool init(int argc,char **argv);
    bool destroy();
  
    bool behave();
    bool onRefereeMessage(bool PMChange);
    bool onKeyboardInput(const char *);
  
    static const char modName[];
    const char *getModName() {return modName;}
  
  
};

#endif
