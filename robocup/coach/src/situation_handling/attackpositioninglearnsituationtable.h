#ifndef _ATTACKPOSITIONINGLEARNSITUATIONTABLE_H_
#define _ATTACKPOSITIONINGLEARNSITUATIONTABLE_H_

#include "situationtable.h"
#include <stdlib.h>

class AttackPositioningLearnSituationTable
  : public SituationTable 
{
  struct playerPositionAreas
  {
    float ivAverageX, ivAverageY, ivMaximalXDeviation, ivMaximalYDeviation;
  };
  
  private:
    playerPositionAreas ivpTeam1PositionAreas[11];
    playerPositionAreas ivpTeam2PositionAreas[11];
    bool checkIsBallKickable( float plX, 
                              float plY, 
                              float bX, 
                              float bY, 
                              float kickRadius);
    bool init();
  protected:
  public:
      //Konstruktor
      AttackPositioningLearnSituationTable();

  	  bool createRandomSituationTable(int rows, bool ball=true, int numPlayers=1);
  	  void createSingleRandomSituation( float *(&tmp), 
                                        int      numPlayers = 1, 
                                        bool     ball = true );
};

#endif
