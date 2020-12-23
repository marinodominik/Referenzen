#ifndef _INTERCEPTLEARNSITUATIONTABLE_H_
#define _INTERCEPTLEARNSITUATIONTABLE_H_

#include "situationtable.h"

class InterceptLearnSituationTable 
  : public SituationTable 
{
  private:
    bool checkIsBallKickable( float plX, 
                              float plY, 
                              float bX, 
                              float bY, 
                              float kickRadius);
  protected:
  public:
      //Konstruktor
	  InterceptLearnSituationTable();

  	  bool createRandomSituationTable(int rows, bool ball=true, int numPlayers=1);
  	  void createSingleRandomSituation( float *(&tmp), bool ball=true );
};

#endif
