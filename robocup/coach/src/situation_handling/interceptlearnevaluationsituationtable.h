#ifndef _INTERCEPTLEARNEVALUATIONSITUATIONTABLE_H_
#define _INTERCEPTLEARNEVALUATIONSITUATIONTABLE_H_

#include "situationtable.h"

class InterceptLearnEvaluationSituationTable 
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
	  InterceptLearnEvaluationSituationTable();

  	  bool createRandomSituationTable(int rows, bool ball=true, int numPlayers=1);
  	  void createSingleRandomSituation( float *(&tmp), bool ball=true,
  	                                    float ballVelocityX = 0.0,
  	                                    float ballVelocityY = 0.0 );
};

#endif
