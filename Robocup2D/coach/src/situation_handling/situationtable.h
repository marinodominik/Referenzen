#ifndef _SITUATIONTABLE_H_
#define _SITUATIONTABLE_H_

#include <iostream>
#include <iomanip>
#include <stdlib.h>

// einfaches Makro f?r Zufallszahlen
#define TG_MY_RANDOM_NUMBER ((float)rand() / (float)RAND_MAX)

using namespace std;

//===========================================================================
// CLASS SituationTable
//===========================================================================
class SituationTable 
{
 	 friend ostream& operator<< (ostream& o,const SituationTable& t);
	 
	protected:
	  //instance variables
	  int ivNumRows;
	  int ivNumCols;
	  int ivMaxRows;
	  float **ivppTableArray;

      //private methods
      float getRandomPlayerPosWithinRange(float from, float to);
	  float getRandomPlayerPosX();
	  float getRandomPlayerPosY();
	  float getRandomPlayerAngle();
	  float getRandomPlayerVelX();
	  float getRandomPlayerVelY();
	  float getRandomBallPosX();
	  float getRandomBallPosY();
	  float getRandomBallVelX();
	  float getRandomBallVelY();
	  bool checkBallVelocity(float vx, float vy, float ballSpeedDivisor);
	  bool checkPlayerVelocity(float vx, float vy);



	public:
      //Konstruktor
      SituationTable ();
      //Desnstruktor
      virtual ~SituationTable ();
	
	  //?ffentliche Methoden
	  float operator()(int row,int col) const;
	  const float * operator()(int row) const;
	  int getNumberOfColumns();
	  void set(int row, int col, float value);
	  void setMaxRows(int max);
	  void setOffObject(int index);
	  int getNumberOfRows() const  {	return ivNumRows;  }
	
	  string toString(int num);
	  bool save(const char*) const;
	  bool load(int col,const char*);
	  virtual bool createRandomSituationTable(int rows, bool ball, int numPlayers);

 	  float   getElement(int row,int col) const;
	  float * getRow(int row) const;
};

#endif
