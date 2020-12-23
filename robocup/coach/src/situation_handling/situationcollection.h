#ifndef _SITUATIONCOLLECTION_H_
#define _SITUATIONCOLLECTION_H_

#include "situationtable.h"

#define			NUMBER_OF_OBJECTS_IN_SITUATION		23
#define			SITUATION_TABLE_NUMBER_OF_COLUMNS	(4+(11*5)+(11*5))

//===========================================================================
// CLASS SituationCollection
//===========================================================================
class SituationCollection
{
	private:
	  
	protected:
	  SituationTable *	ivpTable;
	  bool 				ivActiveArray [NUMBER_OF_OBJECTS_IN_SITUATION];
	  int 				ivNumberOfSituations;
	  int 				ivCurrentSituation;
	  
	public:
bool wiederholungsFlag;
	  //Konstanten
	    //Typen von Situationssammlungen
	    const static int cvs_SITUATION_COLLECTION_TYPE_NORMAL          = 0;
	    const static int cvs_SITUATION_COLLECTION_TYPE_KICK_LEARN      = 1;
	    const static int cvs_SITUATION_COLLECTION_TYPE_INTERCEPT_LEARN = 2;
	    const static int 
               cvs_SITUATION_COLLECTION_TYPE_EVALUATE_INTERCEPT_LERN = 3;
      const static int
              cvs_SITUATION_COLLECTION_TYPE_ATTACK_POSITIONING_LEARN = 4;
      const static int cvs_SITUATION_COLLECTION_TYPE_HASSLE_LEARN    = 5;
	  
	  //Konstruktoren
	  SituationCollection();
	  ~SituationCollection();
	  SituationCollection(int type);
	  //Methoden
	  int  getNumberOfSituations() const;
	  void setCurrentSituation(int i);
	  
	  void loadTable(const char* fname);
	  void loadTable(const char* fname, int maxSize);
	  void createRandomTable(int numOfSituations);
    void saveTable(const char * filename);
    
	  
	  void setObjectWithIndexToActive(int i);
	  void setObjectWithIndexToInactive(int i);
	  void sendSituation(int);
	  bool doNextSituation();
    bool redoCurrentSituation();
}; 

#endif
