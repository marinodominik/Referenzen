#include "situationcollection.h"
#include "attackpositioninglearnsituationtable.h"
#include "interceptlearnsituationtable.h"
#include "interceptlearnevaluationsituationtable.h"
#include "hasslelearnsituationtable.h"
#include "kicklearnsituationtable.h"

#include "../logger.h"
#include "../messages.h"
#include "../coach.h"


//###########################################################################
// KLASSE SituationCollection
//
//
//###########################################################################

//===========================================================================
// OEFFENTLICHE METHODEN (sowie Freunde)
//===========================================================================

//-------------------------------------------------------------------------
// DEFAULT CONSTRUCTOR
//-------------------------------------------------------------------------
SituationCollection::SituationCollection() 
{
wiederholungsFlag=true;
  ivpTable = new SituationTable();
  ivNumberOfSituations = 0; 
  ivCurrentSituation= -1;
  for (int i=0; i<NUMBER_OF_OBJECTS_IN_SITUATION; i++)
      setObjectWithIndexToActive(i);
}

//-------------------------------------------------------------------------
// CONSTRUCTOR WITH TYPE SPECIFICATION
//-------------------------------------------------------------------------
SituationCollection::SituationCollection(int type) 
{
  LOG_FLD(3,"SituationCollection: Entered constructor for type "<<type<<".");
  switch (type)
  {
  	case cvs_SITUATION_COLLECTION_TYPE_NORMAL:
    {
      LOG_FLD(3,"SituationCollection: Constructor -> case NORMAL / "
        <<cvs_SITUATION_COLLECTION_TYPE_NORMAL);
  	  ivpTable = new SituationTable();
  	  break;
    }
  	case cvs_SITUATION_COLLECTION_TYPE_KICK_LEARN:
    {
      LOG_FLD(3,"SituationCollection: Constructor -> case KICK / "
        <<cvs_SITUATION_COLLECTION_TYPE_KICK_LEARN);
  	  ivpTable = new KickLearnSituationTable();
  	  break;
    }
  	case cvs_SITUATION_COLLECTION_TYPE_INTERCEPT_LEARN:
    {
      LOG_FLD(3,"SituationCollection: Constructor -> case INTERCEPT / "
        <<cvs_SITUATION_COLLECTION_TYPE_INTERCEPT_LEARN);
  	  ivpTable = new InterceptLearnSituationTable();
  	  break;
    }
  	case cvs_SITUATION_COLLECTION_TYPE_EVALUATE_INTERCEPT_LERN:
    {
      LOG_FLD(3,"SituationCollection: Constructor -> case EVALUATE_INTERCEPT / "
        <<cvs_SITUATION_COLLECTION_TYPE_EVALUATE_INTERCEPT_LERN);
  	  ivpTable = new InterceptLearnEvaluationSituationTable();
  	  break;
    }
    case cvs_SITUATION_COLLECTION_TYPE_ATTACK_POSITIONING_LEARN:
    {
      LOG_FLD(3,"SituationCollection: Constructor -> case ATTACK_POSITIONING / "
        <<cvs_SITUATION_COLLECTION_TYPE_ATTACK_POSITIONING_LEARN);
      ivpTable = new AttackPositioningLearnSituationTable();
      break;
    }
    case cvs_SITUATION_COLLECTION_TYPE_HASSLE_LEARN:
    {
      LOG_FLD(3,"SituationCollection: Constructor -> case HASSLE_LEARN / "
        <<cvs_SITUATION_COLLECTION_TYPE_HASSLE_LEARN);
      ivpTable = new HassleLearnSituationTable();
      break;
    }
  }
  ivNumberOfSituations = 0; 
  ivCurrentSituation= -1;
  for (int i=0; i<NUMBER_OF_OBJECTS_IN_SITUATION; i++)
      setObjectWithIndexToActive(i);
}

//-------------------------------------------------------------------------
// DENSTRUCTOR
//-------------------------------------------------------------------------
SituationCollection::~SituationCollection() 
{
  if (ivpTable) 
    delete ivpTable;
}

//---------------------------------------------------------------------------
// METHOD getNumberOfSituations
//---------------------------------------------------------------------------
int
SituationCollection::getNumberOfSituations() const 
{ 
	 return ivNumberOfSituations; 
}
	  
//---------------------------------------------------------------------------
// METHOD setCurrentSituation
//---------------------------------------------------------------------------
void 
SituationCollection::setCurrentSituation(int i) 
{ 
	 if (i>=0 && i<ivNumberOfSituations) 
	   ivCurrentSituation = i; 
}
	  
//---------------------------------------------------------------------------
// METHOD setObjectWithIndexToActive
//---------------------------------------------------------------------------
void 
SituationCollection::setObjectWithIndexToActive(int i) 
{ 
	 if (i>=0 && i<NUMBER_OF_OBJECTS_IN_SITUATION) 
	   ivActiveArray[i]= true; 
}
	  
//---------------------------------------------------------------------------
// METHOD setObjectWithIndexToInactive
//---------------------------------------------------------------------------
void 
SituationCollection::setObjectWithIndexToInactive(int i) 
{ 
  	 if (i>=0 && i<NUMBER_OF_OBJECTS_IN_SITUATION) 
  	   ivActiveArray[i]= false; 
}
	  
//-------------------------------------------------------------------------
// METHOD doNextSituation()
//-------------------------------------------------------------------------
/**
 * This method starts the next of its sequences (note, that the terms situation
 * and sequence are used equally). It returns FALSE if the current situation
 * which has been started is the FIRST in this SituationCollection, and it
 * returns TRUE otherwise. Exception: When starting the very first sequence
 * TRUE is returned as well. */
bool
SituationCollection::doNextSituation() 
{
  //Usually, return FALSE.
  bool returnValue = false;
wiederholungsFlag=false;
if (wiederholungsFlag == false) 
{
  ivCurrentSituation ++;
wiederholungsFlag=true;
}
else wiederholungsFlag=false;
    //Return TRUE, if this is the first sequence.
    if (ivCurrentSituation > 0) returnValue = true;
    //Set this situation.
  sendSituation( ivCurrentSituation );
  //Consider the maximal amount of situations to be repeated.
    if ( ivCurrentSituation >= ivNumberOfSituations-1 ) 
      ivCurrentSituation = -1;
cout<<"### ivCurrentSituation = "<<ivCurrentSituation<<"   RETURNVALUE = "<<returnValue<<"   ivNumberOfSituations = "<<ivNumberOfSituations<<endl<<flush;
    return returnValue;
}
  
//-------------------------------------------------------------------------
// METHOD redoCurrentSituation()
//-------------------------------------------------------------------------
/**
 *
*/
bool
SituationCollection::redoCurrentSituation() 
{
  //Set this situation.
  if (ivCurrentSituation == -1)
    sendSituation( ivNumberOfSituations-1 );
  else
    sendSituation( ivCurrentSituation );
  return true;
}
  
//-------------------------------------------------------------------------
// LOAD the situations' TABLE
//-------------------------------------------------------------------------
void 
SituationCollection::loadTable(const char* fname) 
{
  ivpTable->load( SITUATION_TABLE_NUMBER_OF_COLUMNS, fname); 
//  ivpTable->createRandomSituationTable(5, true, 22) ;
  ivNumberOfSituations = ivpTable->getNumberOfRows();
}
void 
SituationCollection::loadTable(const char* fname, int maxSize) 
{
  this->loadTable(fname);
  cout<<"Es wurden "<<ivNumberOfSituations<<" Situationen geladen."<<endl;
  if (maxSize<ivNumberOfSituations)
    ivNumberOfSituations = maxSize;
}

//#include <sys/time.h>
//-------------------------------------------------------------------------
// CREATE a RANDOM TABLE of situations
//-------------------------------------------------------------------------
void 
SituationCollection::createRandomTable(int numOfSituations) 
{
//timeval tval;
//gettimeofday(&tval,NULL);
//srand(tval.tv_usec);
  //Wir generieren eine Situation f?r alle 22 Spieler und den Ball (true).
  ivpTable->createRandomSituationTable(numOfSituations, true, 22);
  ivNumberOfSituations = ivpTable->getNumberOfRows();
}

//-------------------------------------------------------------------------
// SAVE TABLE of situations
//-------------------------------------------------------------------------
void 
SituationCollection::saveTable(const char * filename) 
{
  //Nachdem wir eine zufaellige Tabelle erzeugt haben, speichern wir
  //diese auch ab.
  ivpTable->save( filename );
}

//-------------------------------------------------------------------------
// SEND (i.e. do) a specific SITUATION
//-------------------------------------------------------------------------
void 
SituationCollection::sendSituation(int numberOfSituationToBeSent) 
{
  if (   numberOfSituationToBeSent<0 
      || numberOfSituationToBeSent >= ivNumberOfSituations) 
    return;

  // Ein inaktiver Ball macht wenig Sinn!
  if (!ivActiveArray[0])
    return;

  //reset players' intentions
    //ACHTUNG: Dies ist ein grauenhafter Hack ((c) by Hauke Strasdat).
    //         Es wird ein String "(say (before_kick_off (reset int)))"
    //		   zusammengebaut und verschickt.
    //Der Parameter p="6" stellt dabei den Hack dar, der den Teilstring
    //"before_kick_off" auswaehlt. Ueber p wird auf ein Array a von Message-
    //Typen zugegriffen, das eigentlich nur 5 Elemente hat. Der tatsaechliche
    //erfolgende Zugriff a[6] bewirkt, dass das zweite Element (gluecklicherweise
    //ebenfalls ein Array von char*) des darauffolgenden Arrays zugegriffen
    //wird, in dem (vgl. defs.h) quasi durch Zufall die Zeichenkette
    //"before_kick_off" zu finden ist.
  MSG::sendMsg(MSG::MSG_SAY, 6, "reset int");
  
  //Hier wird die Zeichenkette "(change_mode play_on)" versendet.
  MSG::sendMsg(MSG::MSG_CHANGE_MODE, PM_play_on);
  
  char teamName[100],
       objectIdentifier[100];
  int  playerNumber = -1;
  float xCoordinate, yCoordinate, xVelocity, yVelocity, direction;
  //Iteration ueber alle (23) Elemente einer Situation
  for (int i=0; i<NUMBER_OF_OBJECTS_IN_SITUATION; i++)
  {
    if (!ivActiveArray[i])
      continue;
  	if (i==0) //Ball
  	  strcpy(objectIdentifier,"ball");
  	else
  	{
  	  if (i<=11) //my team
  	  {
  	    sprintf(teamName,"%s",fld.getMyTeamName());
  	    //Dekrementierung ist notwendig, da Arrays ab 0 zu zaehlen beginnen
  	    playerNumber = i-1;

        if (!fld.myTeam[playerNumber].alive) 
        {
          //continue;
        }
  	  }
  	  else
  	  {
  	    //i>11, i.e. opponent team
        sprintf(teamName, "%s", fld.getHisTeamName());
  	    playerNumber = i-12; //d.h. playerNumber = [12..22]-12 = [0..10]
//if (i==12) playerNumber=1; //<<== HACK FOR HASSLING :-) ZUI ZUI ZUI
//if (i==13) playerNumber=2;
        if (!fld.hisTeam[playerNumber].alive) continue;
  	  }
  	  //playerNumber+1 deshalb, weil der Server von 1..11 zaehlt
  	  sprintf(objectIdentifier, "player %s %d", teamName, playerNumber+1);
   	}
    //Korrekturterm (i==0?1:0) ist n?tig, weil f?r den Ball nur 4 Daten
    //in der Situation abgelegt sind (Ball hat keine Direction)
   	xCoordinate = ivpTable->getElement ( numberOfSituationToBeSent,
   	                                   i*5 - 1 + ((i==0?1:0)) );//0,4,9,...
   	yCoordinate = ivpTable->getElement ( numberOfSituationToBeSent,
   	                                   i*5 + 0 + ((i==0?1:0)) );//1,5,10,...
   	direction   = ivpTable->getElement ( numberOfSituationToBeSent,
   	                      			   i*5 + 1 ); //1,6,11,...
   	xVelocity   = ivpTable->getElement ( numberOfSituationToBeSent,
   	                                   i*5 + 2 ); //2,7,12,...
   	yVelocity   = ivpTable->getElement ( numberOfSituationToBeSent,
   	                                   i*5 + 3 ); //3,8,13,...
    if (direction >= PI) direction -= 2*PI;
    if (direction < -PI) direction += 2*PI;
if (i<2) cout<<"MOVE: "<<objectIdentifier<<" to "
         <<xCoordinate<<","
         <<yCoordinate
         <<" with "
         <<xVelocity<<","<<yVelocity<<"  and with direction="<<RAD2DEG(direction)<<endl<<flush;
    MSG::sendMsg( MSG::MSG_MOVE,
                  objectIdentifier,
                  xCoordinate,
                  -yCoordinate,
                  RAD2DEG(direction) ,
                  xVelocity,
                  -yVelocity);
  }
}
