/*
 * mod_coach_language_sender.cpp
 *
 *  Created on: 04.01.2016
 *      Author: tgabel
 */

#include "mod_coach_language_sender.h"
#include "logger.h"
#include <sstream>
#include <iomanip>

//############################################################################
// STATIC MEMBER DEFINITION AND INITIALIZATION
//############################################################################
const char ModCoachLanguageSender::modName[]="ModCoachLanguageSender";
ModCoachLanguageSender* ModCoachLanguageSender::cvpInstance = NULL;
const string ModCoachLanguageSender::CL_ADVICE_INFO_HEADER = "6000 (true) ";
const string ModCoachLanguageSender::CL_ADVICE_INFO_FOOTER = "";
const string ModCoachLanguageSender::CL_DEFINE_HEADER = "definerule R direc ( (true) ";
const string ModCoachLanguageSender::CL_DEFINE_FOOTER = " )";

//############################################################################
// CONSTRUCTOR / DESTRUCTOR
//############################################################################
ModCoachLanguageSender::ModCoachLanguageSender()
{
}

ModCoachLanguageSender::~ModCoachLanguageSender()
{
}

//############################################################################
// CLASS METHODS
//############################################################################

void
ModCoachLanguageSender::appendSendableData( stringstream& ss )
{
  if ( ivDOAInfo.ivTime >= 0 )          ivDOAInfo.appendToStream( ss );
  if ( ivHeteroPlInfo.ivTime >= 0 )     ivHeteroPlInfo.appendToStream( ss );
  if ( ivHisGoalieNumInfo.ivTime >= 0 ) ivHisGoalieNumInfo.appendToStream( ss );
  if ( ivOppTeamIdInfo.ivTime >= 0 )    ivOppTeamIdInfo.appendToStream( ss );
  if ( ivStaminaCapInfo.ivTime >= 0 )   ivStaminaCapInfo.appendToStream( ss );
}

/**
 * Main method of this module proceeds in three steps:
 * 1. update all information chunks (retrieve data)
 * 2. check whether conditions indicate that sending of data must be done
 * 3. if 2. returns true, send the data to the Soccer Server
 */
bool
ModCoachLanguageSender::behave()
{
  //update all information holders that retrieve data by themselves
  ivStaminaCapInfo.update();
  ivHisGoalieNumInfo.update();
  ivOppTeamIdInfo.update();

  if ( shallISend() )
  {
    //do the sending
    LOG_FLD(1,<<"ModCoachLanguageSender: I SHALL SEND CL DATA IN T="<<fld.getTime()<<" AT PM="<<fld.getPM());
    if ( sendData() )
      resetAfterSending();
  }

  return true;
}

bool
ModCoachLanguageSender::destroy()
{
  return true;
}

bool
ModCoachLanguageSender::init(int argc,char **argv)
{
  return true;
}

/**
 * Singleton object access method
 */
ModCoachLanguageSender*
ModCoachLanguageSender::getInstance()
{
  if (cvpInstance == NULL )
    cvpInstance = new ModCoachLanguageSender();
  return cvpInstance;
}

/**
 * Format message according to coach language style (appropriate headers).
 */
void
ModCoachLanguageSender::prependHeaderForChannel( int channel, stringstream& ss )
{
  switch (channel)
  {
    case MSGT_ADVICE:
    case MSGT_INFO:
    {
      string copy = ss.str();
      ss.str( string() );
      ss << CL_ADVICE_INFO_HEADER << copy << CL_ADVICE_INFO_FOOTER;
      break;
    }
    case MSGT_DEFINE:
    {
      string copy = ss.str();
      ss.str( string() );
      ss << CL_DEFINE_HEADER << copy << CL_DEFINE_FOOTER;
      break;
    }
    case MSGT_FREEFORM:
    {
      //nothing to do, no header/footer
      break;
    }
    default:
    {
      LOG_FLD(0, << "ModCoachLanguageSender: ERROR: Undefined communication channel.");
    }
  }
}

/**
 * Data holders for all information chunks handled by this module will
 * be resetted.
 */
void
ModCoachLanguageSender::resetAfterSending()
{
  //flush all information holders
  ivDOAInfo.reset();
  ivStaminaCapInfo.reset();
  ivHeteroPlInfo.reset();
  ivHisGoalieNumInfo.reset();
  ivOppTeamIdInfo.reset();
}

/**
 * Coach language offers four communication channels: ADVICE, DEFINE, INFO
 * and FREEFORM.
 *
 * FREEFORM messages can be sent only during non-playon play modes, can
 * contain arbitrary data strings and will not be sent by this module.
 *
 * ADVICE, DEFINE and INFO messages can be sent during non-playon play modes
 * where they reach the players right in the subsequent time step as well as
 * in play mode playon, where they reach the players with a time delay of
 * 50 time steps (clang_delay).
 *
 * During play mode playon any of the three types of messages (ADVICE, DEFINE,
 * INFO) can be sent once within 300 time steps (clang_win_size). However,
 * each of these three types can carry the same information (though, eventually
 * differently formatted). So, this method determines which of the available
 * messaage types - here called 'communication channels' - should be used for
 * sending the next message.
 */
int
ModCoachLanguageSender::selectCommunicationChannel()
{
  if ( fld.getPM() == PM_play_on )
  {
    if (   fld.sayInfo.lastInPlayOn[MSGT_ADVICE] < 0
        ||    fld.getTime() - fld.sayInfo.lastInPlayOn[MSGT_ADVICE]
           >= ServerParam::clang_win_size)
      return MSGT_ADVICE;
    else
    if (   fld.sayInfo.lastInPlayOn[MSGT_INFO] < 0
        ||    fld.getTime() - fld.sayInfo.lastInPlayOn[MSGT_INFO]
           >= ServerParam::clang_win_size)
      return MSGT_INFO;
    else
    if (   fld.sayInfo.lastInPlayOn[MSGT_DEFINE] < 0
        ||    fld.getTime() - fld.sayInfo.lastInPlayOn[MSGT_DEFINE]
           >= ServerParam::clang_win_size)
      return MSGT_DEFINE;
  }
  else
  {
    return MSGT_ADVICE;
  }
  return -1;
}

/**
 * Send data to the Soccer Server:
 * 1. select a communication channels (message type)
 *    NOTE: It may happen that no communication channel is available, in case
 *          of which nothing can be sended. This case occurs, for example, if
 *          in play mode play on, a lot of data has just recently been sended.
 * 2. construct a message, put in all data to be sended
 * 3. wrap the appropriate headers around the message data
 * 4. send it
 */
bool
ModCoachLanguageSender::sendData()
{
  // select communication channel
  int channel = selectCommunicationChannel();
  if (channel == -1)
  {
    LOG_FLD(0,<< "ModCoachLanguageSender: Tried sending in t=" << fld.getTime()
              << ", but no communication channel is available.");
    return false;
  }
  LOG_FLD(0,<<"ModCoachLanguageSender: Selected communication channel is " << channel);

  // create send stream string
  stringstream ss;
  appendSendableData( ss );
  prependHeaderForChannel( channel, ss );

  // go and send it
  sendMsg(MSG::MSG_SAY,channel,ss.str());
  LOG_FLD(0, << "ModCoachLanguageSender: Sending to server in pm="<< fld.getPM()
             << " via channel="<<channel<<": "<< ss.str() << endl);
  stringstream ss2;
  ss2 << "ModCoachLanguageSender: DOATRACK: t="<<fld.getTime()<<" pm="<<fld.getPM()<<"\t: ";
  if (ivDOAInfo.ivTime > -1)
  {
    for (int i=0; i<NUM_PLAYERS; i++)
    {
      ss2 << setw(4) << ivDOAInfo.ivDOAs[i];
    }
    LOG_FLD(0,<< ss2.str() );
  }
  return true;
}

/*
 * This method implements the logic that decides whether to send data to
 * the Soccer Server in the current time step.
 *
 * The decision is based on whether up-to-date data from the different
 * information chunks is available (indicated by ivTime>=0).
 * 1. new direct opponent assignments -> always send
 * 2. new/changed goalie number -> always send
 * 3. changed team identifier -> always send
 * 4. new staminca capacity information -> always send in non-playon play modes
 * 5. new heterogeneous player information -> never send it before time step
 *    15 (it's almost impossible to have that information as early as that),
 *    after that time send it in non-playon play modes, if it hasn't been sent
 *    before or if at least 50 time steps have elapsed after the recent sending
 */
bool
ModCoachLanguageSender::shallISend()
{
  LOG_FLD(0,<< "ModCoachLanguageSender: Shall I send? doa:" << ivDOAInfo.ivTime
            << " hpt:" << ivHeteroPlInfo.ivTime << " hgn:" << ivHisGoalieNumInfo.ivTime
            << " oti:" << ivOppTeamIdInfo.ivTime << " sci: " << ivStaminaCapInfo.ivTime );
  return (   ivDOAInfo.ivTime >= 0
          || (   ivHeteroPlInfo.ivTime >= 0 && fld.getTime() >= 15
              && (   fld.getTime() - ivHeteroPlInfo.ivLastTimeSent >= 50
                  || ivHeteroPlInfo.ivLastTimeSent < 0
                  || fld.getPM() != PM_play_on) )
          || ivHisGoalieNumInfo.ivTime >= 0
          || ivOppTeamIdInfo.ivTime >= 0
          || (ivStaminaCapInfo.ivTime >= 0 && fld.getPM() != PM_play_on)
         );
}

/**
 * Externally accessable method to set sendable information to this module.
 * (intended to be set by module ModAnalyse*)
 */
void
ModCoachLanguageSender::setHisHeteroPlayerType( int num, int type )
{
  if ( ivHeteroPlInfo.ivHPTs[num] != type )
  {
    ivHeteroPlInfo.ivTime = fld.getTime();
    LOG_FLD(0,<<"ModCoachLanguageSender: New hetero info: OPP #"<<num<<" is TYPE "<<type);
  }
  ivHeteroPlInfo.ivHPTs[num] = type;
}

/**
 * Externally accessable method to set sendable information to this module.
 * (intended to be set by module ModDirectOpponentAssignment*)
 */
void
ModCoachLanguageSender::setDirectOpponentAssignment( int my, int his )
{
  if ( ivDOAInfo.ivDOAs[my] != his )
  {
    ivDOAInfo.ivTime = fld.getTime();
    LOG_FLD(0,<<"ModCoachLanguageSender: New DOA marking: MY #" << my+1 << " marks HIS #" << his);
  }
  ivDOAInfo.ivDOAs[my] = his;
}


bool
ModCoachLanguageSender::onRefereeMessage(bool PMChange)
{
  return false;
}

bool
ModCoachLanguageSender::onHearMessage(const char *str)
{
  return false;
}

bool
ModCoachLanguageSender::onKeyboardInput(const char *str)
{
  return false;
}

bool
ModCoachLanguageSender::onChangePlayerType(bool,int,int)
{
  return false;
}

//############################################################################
// IMPLEMENTATION OF INNER CLASS DOAInformation
//############################################################################

ModCoachLanguageSender::DOAInformation::DOAInformation()
  : ivTime(-1)
{
  for (int i=0; i<NUM_PLAYERS; i++) this->ivDOAs[i] = -1;
}

/**
 * Format direct opponent data according to coach language (marking
 * directive) and write it into the stringstream that will be sended.
 */
void
ModCoachLanguageSender::DOAInformation::appendToStream( stringstream& ss )
{
  for (int i=0; i<NUM_PLAYERS; i++)
  {
    ss << "(do our {" << i+1 << "} (mark {"
       << ((ivDOAs[i] == -1) ? 0 : ivDOAs[i]) //negative numbers disallowed
       << "}) ) ";
  }
}

void
ModCoachLanguageSender::DOAInformation::reset()
{
  ivTime = -1;
}

//############################################################################
// IMPLEMENTATION OF INNER CLASS StaminaCapacityInformation
//############################################################################

ModCoachLanguageSender::StaminaCapacityInformation::StaminaCapacityInformation()
  : ivTime(-1)
{
  for (int i=0; i<NUM_PLAYERS; i++)
    this->ivSCs[i] = PLAYER_STAMINA_CAPACITY + ServerParam::stamina_max;
}

/**
 * Format direct opponent data according to coach language (home
 * directive) and write it into the stringstream that will be sended.
 *
 * Stamina capacity information of players is a number between zero and
 * 130600 (server::stamina_capacity_max). This value is encoded into the
 * four floating numbers of a home region directive (where a home region
 * is actually characterized by its top left and bottom right point).
 * We utilize a directive for the home region
 *   (home (rec (pt a.b c.d) (pt e.f g.h)) )
 * in which a.b, c.d, e.f, g.h are floating numbers for x and y values on
 * the field such that each digit encodes a digit from the current stamina
 * capacity value. Example:
 *    stamina capacity = 128901
 * => (home (rec (pt 0.0 1.2) (pt 8.9 0.1)) )
 */
void
ModCoachLanguageSender::StaminaCapacityInformation::appendToStream( stringstream& ss )
{
  int capData[8];
  for (int i=0; i<NUM_PLAYERS; i++)
  {
    int capacity = this->ivSCs[i];
    for (int k=0; k<8; k++)
    {
      capData[k] = capacity % 10;
      capacity /= 10;
    }
    //cout << i+1 << ": stamina at t="<<fld.getTime()<< ": " << this->ivSCs[i]
    //<< " " << capData[7] << " " << capData[6] << " " << capData[5] << " " << capData[4] << " " << capData[3] << " " << capData[2] << " " << capData[1] << " " << capData[0]
    //<< endl;
    ss << "(do our {" << i+1 << "} (home (rec (pt "
       << capData[7] << "." << capData[6] << " " << capData[5] << "." << capData[4]
       << ") (pt " << capData[3] << "." << capData[2] << " " << capData[1]
       << "." << capData[0] << ")))) ";
  }
}

/**
 * Retrieve stamina capacity information for all players from the global
 * fld data structure.
 */
bool
ModCoachLanguageSender::StaminaCapacityInformation::update()
{
  for (int i=0; i<NUM_PLAYERS; i++)
    if (fabs(ivSCs[i] - fld.myTeam[i].staminaCapacityBound) > 800 )
      ivTime = fld.getTime();

  for (int i=0; i<NUM_PLAYERS; i++)
  {
    if ( ivTime == fld.getTime() )
      this->ivSCs[i] = fld.myTeam[i].staminaCapacityBound;
  }
  return true;
}

void
ModCoachLanguageSender::StaminaCapacityInformation::reset()
{
  ivTime = -1;
}

//############################################################################
// IMPLEMENTATION OF INNER CLASS HeterogeneousPlayerInformation
//############################################################################

ModCoachLanguageSender::HeterogeneousPlayerInformation::HeterogeneousPlayerInformation()
  : ivTime(-1), ivLastTimeSent(-1)
{
  for (int i=0; i<NUM_PLAYERS; i++)
    this->ivHPTs[i] = -1;
}

/**
 * Format heterogeneous player data according to coach language (htype
 * directive) and write it into the stringstream that will be sended.
 */
void
ModCoachLanguageSender::HeterogeneousPlayerInformation::appendToStream( stringstream& ss )
{
  for (int i=0; i<NUM_PLAYERS; i++)
    ss << "(do opp {" << i+1 << "} (htype " << ivHPTs[i] << ")) ";
}

void
ModCoachLanguageSender::HeterogeneousPlayerInformation::reset()
{
  if (ivTime >= 0) ivLastTimeSent = ivTime;
  ivTime = -1;
}

//############################################################################
// IMPLEMENTATION OF INNER CLASS HisGoalieNumberInformation
//############################################################################

ModCoachLanguageSender::HisGoalieNumberInformation::HisGoalieNumberInformation()
  : ivTime(-1), ivHisGoalieNumber(-1)
{
}

/**
 * Format heterogeneous player data according to coach language (hold
 * directive) and write it into the stringstream that will be sended.
 *
 * Here, we exploit the hold directive to transmit his goalie's number.
 */
void
ModCoachLanguageSender::HisGoalieNumberInformation::appendToStream( stringstream& ss )
{
  ss << "(do opp {" << ivHisGoalieNumber << "} (hold) ) ";
}

/**
 * Retrieve information about his goalie's number from the global
 * fld data structure. Set ivTime to a non-zero value (which triggers
 * a sending), in case that the goalie number has actually changed.
 */
bool
ModCoachLanguageSender::HisGoalieNumberInformation::update()
{
  int goalie = -1;
  for (int i=0; i<NUM_PLAYERS; i++)
    if (fld.hisTeam[i].goalie)
    {
      goalie = fld.hisTeam[i].number;
      LOG_FLD(0, << "ModCoachLanguageSender: His goalie is #" << goalie);
    }
  if (goalie != ivHisGoalieNumber)
    ivTime = fld.getTime();
  ivHisGoalieNumber = goalie;
  if(ivHisGoalieNumber < 0 && fld.getTime()>=10)
  {
    LOG_FLD(0, <<"ModCoachLanguageSender: Opponent does not seem to have a goalie!");
  }
  return true;
}

void
ModCoachLanguageSender::HisGoalieNumberInformation::reset()
{
  ivTime = -1;
}

//############################################################################
// IMPLEMENTATION OF INNER CLASS OpponentTeamIdentifierInformation
//############################################################################

ModCoachLanguageSender::OpponentTeamIdentifierInformation::OpponentTeamIdentifierInformation()
  : ivTime(-1), ivOppTeamId(-1)
{
}

void
ModCoachLanguageSender::OpponentTeamIdentifierInformation::appendToStream( stringstream& ss )
{
  ss << "(do opp {" << ivOppTeamId << "} (shoot) ) ";
}

/**
 * Retrieve information about his team name from the global
 * fld data structure. Set ivTime to a non-zero value (which triggers
 * a sending), in case that the team name changed (which actually
 * cannot occur - hopefully).
 *
 * New team name identifiers can be added in defs.h
 *
 * ATTENTION: Oxsy is currently mapped to WrightEagle due to their
 *            similar way of playing. This should perhaps be changed. TODO
 */
bool
ModCoachLanguageSender::OpponentTeamIdentifierInformation::update()
{
  if ( strlen(fld.getHisTeamName()) == 0 )
  {
    ivTime = -1;
  }
  else
  {
    int oldId = ivOppTeamId;
    if (   strstr( fld.getHisTeamName(), "umbol" ) != NULL
        || strstr( fld.getHisTeamName(), "UMBOL" ) != NULL
        || strstr( fld.getHisTeamName(), "AT-H" ) != NULL )
      ivOppTeamId = TEAM_IDENTIFIER_ATHUMBOLDT - TEAM_IDENTIFIER_BASE_CODE;
    if (   strstr( fld.getHisTeamName(), "elios" ) != NULL
	    || strstr( fld.getHisTeamName(), "ELIOS" ) != NULL )
	  ivOppTeamId = TEAM_IDENTIFIER_HELIOS - TEAM_IDENTIFIER_BASE_CODE;
    if (   strstr( fld.getHisTeamName(), "WE20" ) != NULL
	    || strstr( fld.getHisTeamName(), "WE0" ) != NULL
	    || strstr( fld.getHisTeamName(), "Wright" ) != NULL
	    || strstr( fld.getHisTeamName(), "wright" ) != NULL
	   )
	  ivOppTeamId = TEAM_IDENTIFIER_WRIGHTEAGLE - TEAM_IDENTIFIER_BASE_CODE;
    if (   strstr( fld.getHisTeamName(), "xsy" ) != NULL
        || strstr( fld.getHisTeamName(), "XSY" ) != NULL )
      ivOppTeamId = TEAM_IDENTIFIER_OXSY - TEAM_IDENTIFIER_BASE_CODE;
    if (   strstr( fld.getHisTeamName(), "HERMES" ) != NULL 
        || strstr( fld.getHisTeamName(), "ermes" ) != NULL )
      ivOppTeamId = TEAM_IDENTIFIER_HERMES - TEAM_IDENTIFIER_BASE_CODE;
    if (   strstr( fld.getHisTeamName(), "CYRUS" ) != NULL
        || strstr( fld.getHisTeamName(), "yrus" ) != NULL )
      ivOppTeamId = TEAM_IDENTIFIER_CYRUS - TEAM_IDENTIFIER_BASE_CODE;
    if (   strstr( fld.getHisTeamName(), "liders" ) != NULL
        || strstr( fld.getHisTeamName(), "LIDERS" ) != NULL )
      ivOppTeamId = TEAM_IDENTIFIER_GLIDERS - TEAM_IDENTIFIER_BASE_CODE; 
    if (ivOppTeamId != oldId)
      ivTime = fld.getTime();
	LOG_FLD(0,<< "ModCoachLanguageSender: Opponent team ID: " << ivOppTeamId
	          << " (" << fld.getHisTeamName() << ")");
  }
  return true;
}

void
ModCoachLanguageSender::OpponentTeamIdentifierInformation::reset()
{
  ivTime = -1;
}
