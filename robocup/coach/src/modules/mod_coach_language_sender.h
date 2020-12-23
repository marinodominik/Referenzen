/*
 * mod_coach_language_sender.h
 *
 *  Created on: 04.01.2016
 *      Author: tgabel
 *
 * This coach module
 * (a) collects and stores data from all other modules that want to send
 *     coach language-based data to the Soccer Server
 * (b) and manages how to send this information in the appropriate format
 *     to the Soccer Server.
 *
 * The module currently handles the following chunks of data
 * 1. direct opponent assignments [<- ModDirectOpponentAssignment*]
 * 2. heterogeneous player type information [<- ModAnalyse*]
 * 3. his goalie number
 * 4. opponent team name / identifier
 * 5. stamina capacity of teammates as encoded by their pointing arm
 *
 * Information chunks 1-2 must be set by separate modules using methods
 * setDirectOpponentAssignment(.) and setHisHeteroPlayerType(.).
 * Data for information 3-5 are retrieved by this module on itw own from
 * global information (via fld.*).
 *
 * The module is implemented according to the singleton pattern.
 */

#ifndef MOD_COACH_LANGUAGE_SENDER_H_
#define MOD_COACH_LANGUAGE_SENDER_H_

#include "modules.h"

class ModCoachLanguageSender : public AbstractModule
{
  private:
    static ModCoachLanguageSender* cvpInstance;
    static const string CL_ADVICE_INFO_HEADER, CL_ADVICE_INFO_FOOTER,
                        CL_DEFINE_HEADER, CL_DEFINE_FOOTER;

    struct DOAInformation
    {
      int ivTime;
      int ivDOAs[NUM_PLAYERS];
      DOAInformation();
      void appendToStream( stringstream& ss );
      void reset();
    };
    struct HisGoalieNumberInformation
    {
      int ivTime;
      int ivHisGoalieNumber;
      HisGoalieNumberInformation();
      void appendToStream( stringstream& ss );
      bool update();
      void reset();
    };
    struct StaminaCapacityInformation
    {
      int ivTime;
      int ivSCs[NUM_PLAYERS];
      StaminaCapacityInformation();
      void appendToStream( stringstream& ss );
      bool update();
      void reset();
    };
    struct HeterogeneousPlayerInformation
    {
      int ivTime;
      int ivLastTimeSent;
      int ivHPTs[NUM_PLAYERS];
      HeterogeneousPlayerInformation();
      void appendToStream( stringstream& ss );
      void reset();
    };
    struct OpponentTeamIdentifierInformation
    {
      int ivTime;
      int ivOppTeamId;
      OpponentTeamIdentifierInformation();
      void appendToStream( stringstream& ss );
      bool update();
      void reset();
    };

    DOAInformation                    ivDOAInfo;
    HisGoalieNumberInformation        ivHisGoalieNumInfo;
    StaminaCapacityInformation        ivStaminaCapInfo;
    HeterogeneousPlayerInformation    ivHeteroPlInfo;
    OpponentTeamIdentifierInformation ivOppTeamIdInfo;

    //private methods
    void appendSendableData( stringstream& ss );
    void prependHeaderForChannel(int channel, stringstream& ss);
    void resetAfterSending();
    int  selectCommunicationChannel();
    bool sendData();
    bool shallISend();

  public:
    //singleton
    ModCoachLanguageSender();
    virtual ~ModCoachLanguageSender();
    static ModCoachLanguageSender* getInstance();

    //public functionality
    void setHisHeteroPlayerType( int num, int type );
    void setDirectOpponentAssignment( int my, int his );

    //default module methods
    bool init(int argc,char **argv);             /** init the module */
    bool destroy();                              /** tidy up         */

    bool behave();                               /** called once per cycle, after visual update   */
    bool onRefereeMessage(bool PMChange);        /** called on playmode change or referee message */
    bool onHearMessage(const char *str);         /** called on every hear message from any player */
    bool onKeyboardInput(const char *str);       /** called on keyboard input                     */
    bool onChangePlayerType(bool,int,int);       /** SEE mod_template.c!                          */

    //module definition
    static const char modName[];                 /** module name, should be same as class name    */
    const char* getModName() {return modName;}   /** do not change this!                          */

};

#endif /* MOD_COACH_LANGUAGE_SENDER_H_ */
