/*
 * mod_coach_comm_tester.cpp
 *
 *  Created on: 04.01.2016
 *      Author: tgabel
 *
 * This module is intended for testing purposes only. It allows for
 * defining arbitrary coach messages, sending them, and, in turn,
 * analysing how and whether they where received by the players.
 *
 * This module is not meant to be used during competitions, it is
 * merely a tool for playing around with the coach language.
 */

#include "mod_coach_comm_tester.h"
#include "logger.h"
#include "messages.h"

#include <sstream>

const char ModCoachCommTester::modName[]="ModCoachCommTester";

bool
ModCoachCommTester::init(int argc,char **argv)
{
  return true;
}

bool
ModCoachCommTester::destroy()
{
  return true;
}

bool
ModCoachCommTester::behave()
{
 //NOTE: advice/info/define support mark/home as well
 if (fld.getTime() % 5 == 1)
 {
  ostringstream oss;
  oss << "6000 (true) (do our {3} (mark {4}) ) (do our {4} (mark {5 7}) ) (do our {2} (home (rec (pt 30.0 12.5) (pt 14.6 20.3))))  ";
  //oss << "6000 (true) (do our {5} (mark {6}))"; //OK
  sendMsg(MSG::MSG_SAY,MSGT_ADVICE,oss.str());
  LOG_FLD(0, << "ModCoachCommTester::behave(): Sending to server: "<< oss.str() << endl);
  return true;
 }
 if (fld.getTime() % 20 == 6)
 {
  ostringstream oss;
  oss << "6000 (true) (do our {5} (mark {6}))"; //OK
  sendMsg(MSG::MSG_SAY,MSGT_ADVICE,oss.str());    //OK
  LOG_FLD(0, << "ModCoachCommTester::behave(): Sending to server: "<< oss.str() << endl);
  return true;
 }
 if (fld.getTime() % 20 == 12)
 {
  ostringstream oss;
  oss << "definerule R direc ( (true) (do our {2} (mark {3})) (do our {2} (home (rec (pt 30.0 12.5) (pt 14.6 20.3)))) )"; //OK
  sendMsg(MSG::MSG_SAY,MSGT_DEFINE,oss.str());    //OK
  LOG_FLD(0, << "ModCoachCommTester::behave(): Sending to server: "<< oss.str() << endl);
  return true;
 }
 if (fld.getPM() != PM_play_on)
 {
  ostringstream oss;
  oss << "6000 (true) (do our {5} (mark {"<<fld.getTime()<<"}))"; //OK
  sendMsg(MSG::MSG_SAY,MSGT_INFO,oss.str());    //OK
  LOG_FLD(0, << "ModCoachCommTester::behave(): Sending in NON-play-on to server: "<< oss.str() << endl);
 }
 //INFO: * nutze mark für doa inkl. gegnertormann DONE
 //      * nutze home für stamina capacity mit rec/pt als angabe der stamina
 //      * nutze markl für conflicts (oder ggf. gar nicht)
 //      * nutze htype für analyse/gegnertypen DONE
 //      * nutze tackle für gegnermannschaftsname
 return false;
}

bool
ModCoachCommTester::onRefereeMessage(bool PMChange)
{
  return false;
}

bool
ModCoachCommTester::onHearMessage(const char *str)
{
//  LOG_FLD(0,<<"ModCoachCommTester: I HEARD "<< str << endl);
  return false;
}

bool
ModCoachCommTester::onKeyboardInput(const char *str)
{
  return false;
}

bool
ModCoachCommTester::onChangePlayerType(bool,int,int)
{
  return false;
}
