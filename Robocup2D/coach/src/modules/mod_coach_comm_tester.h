/*
 * mod_coach_comm_tester.h
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

#ifndef _MOD_COACH_COMM_TESTER_H_
#define _MOD_COACH_COMM_TESTER_H_

#include "modules.h"

class ModCoachCommTester : public AbstractModule
{
  private:

  public:
    bool init(int argc,char **argv);             /** init the module */
    bool destroy();                              /** tidy up         */

    bool behave();                               /** called once per cycle, after visual update   */
    bool onRefereeMessage(bool PMChange);        /** called on playmode change or referee message */
    bool onHearMessage(const char *str);         /** called on every hear message from any player */
    bool onKeyboardInput(const char *str);       /** called on keyboard input                     */
    bool onChangePlayerType(bool,int,int);       /** SEE mod_template.c!                          */

    static const char modName[];                 /** module name, should be same as class name    */
    const char* getModName() {return modName;}   /** do not change this!                          */

};

#endif
