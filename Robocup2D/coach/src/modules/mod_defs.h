/* Author: Manuel "Sputnick" Nickschas, 02/2002
 *
 * This file contains information about available modules.
 *
 * To add a module, change information here and don't forget
 * to add a section for your module in coach.conf, too!
 *
 */

#ifndef _MOD_DEFS_H_
#define _MOD_DEFS_H_

#include "str2val.h"

/** Include module header files here */

#include "mod_core.h"
#include "mod_change.h"
#include "mod_change05.h"
#include "mod_change08.h"
#include "mod_analyse.h"
#include "mod_analyse08.h"
#include "mod_art.h"
#include "mod_train.h"
#include "mod_demo_train.h"
#include "mod_bs2k.h"
#include "mod_setsit.h"
#include "mod_direct_opponent_assignment.h"
#include "mod_direct_opponent_assignment_10.h"
#include "mod_learnTrainer.h"
#include "mod_learnPositioningLearner.h"
#include "mod_analyse09.h"
#include "mod_analyse2010.h"
#include "mod_coach_comm_tester.h"
#include "mod_coach_language_sender.h"

/** Add a line for your module here */
AbstractModule *MOD::getModule(const char *name) {

  if(!strcmp(name,ModCore::modName)) {return new ModCore();}
  if(!strcmp(name,ModChange::modName)) {return new ModChange();}
  if(!strcmp(name,ModChange05::modName)) {return new ModChange05();}
  if(!strcmp(name,ModChange08::modName)) {return new ModChange08();}
  if(!strcmp(name,ModAnalyse::modName)) {return new ModAnalyse();}
  if(!strcmp(name,ModAnalyse08::modName)) {return new ModAnalyse08();}
  if(!strcmp(name,ModArt::modName)) {return new ModArt();}   
  if(!strcmp(name,ModTrain::modName)) {return new ModTrain();}
  if(!strcmp(name,ModDemoTrain::modName)) {return new ModDemoTrain();}
  if(!strcmp(name,ModBS2K::modName)) {return new ModBS2K();}   
  if(!strcmp(name,ModSetSit::modName)) {return new ModSetSit();}  
  if(!strcmp(name,ModDirectOpponentAssignment::modName)) 
    {return new ModDirectOpponentAssignment();}  
  if(!strcmp(name,ModLearnTrainer::modName)) {return new ModLearnTrainer();}  
  if(!strcmp(name,ModLearnPositioningLearner::modName)) 
    {return new ModLearnPositioningLearner();}
  if(!strcmp(name,ModAnalyse09::modName)) {return new ModAnalyse09();}
  if(!strcmp(name,ModAnalyse2010::modName)) {return new ModAnalyse2010();}
  if(!strcmp(name,ModDirectOpponentAssignment10::modName)) 
    {return new ModDirectOpponentAssignment10();}
  if(!strcmp(name,ModCoachCommTester::modName)) {return new ModCoachCommTester();}
  if(!strcmp(name,ModCoachLanguageSender::modName)) {return ModCoachLanguageSender::getInstance();}
  
  return NULL;
}

#endif
