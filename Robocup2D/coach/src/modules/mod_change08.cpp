/* Author: Thomas Gabel, 04/2008
 *
 * ModChange changes player types in own team.
 * See mod_change05.h for description.
 *
 */

#include "defs.h"
#include "coach.h"
#include "logger.h"
#include "valueparser.h"
#include "str2val.h"
#include "mod_change08.h"

const char ModChange08::modName[]="ModChange08";

//============================================================================
// init()
//============================================================================
/**
 * 
 */
bool ModChange08::init(int argc,char **argv) 
{
  ValueParser vp(Options::coach_conf,"ModChange08");
  //vp.get("fastest",fastestPlayers,3);
  //vp.get("wait_for_team",waitForTeam,1);

  ivSubstitutionQueueCounter = 0;
  ivTotalSubstitutions       = 0;
  ivPreparationDone          = false;
  ivCurrentScoreLeft         = 0;
  ivCurrentScoreRight        = 0;
  ivDoEmergencyChange        = false;

  //We hereby determine for which players heterogeneuous player types ought
  //to be assigned first (currently: prioritization for good defenders).
  ivInitialSubstitutionOrder[10].unum = 1;//goalie
  ivInitialSubstitutionOrder[10].role = GOALIE;
  ivInitialSubstitutionOrder[ 9].unum = 4;//sweeper
  ivInitialSubstitutionOrder[ 9].role = SWEEPER;
  ivInitialSubstitutionOrder[ 2].unum = 2;//defender left
  ivInitialSubstitutionOrder[ 2].role = DEFENDER;
  ivInitialSubstitutionOrder[ 0].unum = 3;//defender center
  ivInitialSubstitutionOrder[ 0].role = DEFENDER;
  ivInitialSubstitutionOrder[ 1].unum = 5;//defender right
  ivInitialSubstitutionOrder[ 1].role = DEFENDER;
  ivInitialSubstitutionOrder[ 6].unum = 7;//midfielder center
  ivInitialSubstitutionOrder[ 6].role = MIDFIELDER;
  ivInitialSubstitutionOrder[ 8].unum = 8;//midfielder right
  ivInitialSubstitutionOrder[ 8].role = MIDFIELDER;
  ivInitialSubstitutionOrder[ 7].unum = 6;//midfielder left
  ivInitialSubstitutionOrder[ 7].role = MIDFIELDER;
  ivInitialSubstitutionOrder[ 3].unum = 10;//offender center
  ivInitialSubstitutionOrder[ 3].role = OFFENDER;
  ivInitialSubstitutionOrder[ 4].unum = 11;//offender right
  ivInitialSubstitutionOrder[ 4].role = OFFENDER;
  ivInitialSubstitutionOrder[ 5].unum = 9;//offender left
  ivInitialSubstitutionOrder[ 5].role = OFFENDER;
  
  //TG17: NAGOYA HACK FOR FINAL DAY
  if ( strlen(fld.getHisTeamName()) > 0 )
  {
    if (   strstr( fld.getHisTeamName(), "elios" ) != NULL
        || strstr( fld.getHisTeamName(), "ELIOS" ) != NULL )
    {
      ivInitialSubstitutionOrder[ 8].role = DEFENDER;// MIDFIELDER;
      ivInitialSubstitutionOrder[ 7].role = DEFENDER;// MIDFIELDER;
    }
  }

  for (int i=0; i<PlayerParam::player_types; i++) 
  {
    ivTypesOnField[i] = 0;
    ivpPlayerTypesThatWereOnFieldAlready[i] = 0;
  }
  
  return true;
}

//============================================================================
// init()
//============================================================================
/**
 * 
 */
bool ModChange08::destroy() 
{
//  cout << "\nModChange08 destroy!";
  return true;
}

//============================================================================
// behave()
//============================================================================
/**
 * 
 */
bool 
ModChange08::behave() 
{
  this->showSubsQueue();

  if (fld.getTime() > 0)
    this->checkForSubstitutingStaminaLackingPlayers();
  
  this->showSubsQueue();
  
  /*************************************************************************/
  /** Check queue for succeeded substitutions and remove completed entries */

  if (ivSubstitutionQueueCounter > 0) 
    this->sendQueue();
  
  MsgChangePlayerType * msg
    = (MsgChangePlayerType*) MSG::msg[MSG::MSG_CHANGE_PLAYER_TYPE];
  int i,j;
  
  for (i=0; i<ivSubstitutionQueueCounter; i++) 
  {
    if (ivSubstitutionQueue[i].time >=0 ) 
    {
      for( j=msg->subsCnt-1; j>=0; j-- ) 
      {
        if ( msg->subsDone[j].done != 1) 
          continue;
        if (    msg->subsDone[j].unum != ivSubstitutionQueue[i].unum 
             || msg->subsDone[j].type != ivSubstitutionQueue[i].type)
          continue;
        msg->subsDone[j].done = 2;
        ivSubstitutionQueue[i].done = 1;
        if ( fld.getTime() > 0 ) 
          ivTotalSubstitutions++;
        break;
      }
      
      if( ivSubstitutionQueue[i].done == 0 ) 
      {
        if ( fld.getTime() - ivSubstitutionQueue[i].time > 5 ) 
        {
          LOG_ERR(0,<< "ModChange08: Could not change player " 
                    << ivSubstitutionQueue[i].unum << " to type "
                    << ivSubstitutionQueue[i].type << " within 5 cycles!");
          ivSubstitutionQueue[i].done = -1;
	      }
      }
    }
  }

  this->showSubsQueue();
  
  int  newCounter = ivSubstitutionQueueCounter;
  for ( i=0,j=0; i<ivSubstitutionQueueCounter; i++) 
  {
    if ( ivSubstitutionQueue[i].done != 0 ) 
    {
      newCounter--;
      continue;
    }
    ivSubstitutionQueue[j].unum = ivSubstitutionQueue[i].unum;
    ivSubstitutionQueue[j].type = ivSubstitutionQueue[i].type;
    ivSubstitutionQueue[j].time = ivSubstitutionQueue[i].time;
    ivSubstitutionQueue[j].done = ivSubstitutionQueue[i].done;
    j++;
  }
  ivSubstitutionQueueCounter = newCounter;

  /***************************************************************************/
  /**                                                                        */

  if ( ! ivPreparationDone ) 
    this->prepareForGame();


  for ( int i=0; i<PlayerParam::player_types; i++) 
    ivTypesOnField[i] = 0;
  for ( int i=0; i<11; i++) 
  {
    if ( !fld.myTeam[i].alive ) 
      continue;
    ivTypesOnField[fld.myTeam[i].type]++;
  }

  //cout <<"\n";
  //for(int i=0;i<PlayerParam::player_types;i++)
  //  cout << "["<<typesOnFld[i]<<"]";

  if (ivDoEmergencyChange) 
  {
    if (   fld.getTime()>=2800 
           //ACHTUNG: Hier fehlt die Unterscheidung, ob wir links
           //         oder recht spielen!
        && ivCurrentScoreLeft <= ivCurrentScoreRight) 
    {
      changePlayer( 9, ivEmergencyPlayerType);
      changePlayer(10, ivEmergencyPlayerType);
      changePlayer(11, ivEmergencyPlayerType);
      //std::cerr << "\n\n\nCHANGING!!!\n\n\n";
      ivDoEmergencyChange = false;
    }
  }
  this->showSubsQueue();
  return true;
}

//============================================================================
// onRefereeMessage()
//============================================================================
/**
 * 
 */
bool ModChange08::onRefereeMessage(bool PMChange) 
{
  if (    fld.getPM() != PM_play_on 
       && fld.getPM()  < PM_MAX) 
    this->sendQueue();
  if ( fld.getPM() == RM_goal_l) 
    ivCurrentScoreLeft++;
  if ( fld.getPM() == RM_goal_r) 
    ivCurrentScoreRight++;
  return true;
}


//============================================================================
// onKeyboardInput()
//============================================================================
/**
 * 
 */
bool ModChange08::onKeyboardInput(const char *str) 
{
  const char *sdum;
  int unum,type;

  if ( strskip(str,"chng ",sdum)) 
  {
    str = sdum;
    if (    str2val(str,unum,str) 
         && str2val(str,type,str) ) 
    {
      changePlayer(unum,type);
    }
    return true;
  }
  return false;
}

//============================================================================
// changePlayer()
//============================================================================
/**
 * 
 */
bool ModChange08::changePlayer(int unum,int type) 
{
  if ( ivSubstitutionQueueCounter >= 11 ) 
  {
    LOG_ERR(0,<<"ModChange08: Too many substitutions during this play_on cycle (cnt="
      <<ivSubstitutionQueueCounter<<")!");return false;
  }
  //std::cerr << "\nchange: "<< unum <<" -> "<<type;
  ivSubstitutionQueue[ivSubstitutionQueueCounter].unum   = unum;
  ivSubstitutionQueue[ivSubstitutionQueueCounter].type   = type;
  ivSubstitutionQueue[ivSubstitutionQueueCounter].time   = -1;
  ivSubstitutionQueue[ivSubstitutionQueueCounter++].done = 0;
  ivSubstitutionsToDo++;
  if (    fld.getPM() != PM_play_on 
       && fld.getPM()  < PM_MAX) 
    this->sendQueue();
  return true;
}

//============================================================================
// sendQueue()
//============================================================================
/**
 * 
 */
bool ModChange08::sendQueue() 
{
  if ( ivSubstitutionQueueCounter > 0 ) 
  {
    if (    fld.getPM() != PM_play_on 
         && fld.getPM()  < PM_MAX) 
    {
      for ( int i=0; i<ivSubstitutionQueueCounter; i++) 
      {
        if (    ! ivSubstitutionQueue[i].done 
             &&   ivSubstitutionQueue[i].time < 0 ) 
        {
          for ( int j=0; j<11; j++) 
          {
            if (    fld.myTeam[j].number == ivSubstitutionQueue[i].unum 
                 && fld.myTeam[j].alive ) 
            {
              LOG_MSG(0,<<"ModChange08: NOW, I DO THE CHANGE! (unum="
                <<ivSubstitutionQueue[i].unum<<", type="
                <<ivSubstitutionQueue[i].type<<")");
              sendMsg( MSG::MSG_CHANGE_PLAYER_TYPE,
                       ivSubstitutionQueue[i].unum,
                       ivSubstitutionQueue[i].type);
	            ivSubstitutionQueue[i].time = fld.getTime();
              ivpPlayerTypesThatWereOnFieldAlready[ivSubstitutionQueue[i].type] ++ ;
	            break;
            }
          }
        }
      }
    }
  }
  return true;
}

//============================================================================
// computeRealPlayerSpeedMax()
//============================================================================
/**
 * 
 */
int ModChange08::compareRealPlayerSpeedMaximum( const PlayerType *t1,
                                                const PlayerType *t2) 
{
  if(t1->real_player_speed_max > t2->real_player_speed_max) return 1;
  if(t1->real_player_speed_max < t2->real_player_speed_max) return -1;
  return 0;
}

//============================================================================
// compareStaminaDemandFor10Meters()
//============================================================================
/**
 * 
 */
int ModChange08::compareStaminaDemandFor10Meters( const PlayerType *t1,
                                                  const PlayerType *t2) 
{
   if(t1->stamina_10m > t2->stamina_10m) return 1;
   if(t1->stamina_10m < t2->stamina_10m) return -1;
   return 0;
}

//============================================================================
// fillPlayerTypeForRoleUsabilityMatrix()
//============================================================================
/**
 * 
 */
void
ModChange08::fillPlayerTypeForRoleUsabilityMatrix()
{
//printf("prepare for game\n");
  for (int i=0; i<MAX_PLAYER_TYPES; i++)
  {
    //printf("%d\t",i);
    for (int j=0; j<NUMBER_OF_PLAYER_ROLES; j++)
    {
      ivPlayerTypeForRoleUsability[j].pt[i] = i;
      ivPlayerTypeForRoleUsability[j].usability[i]
        = computePlayerTypeUsabilityForRole( & fld.plType[i], 
                                             (PLAYER_ROLES)j );
      //printf("%.3f\t",this->ivPlayerTypeForRoleUsability[j].usability[i]);
    }
    //printf("\n");
  }     
  for (int j=0; j<NUMBER_OF_PLAYER_ROLES; j++)
    this->ivPlayerTypeForRoleUsability[j].sort();
}

//============================================================================
// PlayerTypeUsability::sort()
//============================================================================
/**
 * 
 */
void
ModChange08::PlayerTypeUsability::sort()
{
  for (int i=0; i<MAX_PLAYER_TYPES; i++)
    for (int j=0; j<MAX_PLAYER_TYPES - i - 1; j++)
      if ( usability[j] < usability[j+1])
      {
        int tmpPT;
        float tmpU;
        tmpU           = usability[j];     tmpPT   = pt[j];        
        usability[j]   = usability[j+1];   pt[j]   = pt[j+1];        
        usability[j+1] = tmpU;             pt[j+1] = tmpPT;
      }
}


//============================================================================
// computePlayerTypeUsabilityForRole()
//============================================================================
/**
 * 
 */
float 
ModChange08::computePlayerTypeUsabilityForRole(PlayerType    * pt, 
                                               PLAYER_ROLES    role)
{
  float weightRealPlayerSpeedMax,    optimalityRealPlayerSpeedMax,
        weightStaminaDemandPerMeter, optimalityStaminaDemandPerMeter,
        weightSpeedProgression,      optimalitySpeedProgression,
        weightKickableMargin,        optimalityKickableMargin,
        weightKickRandomization,     optimalityKickRandomization;
  
  switch (role)
  {
    case GOALIE:
      weightRealPlayerSpeedMax    = 0.0;
      weightStaminaDemandPerMeter = 0.0;
      weightSpeedProgression      = 0.0;
      weightKickableMargin        = 0.0;
      weightKickRandomization     = 0.0;
    break;
    case SWEEPER:
      weightRealPlayerSpeedMax    = 0.3;
      weightStaminaDemandPerMeter = 0.4;
      weightSpeedProgression      = 0.2;
      weightKickableMargin        = 0.1;
      weightKickRandomization     = 0.0;
    break;
    case DEFENDER:
      weightRealPlayerSpeedMax    = 0.15;
      weightStaminaDemandPerMeter = 0.2;
      weightSpeedProgression      = 0.6;
      weightKickableMargin        = 0.05;
      weightKickRandomization     = 0.0;
    break;
    case MIDFIELDER:
      weightRealPlayerSpeedMax    = 0.2;
      weightStaminaDemandPerMeter = 0.5;
      weightSpeedProgression      = 0.2;
      weightKickableMargin        = 0.1;
      weightKickRandomization     = 0.0;
    break;
    case OFFENDER:
      weightRealPlayerSpeedMax    = 0.35;
      weightStaminaDemandPerMeter = 0.25;
      weightSpeedProgression      = 0.25;
      weightKickableMargin        = 0.075;
      weightKickRandomization     = 0.075;
    break;
  }
  optimalityRealPlayerSpeedMax    
    = computePlayerTypeRealPlayerSpeedMaxOptimality(pt);
  optimalityStaminaDemandPerMeter
    = computePlayerTypeStaminaDemandPerMeterOptimality(pt);
  optimalitySpeedProgression
    = computePlayerTypeSpeedProgressionOptimality(pt);
  optimalityKickableMargin
    = computePlayerTypeKickableMarginOptimality(pt);
  optimalityKickRandomization
    = computePlayerTypeKickRandomizationOptimality(pt);
  return //AMALGAMATION!
      weightRealPlayerSpeedMax    * optimalityRealPlayerSpeedMax
    + weightStaminaDemandPerMeter * optimalityStaminaDemandPerMeter
    + weightSpeedProgression      * optimalitySpeedProgression
    + weightKickableMargin        * optimalityKickableMargin
    + weightKickRandomization     * optimalityKickRandomization;
}

//============================================================================
// computePlayerTypeRealPlayerSpeedMaxOptimality()
//============================================================================
/**
 * 
 */
float 
ModChange08::computePlayerTypeRealPlayerSpeedMaxOptimality(const PlayerType * pt)
{
  float min = 0.75, //TODO: Should be read from server parameters! 
        max = 1.05, //TG09: changed values!
        ptV = pt->real_player_speed_max;
  if (ptV > max) ptV = max;    if (ptV < min) ptV = min;        
  //cout<<"RPSMCONSIDERATION: "<<pt->real_player_speed_max<<" ==> "<<((ptV-min) / (max-min))<<endl;
  return (ptV-min) / (max-min);
}
//============================================================================
// computePlayerTypeStaminaDemandPerMeterOptimality()
//============================================================================
/**
 * 
 */
float 
ModChange08::computePlayerTypeStaminaDemandPerMeterOptimality(const PlayerType * pt)
{
  float min = 1.0, //40.0, 
        max = 2.0, //55.0,
        ptV = pt->stamina_demand_per_meter / pt->stamina_inc_max;
  if (ptV > max) ptV = max;    if (ptV < min) ptV = min;        
  //cout<<"STAMINACONSIDERATION: "<<pt->stamina_demand_per_meter<<" / "<<pt->stamina_inc_max<<" = "<<ptV<<" ==>"<<1.0 - ((ptV-min) / (max-min))<<endl;
  return 1.0 - ((ptV-min) / (max-min));
}
//============================================================================
// computePlayerTypeSpeedProgressionOptimality()
//============================================================================
/**
 * 
 */
float 
ModChange08::computePlayerTypeSpeedProgressionOptimality(const PlayerType * pt)
{
  int   speedProgressSteps = 4;
  float min = (float)speedProgressSteps * 0.6, 
        max = (float)speedProgressSteps * 1.05, //TG09: changed
        ptV = 0.0;
  for (int i=0; i<speedProgressSteps; i++)
    ptV += pt->speed_progress[i];
  if (ptV > max) ptV = max;    if (ptV < min) ptV = min;        
  //cout<<"SPEEDPROGRCONSIDERATION: "<<pt->speed_progress[0]<<" + "<<pt->speed_progress[1]<<" + "<<pt->speed_progress[2]<<" = "<<ptV<<" ==>"<<(ptV-min) / (max-min)<<endl;
  return (ptV-min) / (max-min);
}
//============================================================================
// computePlayerTypeKickableMarginOptimality()
//============================================================================
/**
 * 
 */
float 
ModChange08::computePlayerTypeKickableMarginOptimality(const PlayerType * pt)
{
  float min = 0.6 * 0.6, //TODO: Should be read from server parameters. 
        max = 0.8 * 0.8, 
        ptV = pt->kickable_margin * pt->kickable_margin;
  if (ptV > max) ptV = max;    if (ptV < min) ptV = min;        
  return (ptV-min) / (max-min);
}
//============================================================================
// computePlayerTypeKickRandomizationOptimality()
//============================================================================
/**
 * 
 */
float 
ModChange08::computePlayerTypeKickRandomizationOptimality(const PlayerType * pt)
{
  float min = 0.0, //TODO: Should be read from server parameters. 
        max = 0.2,
        ptV = pt->kick_rand;
  if (ptV > max) ptV = max;    if (ptV < min) ptV = min;        
  return 1.0 - ( (ptV-min) / (max-min) );
}


//============================================================================
// prepareForGame() 
//============================================================================
/**
 * This method can be considered as this module's most important one.
 * It determines the initial employment of heterogeneous player types.
 * 
 * In the old days prepareForGames() mainly considered a player's real
 * speed maximum and/or the players stamina demand in order to decide for
 * certain player types. It used player types considered best for offenders,
 * followed by midfielders. The worst player types we used for defense.
 * Furthermore, the old version of this module (before ModChange08) used
 * to have a bug which hindered our coach from using heterogeneous player
 * types for our defenders at all.
 * 
 */
void ModChange08::prepareForGame() 
{
  //I should wait until my team is complete.
  if ( ivWaitForTeam) 
  {
    int cnt=0;
    for (int i=0; i<NUM_PLAYERS; i++) 
      if(fld.myTeam[i].alive) cnt++;
    if (cnt<11) return;
  }

  //Let's determine the quality of all player types!
  this->fillPlayerTypeForRoleUsabilityMatrix();
  
  int numberOfPlayersOfPlayerType[PlayerParam::player_types];
  for (int i=0; i<PlayerParam::player_types; i++) 
    numberOfPlayersOfPlayerType[i] = 0;
  for (int i=0; i<NUM_PLAYERS; i++)
  {
    int playerNumberToBeConsidered  = ivInitialSubstitutionOrder[i].unum;
    PLAYER_ROLES roleToBeConsidered = ivInitialSubstitutionOrder[i].role;
    //cout<<"############### "<<i<<" ==> "<<playerNumberToBeConsidered<<"("<<roleToBeConsidered<<") #############"<<endl<<flush;
    int indexInSortedList = 0;
    bool isChosenPlayerTypeAllowed = false;
    do
    {
      //cout<<"indexInSL="<<indexInSortedList<<flush;
      int chosenPlayerType;
      float chosenPlayerTypesUsability;
      if (indexInSortedList < MAX_PLAYER_TYPES )
      {
        chosenPlayerType
          = ivPlayerTypeForRoleUsability[roleToBeConsidered].pt[indexInSortedList];
        chosenPlayerTypesUsability
          = ivPlayerTypeForRoleUsability[roleToBeConsidered].
              usability[indexInSortedList];
      }
      else
      {
        chosenPlayerType = 0; chosenPlayerTypesUsability = 0.0;
      }
      //cout<<" chosPT="<<chosenPlayerType<<" chosPTUsab="<<chosenPlayerTypesUsability<<flush;
      if (MULTIPLE_DEFAULT_PT_ALLOWED)
      {
        //We allow usabilities of more than 0.5 only. If a player type has not
        //at least that usability we make use of the default player type.
        //                              ###########
        if (chosenPlayerTypesUsability < 0.25)//###
          chosenPlayerType = 0;//       ###########
      }
      if (   (   (   chosenPlayerType == 0 
                  && MULTIPLE_DEFAULT_PT_ALLOWED == true)
              ||   numberOfPlayersOfPlayerType[ chosenPlayerType ] 
                 < MAX_PLAYERS_OF_PLAYER_TYPE
             )
          && ! (    chosenPlayerType == 0
                 && MULTIPLE_DEFAULT_PT_ALLOWED == false
                 && GOALIE_MUST_BE_DEFAULT_PT == true
                 && roleToBeConsidered != GOALIE )
         )  
      {
        isChosenPlayerTypeAllowed = true;
        changePlayer( playerNumberToBeConsidered, chosenPlayerType );
        numberOfPlayersOfPlayerType[chosenPlayerType] ++ ;
        //cout<<" numOfPT="<<numberOfPlayersOfPlayerType[chosenPlayerType]<<endl<<flush;
      }
      indexInSortedList ++ ;
      //cout<<endl<<flush;
    }
    while ( ! isChosenPlayerTypeAllowed);    
    //cout<<endl<<flush;
  }

  ivPreparationDone = true;
}

void
ModChange08::checkForSubstitutingStaminaLackingPlayers()
{
  if (fld.getTime() == 10*ServerParam::half_time*ServerParam::slow_down_factor)
    for (int i=0; i<NUM_PLAYERS; i++)
    {
     fld.myTeam[i].staminaCapacityBound 
       = PLAYER_STAMINA_CAPACITY + ServerParam::stamina_max;
    }
   
  float myGoalX = (RUN::side==RUN::side_LEFT)?-52.0:52.0;
  float delta = sqrt(  (fld.ball.pos_x - myGoalX)*(fld.ball.pos_x - myGoalX)
                     + (fld.ball.pos_y)*(fld.ball.pos_y));
  if (delta < 25.0) delta /= 3.0;
  else if (delta < 50.0) delta /= 2.0;
  else delta /= 1.0;
  if (   (   fld.getTime() >= 10*ServerParam::half_time*ServerParam::slow_down_factor - (int)delta
          && fld.getTime() <= 10*ServerParam::half_time*ServerParam::slow_down_factor )
      || (   fld.getTime() >= 2 * 10*ServerParam::half_time*ServerParam::slow_down_factor - (int)delta
          && fld.getTime() <= 2 * 10*ServerParam::half_time*ServerParam::slow_down_factor )
     )
  {
    LOG_MSG(0,<<"ModChange08: I reset the ivSubstitutionQueue because it's half time, soon.");
    ivSubstitutionQueueCounter = 0;
    for (int q=0; q<11; q++) ivSubstitutionQueue[q].done = 1;
    return;
  }
  
/*  bool allowedPlayerTypes[PlayerParam::player_types];
  for (int t=0; t<PlayerParam::player_types; t++) 
    allowedPlayerTypes[t] 
      = (ivpPlayerTypesThatWereOnFieldAlready[t] < PlayerParam::pt_max) ;
  */
  //alle spieler betrachten, den schlimmsten selektieren
  int worstPlayerStamina = 1000000, worstPlayerNumber = -1;
  for (int i=0; i<NUM_PLAYERS; i++)
  {
    if (   fld.myTeam[i].staminaCapacityBound < worstPlayerStamina
        && isStaminaLevelCriticalForPlayer( i )
        && isPlayerInSubsQueue(i+1) == false )
    {
      worstPlayerStamina = fld.myTeam[i].staminaCapacityBound;
      worstPlayerNumber  = i; 
    }
  }
  //betrachtung des schlimmsten spielers
  if (   worstPlayerNumber != -1
      && ivSubstitutionQueueCounter + ivTotalSubstitutions 
         < PlayerParam::subs_max )
  {
    //suche nach dem besten ersatz
    ModChange08::PLAYER_ROLES 
      worstPlayerRole = getPlayerRoleForPlayer(worstPlayerNumber+1);
    LOG_MSG(0,<<"ModChange08: HELP! I need a new player for #"<<worstPlayerNumber+1
      <<", because that player has only "<<worstPlayerStamina
      <<" remaining stamine. Role: "<<worstPlayerRole
      <<". ivSubstitutionQueueCounter="<<ivSubstitutionQueueCounter
      <<", ivTotalSubstitutions="<<ivTotalSubstitutions);
      
    int playerNumberToBeConsidered  = worstPlayerNumber + 1;
    PLAYER_ROLES roleToBeConsidered = worstPlayerRole;
    
    int indexInSortedList = 0;
    bool isChosenPlayerTypeAllowed = false;
    do
    {
      //cout<<"indexInSL="<<indexInSortedList<<flush;
      int chosenPlayerType;
      float chosenPlayerTypesUsability;
      if (indexInSortedList < PlayerParam::player_types )
      {
        chosenPlayerType
          = ivPlayerTypeForRoleUsability[roleToBeConsidered].pt[indexInSortedList];
        chosenPlayerTypesUsability
          = ivPlayerTypeForRoleUsability[roleToBeConsidered].
              usability[indexInSortedList];
      }
      else
      {
        chosenPlayerType = 0; chosenPlayerTypesUsability = 0.0;
      }
      //cout<<" chosPT="<<chosenPlayerType<<" chosPTUsab="<<chosenPlayerTypesUsability<<flush;
      if (MULTIPLE_DEFAULT_PT_ALLOWED)
      {
        //We allow usabilities of more than 0.5 only. If a player type has not
        //at least that usability we make use of the default player type.
        //                              ###########
        if (chosenPlayerTypesUsability < 0.25)//###
          chosenPlayerType = 0;//       ###########
      }
      int asManyTimesIsChosenPlayerTypeAlreadyQueued = 0;
      for (int q=0; q<ivSubstitutionQueueCounter; q++)
        if (   ivSubstitutionQueue[q].done == 0
            && ivSubstitutionQueue[q].type == chosenPlayerType )
          asManyTimesIsChosenPlayerTypeAlreadyQueued ++ ;
      if (   (   (   chosenPlayerType == 0 
                  && MULTIPLE_DEFAULT_PT_ALLOWED == true)
              ||     ivpPlayerTypesThatWereOnFieldAlready[ chosenPlayerType ]
                   + asManyTimesIsChosenPlayerTypeAlreadyQueued
                 < PlayerParam::pt_max
             )
          && ! (    chosenPlayerType == 0
                 && MULTIPLE_DEFAULT_PT_ALLOWED == false
                 && GOALIE_MUST_BE_DEFAULT_PT == true
                 && roleToBeConsidered != GOALIE )
         )  
      {
        isChosenPlayerTypeAllowed = true;
        LOG_MSG(0,<<"ModChange08: HELP FOUND. Replace "<<playerNumberToBeConsidered
          <<" by new type "<<chosenPlayerType);
        changePlayer( playerNumberToBeConsidered, chosenPlayerType );
        //cout<<" numOfPT="<<numberOfPlayersOfPlayerType[chosenPlayerType]<<endl<<flush;
      }
      indexInSortedList ++ ;
      LOG_MSG(0,<<"ModChange08: "<<indexInSortedList);
      //cout<<endl<<flush;
    }
    while (    ! isChosenPlayerTypeAllowed
            && indexInSortedList <= PlayerParam::player_types);    
      
  }  
/*  
  int aFreeType = -1;
  for (int t=0; t<18; t++) 
    if (isChosenPlayerTypeAllowed[t] == true)
      aFreeType = t;
  LOG_MSG(0,<<"ModChange08: aFreeType="<<aFreeType);
  if (fld.getTime() == 10)
    changePlayer(2,aFreeType);*/
}

bool 
ModChange08::isStaminaLevelCriticalForPlayer( int idx )
{
  if ( fld.myTeam[idx].staminaCapacityBound > 20000 )
    return false; 
  PLAYER_ROLES playerRole = getPlayerRoleForPlayer(idx+1);
  if (fld.getTime() > 2*10*ServerParam::slow_down_factor*ServerParam::half_time)
    return false; //ignore extra time
#if LOGGING && BASIC_LOGGING
  int currentTimeInHalftime = fld.getTime() % (10*ServerParam::slow_down_factor*ServerParam::half_time);
#endif
  int halfTime = (fld.getTime() < 10*ServerParam::slow_down_factor*ServerParam::half_time) ? 1 : 2;
  
  float usedStaminaSoFar = (float)(  PLAYER_STAMINA_CAPACITY + ServerParam::stamina_max
                                   - fld.myTeam[idx].staminaCapacityBound);
  float staminaPerTimeStep 
    = (halfTime==1) ? usedStaminaSoFar / (float)fld.getTime()
                    : usedStaminaSoFar / (float)(fld.getTime()-10*ServerParam::slow_down_factor*ServerParam::half_time);
  float remainingStaminaSufficesForAsManyTimeSteps
    = fld.myTeam[idx].staminaCapacityBound / staminaPerTimeStep;
  int timeStepsTillHalfTime 
    = (halfTime==1) ? 10*ServerParam::slow_down_factor*ServerParam::half_time - fld.getTime()
                    : 2*10*ServerParam::slow_down_factor*ServerParam::half_time - fld.getTime();
  
  int timeStepsWithoutStamina 
    = timeStepsTillHalfTime - remainingStaminaSufficesForAsManyTimeSteps;
  
  int criticalStaminalessTimeSteps = 0;

  // Values for 1st half time cearly increased, in order to avoid
  // substitutions before half time. TG16
  switch (playerRole)
  {
    case GOALIE: 
    {
      criticalStaminalessTimeSteps = (halfTime==1) ? 200//-100
                                                   : -200;
      break;
    }
    case SWEEPER: 
    {
      criticalStaminalessTimeSteps = (halfTime==1) ? 1000//100
                                                   : 0;
      break;
    }
    case DEFENDER: 
    {
      criticalStaminalessTimeSteps = (halfTime==1) ? 500//-50
                                                   : -100;
      break;
    }
    case MIDFIELDER: 
    {
      criticalStaminalessTimeSteps = (halfTime==1) ? 1000//200
                                                   : 0;
      break;
    }
    case OFFENDER: 
    {
      criticalStaminalessTimeSteps = (halfTime==1) ? 2000//500
                                                   : 150;
      break;
    }
  }
  if (criticalStaminalessTimeSteps < 0)
    criticalStaminalessTimeSteps 
      = (int)(  (float)criticalStaminalessTimeSteps
              * ((float)timeStepsTillHalfTime / ((float)ServerParam::half_time*10*ServerParam::slow_down_factor)) );
  
  LOG_MSG(2,<<"ModChange08: CRITICAL CHECK for #"<<(idx+1)
    <<" currentTimeInHalftime="<<currentTimeInHalftime
    <<" halfTime="<<halfTime
    <<" usedStaminaSoFar="<<usedStaminaSoFar
    <<" staminaPerTimeStep="<<staminaPerTimeStep
    <<" remainingStaminaSufficesForAsManyTimeSteps="<<remainingStaminaSufficesForAsManyTimeSteps
    <<" timeStepsTillHalfTime="<<timeStepsTillHalfTime
    <<" timeStepsWithoutStamina="<<timeStepsWithoutStamina
    <<" criticalStaminalessTimeSteps="<<criticalStaminalessTimeSteps
    );
  
  if (timeStepsWithoutStamina > criticalStaminalessTimeSteps)
    return true;
  return false;
}

bool 
ModChange08::isPlayerInSubsQueue( int number )
{
  for (int i=0; i<ivSubstitutionQueueCounter; i++)
  {
    LOG_MSG(0,<<"ModChange08: check for ivSubstitutionQueue at entry "<<i);
    if (   ivSubstitutionQueue[i].unum == number
        && ivSubstitutionQueue[i].done == 0 )
    {
      LOG_MSG(0,<<"ModChange08: player number "<<number<<" is in ivSubstitutionQueue");
      return true;
    }
  }
  LOG_MSG(0,<<"ModChange08: player number "<<number<<" is NOT in ivSubstitutionQueue");
  return false;
}

ModChange08::PLAYER_ROLES
ModChange08::getPlayerRoleForPlayer( int p )
{
  ModChange08::PLAYER_ROLES retVal;
  for (int i=0; i<TEAM_SIZE; i++)
    if (ivInitialSubstitutionOrder[i].unum == p)
      retVal = ivInitialSubstitutionOrder[i].role;
  return retVal;
}

void 
ModChange08::showSubsQueue()
{
  LOG_MSG(0,<<"ModChange08: ivSubstitutionQueueCounter = "<<ivSubstitutionQueueCounter)
  for (int i=0; i<11; i++)
  {
    LOG_MSG(0,<<"ModChange08: "<<i
      <<"\t unum:"<<ivSubstitutionQueue[i].unum
      <<"\t type:"<<ivSubstitutionQueue[i].type
      <<"\t time:"<<ivSubstitutionQueue[i].time
      <<"\t done:"<<ivSubstitutionQueue[i].done);
  }  
}

