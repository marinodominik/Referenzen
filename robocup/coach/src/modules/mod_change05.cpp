/* Author: Thomas Gabel, 03/2005
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
#include "mod_change05.h"

const char ModChange05::modName[]="ModChange05";

//============================================================================
// init()
//============================================================================
/**
 * 
 */
bool ModChange05::init(int argc,char **argv) 
{
  ValueParser vp(Options::coach_conf,"ModChange05");
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
  ivInitialSubstitutionOrder[ 0].unum = 2;//defender left
  ivInitialSubstitutionOrder[ 0].role = DEFENDER;
  ivInitialSubstitutionOrder[ 1].unum = 3;//defender center
  ivInitialSubstitutionOrder[ 1].role = DEFENDER;
  ivInitialSubstitutionOrder[ 2].unum = 5;//defender right
  ivInitialSubstitutionOrder[ 2].role = DEFENDER;
  ivInitialSubstitutionOrder[ 6].unum = 7;//midfielder right
  ivInitialSubstitutionOrder[ 6].role = MIDFIELDER;
  ivInitialSubstitutionOrder[ 7].unum = 8;//midfielder center
  ivInitialSubstitutionOrder[ 7].role = MIDFIELDER;
  ivInitialSubstitutionOrder[ 8].unum = 6;//midfielder left
  ivInitialSubstitutionOrder[ 8].role = MIDFIELDER;
  ivInitialSubstitutionOrder[ 3].unum = 10;//offender right
  ivInitialSubstitutionOrder[ 3].role = OFFENDER;
  ivInitialSubstitutionOrder[ 4].unum = 11;//offender center
  ivInitialSubstitutionOrder[ 4].role = OFFENDER;
  ivInitialSubstitutionOrder[ 5].unum = 9;//offender left
  ivInitialSubstitutionOrder[ 5].role = OFFENDER;
  
  for (int i=0; i<PlayerParam::player_types; i++) 
    ivTypesOnField[i] = 0;

  return true;
}

//============================================================================
// init()
//============================================================================
/**
 * 
 */
bool ModChange05::destroy() 
{
  cout << "\nModChange05 destroy!";
  return true;
}

//============================================================================
// behave()
//============================================================================
/**
 * 
 */
bool 
ModChange05::behave() 
{
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
          LOG_ERR(0,<< "Could not change player " 
                    << ivSubstitutionQueue[i].unum << " to type "
                    << ivSubstitutionQueue[i].type << " within 5 cycles!");
          ivSubstitutionQueue[i].done = -1;
	      }
      }
    }
  }
  
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
        && ivCurrentScoreLeft <= ivCurrentScoreRight) 
    {
      changePlayer( 9, ivEmergencyPlayerType);
      changePlayer(10, ivEmergencyPlayerType);
      changePlayer(11, ivEmergencyPlayerType);
      //std::cerr << "\n\n\nCHANGING!!!\n\n\n";
      ivDoEmergencyChange = false;
    }
  }
    
  return true;
}

//============================================================================
// onRefereeMessage()
//============================================================================
/**
 * 
 */
bool ModChange05::onRefereeMessage(bool PMChange) 
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
bool ModChange05::onKeyboardInput(const char *str) 
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
bool ModChange05::changePlayer(int unum,int type) 
{
  if ( ivSubstitutionQueueCounter >= 11 ) 
  {
    LOG_ERR(0,<<"Too many substitutions during this play_on cycle!");return false;
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
bool ModChange05::sendQueue() 
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
              sendMsg( MSG::MSG_CHANGE_PLAYER_TYPE,
                       ivSubstitutionQueue[i].unum,
                       ivSubstitutionQueue[i].type);
	            ivSubstitutionQueue[i].time = fld.getTime();
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
int ModChange05::compareRealPlayerSpeedMaximum( const PlayerType *t1,
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
int ModChange05::compareStaminaDemandFor10Meters( const PlayerType *t1,
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
ModChange05::fillPlayerTypeForRoleUsabilityMatrix()
{
printf("prepare for game\n");
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
ModChange05::PlayerTypeUsability::sort()
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
ModChange05::computePlayerTypeUsabilityForRole(PlayerType    * pt, 
                                               PLAYER_ROLES    role)
{
  float weightRealPlayerSpeedMax,    optimalityRealPlayerSpeedMax,
        weightStaminaDemandPerMeter, optimalityStaminaDemandPerMeter,
        weightSpeedProgression,      optimalitySpeedProgression,
        weightKickableMargin,        optimalityKickableMargin;
  
  switch (role)
  {
    case GOALIE:
      weightRealPlayerSpeedMax    = 0.0;
      weightStaminaDemandPerMeter = 0.0;
      weightSpeedProgression      = 0.0;
      weightKickableMargin        = 0.0;
    break;
    case SWEEPER:
      weightRealPlayerSpeedMax    = 0.3;
      weightStaminaDemandPerMeter = 0.4;
      weightSpeedProgression      = 0.2;
      weightKickableMargin        = 0.1;
    break;
    case DEFENDER:
      weightRealPlayerSpeedMax    = 0.15;
      weightStaminaDemandPerMeter = 0.2;
      weightSpeedProgression      = 0.6;
      weightKickableMargin        = 0.05;
    break;
    case MIDFIELDER:
      weightRealPlayerSpeedMax    = 0.2;
      weightStaminaDemandPerMeter = 0.5;
      weightSpeedProgression      = 0.2;
      weightKickableMargin        = 0.1;
    break;
    case OFFENDER:
      weightRealPlayerSpeedMax    = 0.35;
      weightStaminaDemandPerMeter = 0.25;
      weightSpeedProgression      = 0.25;
      weightKickableMargin        = 0.15;
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
  return //AMALGAMATION!
      weightRealPlayerSpeedMax    * optimalityRealPlayerSpeedMax
    + weightStaminaDemandPerMeter * optimalityStaminaDemandPerMeter
    + weightSpeedProgression      * optimalitySpeedProgression
    + weightKickableMargin        * optimalityKickableMargin;
}

//============================================================================
// computePlayerTypeRealPlayerSpeedMaxOptimality()
//============================================================================
/**
 * 
 */
float 
ModChange05::computePlayerTypeRealPlayerSpeedMaxOptimality(const PlayerType * pt)
{
  float min = 1.0, 
        max = 1.2,
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
ModChange05::computePlayerTypeStaminaDemandPerMeterOptimality(const PlayerType * pt)
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
ModChange05::computePlayerTypeSpeedProgressionOptimality(const PlayerType * pt)
{
  int   speedProgressSteps = 3;
  float min = (float)speedProgressSteps * 0.6, 
        max = (float)speedProgressSteps * 1.2,
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
ModChange05::computePlayerTypeKickableMarginOptimality(const PlayerType * pt)
{
  float min = 0.7 * 0.7, 
        max = 0.9 * 0.9,
        ptV = pt->kickable_margin * pt->kickable_margin;
  if (ptV > max) ptV = max;    if (ptV < min) ptV = min;        
  return (ptV-min) / (max-min);
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
 * Furthermore, the old version of this module (before ModChange05) used
 * to have a bug which hindered our coach from using heterogeneous player
 * types for our defenders at all.
 * 
 */
void ModChange05::prepareForGame() 
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
      // cout<<" chosPT="<<chosenPlayerType<<" chosPTUsab="<<chosenPlayerTypesUsability<<flush;
      //We allow usabilities of more than 0.5 only. If a player type has not
      //at least that usability we make use of the default player type.
      //                              ###########
      if (chosenPlayerTypesUsability < 0.25)//###
        chosenPlayerType = 0;//       ###########
      if (   chosenPlayerType == 0
          || numberOfPlayersOfPlayerType[ chosenPlayerType ] < 3)  
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
