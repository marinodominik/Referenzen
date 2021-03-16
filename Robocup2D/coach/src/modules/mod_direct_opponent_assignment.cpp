/* Author: 
 *
 *
 */

#include "angle.h"
#include "options.h"
#include "logger.h"
#include <sstream>

#include "mod_direct_opponent_assignment.h"
#include "messages.h"

int ModDirectOpponentAssignment::cvDirectOpponentAssignment[TEAM_SIZE];

bool ModDirectOpponentAssignment::init(int argc,char **argv) 
{
  ivLastDirectOpponentComputation = -1000;
  ivLastChangeInComputedDirectOpponents = -1000;
  ivLastSentDirectOpponentAssignments = -1000;
  for (int i=0; i<TEAM_SIZE; i++)
  {
    ivDirectOpponentAssignment[i] = -1;
    ivOpponentPositionArray[i].num = 0;
    ivOpponentPositionArray[i].x = 0.0;
    ivOpponentPositionArray[i].y = 0.0;
  }
  ivCurrentScoreLeft         = 0;
  ivCurrentScoreRight        = 0;
  
  return true;
}

bool ModDirectOpponentAssignment::destroy() 
{
  return true;
}

Player *ModDirectOpponentAssignment::getPlByIndex(int i,Field *field) 
{
  if(i<TEAM_SIZE) 
  {
    if (field->myTeam[i].alive) 
      return &field->myTeam[i];
    return NULL;
  } 
  else 
  {
    if (field->hisTeam[i-TEAM_SIZE].alive) return &field->hisTeam[i-TEAM_SIZE];
    return NULL;
  }
}

void
ModDirectOpponentAssignment::updateOpponentPositions()
{
  float decay = 0.99;
  int myGoalieIndex = -1;
  for (int g=0; g<TEAM_SIZE; g++) if (fld.myTeam[g].goalie) myGoalieIndex = g;  
  int currentPM = fld.getPM();
  if (doWePlayAgainstWrightEagle() == true) decay = 0.95;
  int pmHisKickIn     = (RUN::side==RUN::side_LEFT)?PM_kick_in_r:PM_kick_in_l,
      pmHisFreeKick   = (RUN::side==RUN::side_LEFT)?PM_free_kick_r:PM_free_kick_l,
      pmMyFreeKick    = (RUN::side==RUN::side_LEFT)?PM_free_kick_l:PM_free_kick_r,
      pmHisCornerKick = (RUN::side==RUN::side_LEFT)?PM_corner_kick_r:PM_corner_kick_l,
      pmHisGoalKick   = (RUN::side==RUN::side_LEFT)?PM_goal_kick_r:PM_goal_kick_l,
      pmMyCornerKick  = (RUN::side==RUN::side_LEFT)?PM_corner_kick_l:PM_corner_kick_r;
  if (    doWePlayAgainstWrightEagle() == true 
       && (       currentPM == pmHisKickIn
               || currentPM == pmHisFreeKick
               || (    currentPM == pmMyFreeKick  
                    && myGoalieIndex > -1
                    && fabs( fld.myTeam[myGoalieIndex].pos_x - fld.ball.pos_x) < 0.5
                    && fabs( fld.myTeam[myGoalieIndex].pos_y - fld.ball.pos_y) < 0.5
                  )
               || currentPM == pmHisCornerKick
               || currentPM == pmHisKickIn
               || currentPM == pmHisGoalKick
               || currentPM == pmMyCornerKick
          )
      )
    decay = 0.5;
  for (int i=0; i<TEAM_SIZE; i++)
  {
    if (   (    fabs( fld.ball.pos_x ) < FIELD_BORDER_X*0.8
            && fabs( fld.ball.pos_y ) < FIELD_BORDER_Y*0.8  )
        || doWePlayAgainstWrightEagle() == true
       )
    {
      float avgX = ivOpponentPositionArray[i].getAverageXPosition(),
            avgY = ivOpponentPositionArray[i].getAverageYPosition();
      
      float newAvgX = decay * avgX + (1.0-decay) * fld.hisTeam[i].pos_x,
            newAvgY = decay * avgY + (1.0-decay) * fld.hisTeam[i].pos_y;
            
      if (ivOpponentPositionArray[i].num == 0)
      {
        newAvgX = fld.hisTeam[i].pos_x;
        newAvgY = fld.hisTeam[i].pos_y;
      }
     
      LOG_VIS(0,<<_2D<<C2D(newAvgX,newAvgY,0.3,"ff2222"));
      LOG_VIS(0,<<_2D<<STRING2D(newAvgX,newAvgY,""<<i+1,"ff2222"));
      
      ivOpponentPositionArray[i].num ++ ;
      
      ivOpponentPositionArray[i].x = newAvgX * ivOpponentPositionArray[i].num;
      ivOpponentPositionArray[i].y = newAvgY * ivOpponentPositionArray[i].num;
    }
    /*if ( fld.getTime() == 2999 ) //partial reset at half time
    {  //This reset seems to be not a good idea, provokes instability!
      ivOpponentPositionArray[i].num /= 30;
      ivOpponentPositionArray[i].x /= 30.0;
      ivOpponentPositionArray[i].y /= 30.0;
    }*/
  }
}

Player *
ModDirectOpponentAssignment::findNearestOpponentTo( double posx,
                                                    double posy,
                                                    bool * assignableOpps)
{
  int currentPM = fld.getPM();
  Player * returnValue = NULL;
  float minDist = 1000.0;
  double delta;
  for (int i=0; i<TEAM_SIZE; i++)
  {
    //determine the opponent position
    double ox = fld.hisTeam[i].pos_x,
          oy = fld.hisTeam[i].pos_y;
    if (  (    currentPM == PM_kick_off_l || currentPM == PM_kick_off_r
            || currentPM == PM_before_kick_off)
        &&
          sqrt(ox*ox + oy*oy) < 10.0 ) //player near anstosskreis
    {  //zoom position to centre
      LOG_FLD(1,<<"ModDirectOpponentAssignment: Player "<<fld.hisTeam[i].number<<" is near Anstosskreis, zooming to Anstosspunkt.");
      double offset = (RUN::side==RUN::side_LEFT)?-1.0:1.0;
      ox = offset + (0.3 * ox); 
    } 
    if (ivOpponentPositionArray[i].num > 20)
    {
      double offset = (RUN::side==RUN::side_LEFT)?FIELD_BORDER_X:-FIELD_BORDER_X;
      ox = (ivOpponentPositionArray[i].getAverageXPosition()
            + offset) / 2.0;
      oy = ivOpponentPositionArray[i].getAverageYPosition();
      LOG_FLD(1,<<"ModDirectOpponentAssignment: Opp "<<fld.hisTeam[i].number<<" with numOfPosUpdates="<<ivOpponentPositionArray[i].num<<" avgX = "<<ox<<" avgY = "<<oy);
    }
    
        //////////////////////////////////////////////////////////////////////
        if (1 && fld.getTime() < 2)
        {
          LOG_FLD(0,<<"ModDOA: OUTER LOOP ### "<<i<<" ###");
          //determine his foremost
          double foremostX = 100.0;
#if LOGGING && BASIC_LOGGING
          int foremostXNr = 0;
#endif
          for (int a=0; a<TEAM_SIZE; a++)
          {
            if (    currentPM == PM_kick_off_r
                 || currentPM == PM_before_kick_off)
              if (    fabs(fld.hisTeam[a].pos_x) < 3.0
                   && fabs(fld.hisTeam[a].pos_y) < 3.0 )
                continue;
            if ( fld.hisTeam[a].pos_x < foremostX )
            {
              foremostX = fld.hisTeam[a].pos_x;
#if LOGGING && BASIC_LOGGING
              foremostXNr = fld.hisTeam[a].number;
#endif
              LOG_FLD(0,<<"ModDOA: foremostX="<<foremostX<<" foremostXNr="<<foremostXNr);
            }
          } 
          //determine his players on a line
          int playersNearlyOnOneLine = 0;
          for (int a=0; a<TEAM_SIZE; a++)
          {
            if ( fld.hisTeam[a].pos_x < foremostX + 4.0 )
              playersNearlyOnOneLine += 1;
            LOG_FLD(0,<<"ModDOA: playersNearlyOnOneLine="<<playersNearlyOnOneLine);  
          }
          //special handling needed
          if (playersNearlyOnOneLine > 5) 
          {
            for (int a=0; a<TEAM_SIZE; a++)
            {
              if (    a == i 
                   && fld.hisTeam[a].pos_x < foremostX + 4.0 )
              {
                int consideredPlayerNumber = fld.hisTeam[a].number;
                int hisGoalieNumber = 0;
                for (int g=0; g<TEAM_SIZE; g++) if (fld.hisTeam[g].goalie) 
                                                hisGoalieNumber=fld.hisTeam[g].number;
                LOG_FLD(0,<<"ModDOA: hisGoalieNumber="<<hisGoalieNumber);
                if ( hisGoalieNumber > 0 && hisGoalieNumber < 12)
                {
                  if ( hisGoalieNumber > consideredPlayerNumber )
                    consideredPlayerNumber += 1;
                  else
                  if ( hisGoalieNumber < consideredPlayerNumber )
                    consideredPlayerNumber += 0; //no change
                  else // ==
                    consideredPlayerNumber = 1; //goalie
                }
                LOG_FLD(0,<<"ModDOA: consideredPlayerNumber="<<consideredPlayerNumber);   
                LOG_FLD(0,<<"ModDOA: BEFORE ox: "<< ox);
                switch (consideredPlayerNumber)
                {
                  case 1:  ox += 14.0; break;
                  case 2:  ox += 14.0; break;
                  case 3:  ox += 14.0; break;
                  case 4:  ox += 14.0; break;
                  case 5:  ox += 14.0; break;
                  case 6:  ox +=  7.0; break;
                  case 7:  ox +=  7.0; break;
                  case 8:  ox +=  7.0; break;
                  case 9:  ox +=  0.0; break;
                  case 10: ox +=  0.0; break;
                  case 11: ox +=  0.0; break;
                  default: break;
                }
                LOG_FLD(0,<<"ModDOA: AFTER ox: "<< ox);
                LOG_VIS(0,<<_2D<<C2D(ox,oy,0.3,"22ff22"));
                LOG_VIS(0,<<_2D<<STRING2D(ox,oy,""<<fld.hisTeam[i].number,"22ff22"));
                
              }
            }
          }
        }
        //////////////////////////////////////////////////////////////////////
    
    //compute distance
    delta = sqrt( (posx-ox)*(posx-ox) + (posy-oy)*(posy-oy) );
    LOG_FLD(1,<<"ModDirectOpponentAssignment: iteration "<<i<<", consider opp #"<<fld.hisTeam[i].number<<" at ("
      <<ox<<","<<oy<<") ==> delta="<<delta);

    //sorting regarding y difference
    if (assignableOpps[ fld.hisTeam[i].number ])
    {
      if (   delta < minDist*0.9999
          || (delta >= minDist*0.9999 && returnValue && returnValue->number > fld.hisTeam[i].number)
         )
      {
        minDist = delta;
        returnValue = & fld.hisTeam[i];
        LOG_FLD(1,<<"ModDirectOpponentAssignment: Found a new nearest opponent for ["<<posx<<","<<posy<<"]. It is opp#"<<fld.hisTeam[i].number<<" with delta="<<delta<<" at pos ("<<fld.hisTeam[i].pos_x<<","<<fld.hisTeam[i].pos_y<<")");
      }
    }
  }
  return returnValue;
}

void 
ModDirectOpponentAssignment::createDirectOpponentAssignment()
{
  ivLastDirectOpponentComputation = fld.getTime();
  bool assignableOpponents[TEAM_SIZE+1];
  for (int i=0; i<TEAM_SIZE+1; i++) 
      assignableOpponents[ i ] = true;
  for (int i=0; i<TEAM_SIZE; i++) 
    if (fld.hisTeam[i].goalie)
      assignableOpponents[ fld.hisTeam[i].number ] = false; //not the goalie

  //TG08: BEGIN
  for (int i=0; i<TEAM_SIZE; i++) 
    if ( fld.hisTeam[i].alive == false )
      assignableOpponents[ fld.hisTeam[i].number ] = false;
  //TG08: END


  //ZUI-FFM: begin
  //berechnungsreihenfolge
  const int br[TEAM_SIZE] = { 1, 2, 5, 4, 3, 6, 8, 7, 9, 11, 10 };
  
  for (int i=0; i<TEAM_SIZE; i++)
  {
    LOG_FLD(1,<<"ModDirectOpponentAssignment: oppPosArray["<<i+1<<"] = "
      <<ivOpponentPositionArray[i].getAverageXPosition()<<", "
      <<ivOpponentPositionArray[i].getAverageYPosition());

    //determine the "i" to use
    int usedI = -1;
    for (int j=0; j<TEAM_SIZE; j++)
      if ( br[i] == fld.myTeam[j].number )
        usedI = j;
    LOG_FLD(1,<<"ModDirectOpponentAssignment: Determined the i to use."
      <<" ACTUAL i: "<<i<<" USED i: "<<usedI<<" br[i]="<<br[i]
      <<" NUMBER(fld.myTeam[usedI].number)="<<fld.myTeam[usedI].number);
      
    //Tormann ignorieren
    if (fld.myTeam[ usedI ].goalie) continue;
    //Unsere 4 ist "Ausputzer".
    if (fld.myTeam[ usedI ].number == 4) 
    {
      ivDirectOpponentAssignment[ usedI ] = -1;
      continue;
    }
    //TG08: abgestuerzte Agenten ignorieren
    if (fld.myTeam[ usedI ].alive == false)
    {
      ivDirectOpponentAssignment[ usedI ] = -1;
      continue;
    }
    Player * nearestOpponent 
      = this->findNearestOpponentTo( ivMyInitialPositions[usedI].x,//fld.myTeam[i].pos_x,
                                     ivMyInitialPositions[usedI].y,//fld.myTeam[i].pos_y,
                                     assignableOpponents );
    if (nearestOpponent)
    {
      LOG_FLD(1,<<"ModDirectOpponentAssignment: t="<<fld.getTime()<<" ASSIGN: my"
        << usedI+1 <<" ("<<ivMyInitialPositions[usedI].x<<","
                   <<ivMyInitialPositions[usedI].y<<") "
        <<" TO his"<<nearestOpponent->number<<" ("
        <<ivOpponentPositionArray[usedI].x<<","
        <<ivOpponentPositionArray[usedI].y<<")");
      LOG_VIS(0,<<_2D<<STRING2D(ivMyInitialPositions[usedI].x,
                                ivMyInitialPositions[usedI].y, ""<<(usedI+1), "22ff22"));
      LOG_VIS(0,<<_2D<<L2D(ivMyInitialPositions[usedI].x,
                           ivMyInitialPositions[usedI].y,
                           ivOpponentPositionArray[nearestOpponent->number-1]
                             .getAverageXPosition(),
                           ivOpponentPositionArray[nearestOpponent->number-1]
                             .getAverageYPosition(), "22ff22"));
      //cout<<"ASSIGN: "<<i+1<<" at "<<ivMyInitialPositions[i].x<<","<<ivMyInitialPositions[i].y
  //<<" to "<<nearestOpponent->number<<" at "<<ivOpponentPositionArray[i].x<<","<<ivOpponentPositionArray[i].y<<endl;
      if ( ivDirectOpponentAssignment[usedI] != nearestOpponent->number )
        ivLastChangeInComputedDirectOpponents = fld.getTime();
      ivDirectOpponentAssignment[usedI] = nearestOpponent->number;
      assignableOpponents[nearestOpponent->number] = false;
    }
  }

  //ZUI-FLUG: begin
  for (int i=0; i<TEAM_SIZE; i++)
  {
    LOG_VIS(0,<<_2D<<C2D(ivOpponentPositionArray[i].getAverageXPosition(),
                         ivOpponentPositionArray[i].getAverageYPosition(),
                         0.6,"ff2222"));
    LOG_VIS(0,<<_2D<<STRING2D(ivOpponentPositionArray[i].getAverageXPosition(),
                              ivOpponentPositionArray[i].getAverageYPosition(),
                              ""<<i+1,"ff2222"));
  }
  //ZUI-FLUG: end

  //ZUI-FFM: end
  for (int i=0; i<TEAM_SIZE; i++) 
    cvDirectOpponentAssignment[i] = ivDirectOpponentAssignment[i];
}

void 
ModDirectOpponentAssignment::sendDirectOpponentAssignment()
{
  //creation of the potential string to be sent
  char strbuf[40]; //TG09: von 20 auf 40 erhoeht
  char chosenString[2]; 
  chosenString[0] = '_'; chosenString[1] = '\0';
  sprintf(strbuf, "%ld (true) \"doa", fld.getTime() );
  for (int i=0; i<TEAM_SIZE; i++)
    switch (ivDirectOpponentAssignment[i])
    {
      case 1: strcat(strbuf, "1"); break;
      case 2: strcat(strbuf, "2"); break;
      case 3: strcat(strbuf, "3"); break;
      case 4: strcat(strbuf, "4"); break;
      case 5: strcat(strbuf, "5"); break;
      case 6: strcat(strbuf, "6"); break;
      case 7: strcat(strbuf, "7"); break;
      case 8: strcat(strbuf, "8"); break;
      case 9: strcat(strbuf, "9"); break;
      case 10:strcat(strbuf, "A");break;
      case 11:strcat(strbuf, "B");break;
      default:
      {
        if ( strlen(fld.getHisTeamName()) == 0 )
          strcat(strbuf, "x"); 
        else
        {
          if (   strstr( fld.getHisTeamName(), "umbol" ) != NULL
              || strstr( fld.getHisTeamName(), "UMBOL" ) != NULL 
              || strstr( fld.getHisTeamName(), "AT-H" ) != NULL )
            chosenString[0] = TEAM_IDENTIFIER_ATHUMBOLDT;
          if (   strstr( fld.getHisTeamName(), "elios" ) != NULL
              || strstr( fld.getHisTeamName(), "ELIOS" ) != NULL )
            chosenString[0] = TEAM_IDENTIFIER_HELIOS;
          if (   strstr( fld.getHisTeamName(), "WE200" ) != NULL
              || strstr( fld.getHisTeamName(), "WE0" ) != NULL 
              || strstr( fld.getHisTeamName(), "Wright" ) != NULL 
              || strstr( fld.getHisTeamName(), "wright" ) != NULL 
              || strstr( fld.getHisTeamName(), "xsy" ) != NULL 
             )
            chosenString[0] = TEAM_IDENTIFIER_WRIGHTEAGLE;
          strcat(strbuf, chosenString); 
        }
        break;
      }
    }
  strcat(strbuf,"\"");

  //find out if i can send
  if ( fld.getPM() == PM_play_on ) 
  {
    if ( fld.getTime()==1 || fld.getTime()>=50 ) 
    {
      if (    fld.sayInfo.lastInPlayOn[MSGT_INFO] >= 0 
           &&    fld.getTime() - fld.sayInfo.lastInPlayOn[MSGT_INFO]
              <= ServerParam::clang_win_size) 
      {
        if (    fld.sayInfo.lastInPlayOn[MSGT_ADVICE] < 0 
             ||   fld.getTime() - fld.sayInfo.lastInPlayOn[MSGT_ADVICE]
                > ServerParam::clang_win_size) 
        {
//          if ( fld.getTime() - fld.sayInfo.lastInPlayOn[MSGT_INFO] >= 100) 
          {
            LOG_FLD(1,<<"ModDirectOpponentAssignment: I am sending as ADVICE ... ["
              <<fld.sayInfo.lastInPlayOn[MSGT_INFO]<<"/"
              <<fld.sayInfo.lastInPlayOn[MSGT_ADVICE]<<"] ... " <<strbuf);
            ivLastSentDirectOpponentAssignments = fld.getTime();
            sendMsg(MSG::MSG_SAY,MSGT_ADVICE,strbuf);
            return;
          }
        }
      } 
      else
      {
        LOG_FLD(1,<<"ModDirectOpponentAssignment: I am sending as INFO1 ... ["
          <<fld.sayInfo.lastInPlayOn[MSGT_INFO]<<"/"
          <<fld.sayInfo.lastInPlayOn[MSGT_ADVICE]<<"] ... " <<strbuf);
        ivLastSentDirectOpponentAssignments = fld.getTime();
        sendMsg(MSG::MSG_SAY,MSGT_INFO,strbuf);
        return;
      }
    }
  }
  else 
  {
    LOG_FLD(1,<<"ModDirectOpponentAssignment: I am sending as INFO2 ... ["
      <<fld.sayInfo.lastInPlayOn[MSGT_INFO]<<"/"
      <<fld.sayInfo.lastInPlayOn[MSGT_ADVICE]<<"] ... " <<strbuf);
    ivLastSentDirectOpponentAssignments = fld.getTime();
    sendMsg(MSG::MSG_SAY,MSGT_INFO,strbuf);
    return;
  }
  LOG_FLD(1,<<"ModDirectOpponentAssignment: I am sending NOT ["
    <<fld.sayInfo.lastInPlayOn[MSGT_INFO]<<"/"
    <<fld.sayInfo.lastInPlayOn[MSGT_ADVICE]<<"] ... " <<strbuf);
}


bool ModDirectOpponentAssignment::behave() 
{
  int currentPM = fld.getPM();
  //LOG_FLD(1,<<"ModDirectOpponentAssignment: currentPM=="<<currentPM<<" ivOldPM=="<<ivOldPM);

  if ( currentPM == PM_before_kick_off || fld.getTime() < 2)
  {
    double offset = (RUN::side==RUN::side_LEFT)?-25.0:25.0;
    for (int i=0; i<TEAM_SIZE; i++)
    {
      ivMyInitialPositions[i].x = fld.myTeam[i].pos_x * 2.0 + offset;
      ivMyInitialPositions[i].y = fld.myTeam[i].pos_y;
    }
    //special consideration of my centre attacker (nr.10) and my defenders
    //[this is - admittedly - an ugly hack]
    ivMyInitialPositions[9].x = 0.0 + offset;
    ivMyInitialPositions[9].y = 0.0;
    ivMyInitialPositions[1].y 
      = (RUN::side==RUN::side_LEFT)?28.0:-28.0; //nr.2 //ZUI-FFM
    ivMyInitialPositions[2].y = 0.0;  //nr.3 //ZUI-FFM
    ivMyInitialPositions[4].y 
      = (RUN::side==RUN::side_LEFT)?-28.0:28.0; //nr.5 //ZUI-FFM
    ivMyInitialPositions[5].y 
      = (RUN::side==RUN::side_LEFT)?26.0:-26.0; //nr.6 //ZUI-FFM
    ivMyInitialPositions[6].y = 0.0;  //nr.7 //ZUI-FFM
    ivMyInitialPositions[7].y 
      = (RUN::side==RUN::side_LEFT)?-26.0:26.0; //nr.8 //ZUI-FFM
  }
  
  
  // CALCULATIONS TO DETERMINE ASSIGNMENT
  if ( (    currentPM != PM_play_on  
         && (   fld.getTime() - ivLastDirectOpponentComputation > 5 ) 
         && (   fld.getTime() > 0 )
       )
       ||
       (    currentPM == PM_play_on  
         && (   fld.getTime() - ivLastDirectOpponentComputation > 50 + (fld.getTime() / 10) ) 
         && (   fld.getTime() > 0 )
       )
     )
       //currentPM == PM_kick_off_l
       //|| currentPM == PM_kick_off_r
       //|| currentPM == PM_before_kick_off)
  {
    int aliveCounter = 0;
    for (int i=0; i<TEAM_SIZE; i++) 
      if ( fld.hisTeam[i].alive ) aliveCounter ++;
    LOG_FLD(1,<<"ModDirectOpponentAssignment: I am computing the assignment now!");

//TG08
//    if (aliveCounter == TEAM_SIZE)
      this->createDirectOpponentAssignment();
  }
  
  
  // (EVTL.) SENDING THE ASSIGNMENT
  if ( 1 /*   (ivOldPM != currentPM && currentPM != PM_play_on )
       || currentPM == PM_before_kick_off 
       || currentPM == PM_kick_off_l
       || currentPM == PM_kick_off_r 
       || (currentPM != PM_play_on && fld.getTime() % 610 == 50) */
     )
  {
    int pauseBetweenSendings = 20;
    if (fld.getPM() != PM_play_on) pauseBetweenSendings = 100;
    if (   fld.getPM() != PM_play_on
        && doWePlayAgainstWrightEagle() == true)
      pauseBetweenSendings = 5;
    if (   ivLastDirectOpponentComputation >= 0
        && ivLastChangeInComputedDirectOpponents > ivLastSentDirectOpponentAssignments
        && ( fld.getTime() - ivLastSentDirectOpponentAssignments > pauseBetweenSendings )
       )
    {
      LOG_FLD(1,<<"ModDirectOpponentAssignment: I (am trying to) send the assignment now (lastComp="
        <<ivLastDirectOpponentComputation<<",lastChg="<<ivLastChangeInComputedDirectOpponents
        <<",lastSent="<<ivLastSentDirectOpponentAssignments<<")!");
      this->sendDirectOpponentAssignment();
    }
  }

  int myGoalieIndex = -1;
  for (int g=0; g<TEAM_SIZE; g++) if (fld.myTeam[g].goalie) myGoalieIndex = g;

  int pmHisKickIn     = (RUN::side==RUN::side_LEFT)?PM_kick_in_r:PM_kick_in_l,
      pmMyKickIn      = (RUN::side==RUN::side_LEFT)?PM_kick_in_l:PM_kick_in_r,
      pmHisFreeKick   = (RUN::side==RUN::side_LEFT)?PM_free_kick_r:PM_free_kick_l,
      pmMyFreeKick    = (RUN::side==RUN::side_LEFT)?PM_free_kick_l:PM_free_kick_r,
      pmHisCornerKick = (RUN::side==RUN::side_LEFT)?PM_corner_kick_r:PM_corner_kick_l,
      pmHisGoalKick   = (RUN::side==RUN::side_LEFT)?PM_goal_kick_r:PM_goal_kick_l,
      pmMyCornerKick  = (RUN::side==RUN::side_LEFT)?PM_corner_kick_l:PM_corner_kick_r;

  if (    currentPM == PM_play_on 
       || (   doWePlayAgainstWrightEagle() == true //TG09: ZUI
           && (   currentPM == PM_play_on
               || currentPM == pmHisKickIn
               || currentPM == pmHisFreeKick
               || (    currentPM == pmMyFreeKick  
                    && myGoalieIndex > -1
                    && fabs( fld.myTeam[myGoalieIndex].pos_x - fld.ball.pos_x) < 0.5
                    && fabs( fld.myTeam[myGoalieIndex].pos_y - fld.ball.pos_y) < 0.5
                  )
               || (    (currentPM == pmMyFreeKick || currentPM == pmMyKickIn)   
                    && fld.ball.pos_x > FIELD_BORDER_X - 22.0
                  )
               || currentPM == pmHisCornerKick
               || currentPM == pmHisKickIn
               || currentPM == pmHisGoalKick
               || currentPM == pmMyCornerKick
               || (currentPM==pmMyKickIn && fld.ball.pos_x > 0.0)
              ) 
          )
     )
  {
    updateOpponentPositions();
    LOG_FLD(0,<<"YES, DO THE UPDATE");
  }
  else
  {
    LOG_FLD(0,<<"NO UPDATE");
  }
  
  ivOldFld = fld;
  ivOldPM  = fld.getPM();


/*
  static char strbuf[200];
  static std::strstream stream(strbuf,200);
  
  if(oldpm==PM_play_on && fld.getPM()==PM_play_on) {
    
    for(int i=START_PL;i<2*TEAM_SIZE;i++)
      for(int j=0;j<pltypes;j++)
	possTypes[i][j]=0;
    
    checkCollisions();
    checkKick();
    checkPlayerDecay();
    //checkInertiaMoment();  Don't check this, this module causes trouble
    checkPlayerSpeedMax();
    
    int cnt;bool flg=false;
    for(int p=0;p<TEAM_SIZE;p++) {
      cnt=0;
      for(int t=0;t<pltypes;t++) {
	if(fld.myTeam[p].possibleTypes[t]>=0) {
	  fld.myTeam[p].possibleTypes[t]+=possTypes[p][t];
	  if(fld.myTeam[p].type!=t) {
	    if(fld.myTeam[p].possibleTypes[t]>HINTS_TO_BE_SURE) {
	      LOG_FLD(1,<<"ModDirectOpponentAssignment: **** I am sure: Own player #"<<p+1<<" is not of type "
		      <<t<<"! ("<<fld.myTeam[p].possibleTypes[t]<<" Points)");
	      fld.myTeam[p].possibleTypes[t]=-1;cnt++;
	    }
	  } else {
	    if(fld.myTeam[p].possibleTypes[t]>HINTS_TO_RECONSIDER) {
	      LOG_FLD(0,<<"ModDirectOpponentAssignment: RECONSIDER own player #"<<p+1<<", not of type "<<t<<"!");
	      std::cerr << "\nModDirectOpponentAssignment: RECONSIDER own player #"<<p+1<<", not of type "<<t<<"!";
	      fld.myTeam[p].possibleTypes[t]=-1;cnt++;
	      //RUN::announcePlayerChange(true,p+1,-1);
	    }
	  }
	}
	else {cnt++;}
      }
      if(cnt==pltypes) {
	LOG_ERR(0,<<"ERROR: ModDirectOpponentAssignment: Something went really wrong, there is no possible type for "
		<<" own player #"<<p+1);
	successMentioned[p]=false;
	onlyOneLeft[p]=false;
	for(int t=0;t<pltypes;t++) {
	  possTypes[p][t]=0;
	  fld.myTeam[p].possibleTypes[t]=0;
	  flg=true;
	}
      }
      if(cnt==pltypes-1) {
	if(!successMentioned[p]) {
	  int t;
	  for(t=0;t<pltypes;t++) if(fld.myTeam[p].possibleTypes[t]>=0) break;
	  LOG_DEF(0,<<"ModDirectOpponentAssignment: ****** I am sure: My player #"<<p+1<<" must be of type "<<t<<"!");
	  successMentioned[p]=true;flg=true;onlyOneLeft[p]=true;
	}
      }

      cnt=0;
      for(int t=0;t<pltypes;t++) {
	if(fld.hisTeam[p].possibleTypes[t]>=0) {
	  fld.hisTeam[p].possibleTypes[t]+=possTypes[p+TEAM_SIZE][t];
	  if(fld.hisTeam[p].type!=t) {
	    if(fld.hisTeam[p].possibleTypes[t]>HINTS_TO_BE_SURE) {
	      LOG_FLD(1,<<"ModDirectOpponentAssignment: **** I am sure: Opponent player #"<<p+1<<" is not of type "
		      <<t<<"! ("<<fld.hisTeam[p].possibleTypes[t]<<" Points)");
	      fld.hisTeam[p].possibleTypes[t]=-1;cnt++;
	    }
	  } else {
	    if(fld.hisTeam[p].possibleTypes[t]>HINTS_TO_RECONSIDER) {
	      LOG_FLD(0,<<"ModDirectOpponentAssignment: RECONSIDER opp player #"<<p+1<<", not of type "<<t<<"!");
	      std::cerr << "\nModDirectOpponentAssignment: RECONSIDER opp player #"<<p+1<<", not of type "<<t<<"!";
	      fld.hisTeam[p].possibleTypes[t]=-1;cnt++;
	      RUN::announcePlayerChange(false,p+1,-1);
	    }
	  }
	}
	else {cnt++;}
      }
      if(cnt==pltypes) {
	LOG_ERR(0,<<"ERROR: ModDirectOpponentAssignment: Something went really wrong, there is no possible type for "
		<<" opp player #"<<p+1);
	flg=true;
	RUN::announcePlayerChange(false,p+1,-1);
      }
      if(cnt==pltypes-1) {
	if(!successMentioned[p+TEAM_SIZE]) {
	  int t;
	  for(t=0;t<pltypes;t++) if(fld.hisTeam[p].possibleTypes[t]>=0) break;
	  LOG_DEF(0,<<"ModDirectOpponentAssignment: ****** I am sure: Opponent player #"<<p+1
		  <<" must be of type "<<t<<"!");
	  successMentioned[p+TEAM_SIZE]=true;flg=true;
	  onlyOneLeft[p+TEAM_SIZE]=true;
	  if(t!=fld.hisTeam[p].type) {
	    RUN::announcePlayerChange(false,p+1,t);
	  }
	}
      }

    }
    if(flg) {
      LOG_DEF(0,<<"ModDirectOpponentAssignment: Revised table of known player types:");
      LOG_DEF(0,<<"ModDirectOpponentAssignment:");
      LOG_DEF(0,<<"ModDirectOpponentAssignment:  Pl | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | A | B |");
      LOG_DEF(0,<<"ModDirectOpponentAssignment: -------------------------------------------------");
      if(START_PL<TEAM_SIZE) {
	stream.seekp(0);
	stream << "ModDirectOpponentAssignment: Own |";
	for(int p=0;p<TEAM_SIZE;p++) {
	  if(!fld.myTeam[p].alive) {stream << " - |";continue;}
	  if(p<START_PL) {
	    stream << " x |";continue;
	  }
	  if(onlyOneLeft[p]) {
	    int j;
	    for(j=0;j<pltypes;j++) if(fld.myTeam[p].possibleTypes[j]>=0) break;
	    stream << " "<<j<<" |";continue;
	  }
	  stream << " ? |";
	}
	LOG_DEF(0,<<strbuf);
      }
      stream.seekp(0);
      stream << "ModDirectOpponentAssignment: Opp |";
      for(int p=0;p<TEAM_SIZE;p++) {
	if(!fld.hisTeam[p].alive) {stream << " - |";continue;}
	if(p<START_PL-TEAM_SIZE) {stream << " x |";continue;}
	if(fld.hisTeam[p].type<0 && fld.hisTeam[p].possibleTypes[0]>=0) {
	  stream << " * |";continue;}
	if(fld.hisTeam[p].type<0) {stream << " ? |";continue;}
	stream << " "<<fld.hisTeam[p].type<<" |";
      }
      stream << ends;
      LOG_DEF(0,<<strbuf);
      
    }
  }
  // Announce changes to players, if necessary (and possible) 

  if(fld.getTime()>0 && (!changeAnnounced || !goalieAnnounced)) {
    using namespace MSG;
    //LOG_MSG(0,<<"Telling the players...");
    stream.seekp(0);
    stream << fld.getTime() << " (true) ";
    stream << "\"pt";
    int goalie=-1;
    for(int i=0;i<TEAM_SIZE;i++) {
      if(fld.hisTeam[i].type<0) stream << "_";
      else stream << fld.hisTeam[i].type;
      if(fld.hisTeam[i].goalie) goalie=fld.hisTeam[i].number;
    }
    if(goalie>0) stream << " g"<<goalie;
    else stream << " g_";
    stream << "\"";
    stream <<ends;
    if(fld.getPM()==PM_play_on) 
    {
      if(fld.getTime()==1 || fld.getTime()>=50) 
      {
        if (    fld.sayInfo.lastInPlayOn[MSGT_INFO]>=0 
             &&    fld.getTime() - fld.sayInfo.lastInPlayOn[MSGT_INFO]
                <= ServerParam::clang_win_size) 
        {
          if (    fld.sayInfo.lastInPlayOn[MSGT_ADVICE] < 0 
               ||   fld.getTime() - fld.sayInfo.lastInPlayOn[MSGT_ADVICE]
                  > ServerParam::clang_win_size) 
          {
            if ( fld.getTime() - fld.sayInfo.lastInPlayOn[MSGT_INFO] >= 100) 
            {
              sprintf(strbuf,"%d (true) \"doa123456789AB\"",fld.getTime());
              sendMsg(MSG_SAY,MSGT_ADVICE,strbuf);
              changeAnnounced=true;
              if(goalie>0) goalieAnnounced=true;
            }
          }
        } 
        else
        {
          sprintf(strbuf,"%d (true) \"doa123456789AB\"",fld.getTime());
          sendMsg(MSG_SAY,MSGT_INFO,strbuf);
          if(goalie>0) goalieAnnounced=true;
          changeAnnounced=true;
        }
      }
    }
    else 
    {
      sprintf(strbuf,"%d (true) \"doa123456789AB\"",fld.getTime());
      sendMsg(MSG_SAY,MSGT_INFO,strbuf);
      if(goalie>0) goalieAnnounced=true;
      changeAnnounced=true;
    }
    if ( !goalieAnnounced && fld.getTime()>=10) 
    {
      LOG_FLD(1,<<"Opponent does not seem to have a goalie!");
      goalieAnnounced=true;
    }
  }
  
  oldfld=fld;oldpm=fld.getPM();
*/
  return true;
}

bool ModDirectOpponentAssignment::onChangePlayerType
       (bool ownTeam,int unum,int type) 
{
  return true;
}

bool ModDirectOpponentAssignment::onRefereeMessage(bool PMChange) {

  if ( fld.getPM() == 17) //RM_goal_l) 
    ivCurrentScoreLeft++;
  if ( fld.getPM() == 18) //RM_goal_r) 
    ivCurrentScoreRight++;
  return true;
}

bool ModDirectOpponentAssignment::onKeyboardInput(const char *str) {

  return false;
}

bool ModDirectOpponentAssignment::onHearMessage(const char *str) {

  return false;
}

const char ModDirectOpponentAssignment::modName[]="ModDirectOpponentAssignment";

bool ModDirectOpponentAssignment::doWePlayAgainstWrightEagle()
{
  
/*  int myScore  = (RUN::side==RUN::side_LEFT) ? ivCurrentScoreLeft
                                             : ivCurrentScoreRight;
  int hisScore = (RUN::side==RUN::side_LEFT) ? ivCurrentScoreRight
                                             : ivCurrentScoreLeft;
*/
  if (   strstr( fld.getHisTeamName(), "WE200" ) != NULL
      || strstr( fld.getHisTeamName(), "WE0" ) != NULL 
      || strstr( fld.getHisTeamName(), "Wright" ) != NULL 
      || strstr( fld.getHisTeamName(), "wright" ) != NULL 
      || strstr( fld.getHisTeamName(), "xsy" ) != NULL 
     )
  {
    return true;
  }
  return false;
}
