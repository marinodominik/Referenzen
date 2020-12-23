/* Author: 
 *
 *
 */

#include "angle.h"
#include "options.h"
#include "logger.h"
#include <sstream>
#include <string.h>

#include "mod_direct_opponent_assignment_10.h"
#include "messages.h"

#define LOG_LINE(LVL,x1,y1,x2,y2,col) LOG_VIS(LVL, << _2D << L2D(x1,y1,x2,y2,col)) 

int ModDirectOpponentAssignment10::cvDirectOpponentAssignment[TEAM_SIZE];

bool 
ModDirectOpponentAssignment10::init(int argc,char **argv) 
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
  ivConflictOverrideAssignment = false;
  ivMinCoveringDistance = 3.f;
  
  ivConflictBoundaryThresh = 2.f;
  return true;
}

double
ModDirectOpponentAssignment10::getConflictResolveBoundary()
{
 return (RUN::side==RUN::side_LEFT) ? -FIELD_BORDER_X + PENALTY_AREA_LENGTH + 3.f 
                                    :  FIELD_BORDER_X - PENALTY_AREA_LENGTH - 3.f ; 
}

bool 
ModDirectOpponentAssignment10::isPlayerBehindConflictBoundary(Player *p)
{
    if ( p == NULL ) 
        return false;
    if ( RUN::side==RUN::side_LEFT ) 
    {
        return p->pos().getX() < getConflictResolveBoundary();
    }
    else
    {
        return p->pos().getX() > getConflictResolveBoundary();
    }
}


double
ModDirectOpponentAssignment10::distanceToMyBaseline(Player *p)
{
    if ( p == NULL )
        return 100; // HIGH Value
    if ( RUN::side==RUN::side_LEFT ) 
    {
        return p->pos().getX() + FIELD_BORDER_X;
    }
    else
    {
        return FIELD_BORDER_X - p->pos().getX();
    }
}

Player **
ModDirectOpponentAssignment10::get_sorted_by(int how_many, double * measured_data) {
  if (how_many < 1)
    return NULL;
  
  Player **arr = new Player*[how_many];
  for (int i=0; i< how_many; i++) {
    int min_idx= i;
    for (int j=i+1; j< how_many; j++) 
      if ( measured_data[j] < measured_data[min_idx] ) 
        min_idx= j;

    if (min_idx != i) { //toggle the elements at i and min_idx
      double int_tmp= measured_data[i];
      measured_data[i]= measured_data[min_idx];
      measured_data[min_idx]= int_tmp;
    }
    // copy over minimal element
    arr[i]= &(fld.hisTeam[min_idx]);
  }
  return arr;
}

bool 
ModDirectOpponentAssignment10::isAttacker(int num) {
  return num > 5;
}

void 
ModDirectOpponentAssignment10::calculateBestReplacementAssignment(int my_p_array_pos, int my_p_num, double *distances, double max_dist)
{
    //Player **opp_l_to_r = get_sorted_by(TEAM_SIZE, distances);
    int curr_best = -1;
    double best_dist = 10000000.f;
#if LOGGING && BASIC_LOGGING
    int orig_assigned;
#endif
    double tmp_dist;
    Player *tmp_player_p;
    switch ( my_p_num )
        {
            case 2: // left defender
            case 3: // middle defender
            case 5: // right defender
                for ( int i = 0; i < TEAM_SIZE; i++ )
                {
                    // first find our player that is assigned to this
                    // particular opponent
                    int opp = fld.hisTeam[i].number;
                    if ( ! fld.hisTeam[i].alive || fld.hisTeam[i].goalie ) 
                        continue; // ignore dead opponents and the goalie
                    int assigned = -1;
                    for ( int j = 0; j < TEAM_SIZE; j++ )
                        if ( ivDirectOpponentAssignment[j] == fld.hisTeam[i].number )
                            assigned = j;
                    if (    assigned == -1  // unassigned
                         && distances[i] <  max_dist // in dangerous position
                       )
                    { 
                      fld.getPlayerByNumber(4, my_TEAM, tmp_player_p);
                      if (    tmp_player_p->pos().distance(fld.hisTeam[i].pos()) > ivMinCoveringDistance //not covered by our sweeper
                           // && // not the player that is executing a free kick
                         )
                      {
                          // no one was assigned to this opponent and he is 
                          // more important to defend than current opponent
                          // --> simply reassign
                            LOG_FLD(1, << "DOAC: FOUND UNASSIGNED dangerous opp: " << opp 
                                       << " reassigning to our player " << my_p_num );
                            ivConflictOverrideAssignment = true;
                            ivDirectOpponentAssignment[my_p_array_pos] = opp;
                            ivLastChangeInComputedDirectOpponents = fld.getTime();
                            break;
                      }
                    }
                    else if (    isAttacker(fld.myTeam[assigned].number) 
                              && ivDirectOpponentAssignmentConflict[opp] == -1 
                              && distances[i] <  max_dist )
                    { // otherwise check if this opponent is defended by an attacker 
                      // if that is the case we found a possible conflict
                        tmp_dist = distances[i];//fld.hisTeam[i].pos().distance( fld.myTeam[my_p_array_pos].pos() ) ;
                        if (    curr_best == -1 
                             || tmp_dist < best_dist )
                        {
                            curr_best = opp;
                            best_dist = tmp_dist;
#if LOGGING && BASIC_LOGGING
                            orig_assigned = fld.myTeam[assigned].number;
#endif
                        }
                    }
                    else 
                    {
                        LOG_FLD(1, << "DOAC: opponent " << opp << " is no valid resolvent "
                                   << " isDefendedByAttacker ? " << isAttacker(fld.myTeam[assigned].number) 
                                   << " isCloserToGoalThanMyOpponent ? " << (distances[i] <  max_dist));
                    }
                }
                if ( curr_best != -1 )
                {
                    LOG_FLD(1, << "DOAC: FOUND CONFLICT RESOLVING ASSIGNMENT FOR our player: " 
                                   <<  my_p_num << " original defender " <<  orig_assigned
                                   << "  opp: " << curr_best);
                    ivDirectOpponentAssignmentConflict[my_p_array_pos] = curr_best;
                }
                else
                { // if we arrive here no resolving assignment was found
                  // the best thing to do at this point is stick to current assignment,
                  // update DOA as frequently as possible and wait for the conflict to resolve
                    LOG_FLD(1, << "DOAC: Could not resolve conflict for our player " << my_p_num);
                }
                break;
            default: // for now do not calculate replacements for any other player
                break;
        }
    //if ( opp_l_to_r ) delete [] opp_l_to_r;
}

void 
ModDirectOpponentAssignment10::checkConflicts()
{
    //std::cout << "CHECKING for conflicts in timestep " << fld.getTime() << std::endl;
    int opp;
    Player *opp_p;
    int countCloser;
    // clear out old conflicts
    for ( int i = 0; i < TEAM_SIZE+1; i++ )
    {
        ivDirectOpponentAssignmentConflict[i] = -1;
    }
    ivConflictOverrideAssignment = false;
    LOG_LINE(1, getConflictResolveBoundary(), -FIELD_BORDER_Y, getConflictResolveBoundary(), FIELD_BORDER_Y, "ff00f3");
    for ( int i = 0; i < TEAM_SIZE; i++ )
    {
        int my_p_num = fld.myTeam[i].number;
        if ( ! fld.myTeam[i].alive )
            continue; // ignore dead players
        switch ( my_p_num )
        {
            case 4: // sweaper << do not check for conflicts
                break;
            case 2: case 3: case 5: // defender
                opp = ivDirectOpponentAssignment[i];
                if ( opp == -1 )
                    continue; // unassigned 
                LOG_FLD(1, << "DOAC: checking conflicts for our defender: " 
                           << my_p_num << " defending " << opp);
                fld.getPlayerByNumber(opp, his_TEAM, opp_p); //&fld.hisTeam[opp];
                if (    ! isPlayerBehindConflictBoundary(opp_p) 
                     && ivDirectOpponentAssignmentConflict[opp] == -1 )
                { // if the opponent player is not behind the conflict
                  // boundary line we check for a conflict to be present
                  // --> first determine the opponents distance to 
                  //     own baseline
                  //std::cout << "checking conflict for our defender: " << my_p_num
                  //          << " with opp: " << opp << " == " <<  opp_p->number << std::endl;
                  double distances [TEAM_SIZE];
                  double dist_opp = distanceToMyBaseline(opp_p);
                  countCloser = 0;
                  for ( int j = 0; j < TEAM_SIZE; j++ )
                  { // check whether there are at least 3 opponent players
                    // closer to own baseline than my current opponent
                    // --> if so my_p_num is not defending an attacker!
                    Player *other_opp_p = &fld.hisTeam[j];
                    if ( other_opp_p->number != opp )
                    {
                        distances[j] = distanceToMyBaseline(other_opp_p);
                        if ( distances[j] < dist_opp - ivConflictBoundaryThresh )
                        {
                            LOG_VIS(1, << _2D << C2D(other_opp_p->pos().getX(), other_opp_p->pos().getY(), 0.7, "ff00f3"));
                            LOG_FLD(1, << "DOAC: checking for " << my_p_num << " " << other_opp_p->number << " is closer to baseline!"); 
                            countCloser++;
                        }
                    }
                  }
                  if ( countCloser >= 3 ) 
                  {
                    calculateBestReplacementAssignment(i, my_p_num, distances, dist_opp - ivConflictBoundaryThresh);
                  }
                }
                break;
            default: // for now do not check any other teammates
                break;
        }
    }
}

bool 
ModDirectOpponentAssignment10::destroy() 
{
  return true;
}

void
ModDirectOpponentAssignment10::updateOpponentPositions()
{
  float decay = 0.99;
  int myGoalieIndex = -1;
  for (int g=0; g<TEAM_SIZE; g++) if (fld.myTeam[g].goalie) myGoalieIndex = g;  
  int currentPM = fld.getPM();
  if (doWePlayAgainstWrightEagle() == true) decay = 0.95;
  if (    doWePlayAgainstWrightEagle() == true 
       && (       currentPM == PM_kick_in_r
               || currentPM == PM_free_kick_r
               || (    currentPM == PM_free_kick_l  
                    && myGoalieIndex > -1
                    && fabs( fld.myTeam[myGoalieIndex].pos_x - fld.ball.pos_x) < 0.5
                    && fabs( fld.myTeam[myGoalieIndex].pos_y - fld.ball.pos_y) < 0.5
                  )
               || currentPM == PM_corner_kick_r
               || currentPM == PM_kick_in_r
               || currentPM == PM_goal_kick_r
               || currentPM == PM_corner_kick_l
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
    if (fld.getTime() == 0)
    {
      ivOpponentPositionArray[i].num = 1;
      ivOpponentPositionArray[i].x = fld.hisTeam[i].pos_x;
      ivOpponentPositionArray[i].y = fld.hisTeam[i].pos_y;
    }
  }
}

Player *
ModDirectOpponentAssignment10::findNearestOpponentTo( double posx,
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
    delta = sqrt( 0.4*(posx-ox)*(posx-ox) + 0.6*(posy-oy)*(posy-oy) );
    LOG_FLD(1,<<"ModDirectOpponentAssignment: iteration "<<i<<", consider opp #"<<fld.hisTeam[i].number<<" at ("
      <<ox<<","<<oy<<") ==> delta="<<delta);

    //sorting regarding y difference
    if (assignableOpps[ fld.hisTeam[i].number ])
    {
      if (   delta < minDist*0.9999
          // JTS: WHAT THE HACK IS THAT FOR ??? >> || (delta >= minDist*0.9999 && returnValue && returnValue->number > fld.hisTeam[i].number)
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
ModDirectOpponentAssignment10::createDirectOpponentAssignment()
{
  ivLastDirectOpponentComputation = fld.getTime();
  bool assignableOpponents[TEAM_SIZE+1];
  for (int i=0; i<TEAM_SIZE+1; i++) {
      assignableOpponents[ i ] = true;
  }
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

char
ModDirectOpponentAssignment10::playerNumToChar(int n)
{
  switch ( n )
  {
      case 1: return '1';
      case 2: return '2';
      case 3: return '3';
      case 4: return '4';
      case 5: return '5';
      case 6: return '6';
      case 7: return '7';
      case 8: return '8';
      case 9: return '9';
      case 10:return 'A';
      case 11:return 'B';
      default: return '_';
  }
}

char
ModDirectOpponentAssignment10::ourPlayerToChar(int n)
{
	switch ( n )
  {
  	  case 0: return '0';
      case 1: return '1';
      case 2: return '2';
      case 3: return '3';
      case 4: return '4';
      case 5: return '5';
      case 6: return '6';
      case 7: return '7';
      case 8: return '8';
      case 9: return '9';
      case 10:return 'A';
      default: return '_';
  }
}

void 
ModDirectOpponentAssignment10::sendDirectOpponentAssignment()
{
  bool soccerServerBeforeV15 = false; // TOPIC FOR A BACHELOR THESIS -> Structured Communication among
                                      //                                Autonomous Agents in Multi-Agent Systems
  //creation of the potential string to be sent
  char strbuf[40 + 11 + 5 * 11]; //JTS10 added enough space for conflicts and stamina capacity
  char confBuf[5];
  char chosenString[2]; 
  chosenString[0] = '_'; chosenString[1] = '\0';
  confBuf[0] = '(';  confBuf[3] = ')';  confBuf[4] = '\0';
  if (soccerServerBeforeV15) // INFO AND ADVICE NO LONGER USABLE IN SOCCER SERVER VERSION 15
    sprintf(strbuf, "%ld (true) \"doa", fld.getTime() );
  else
    sprintf(strbuf, "\"doa" );
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
  // JTS10 also send information about player stamina in base encoding
  for (int i=0; i<TEAM_SIZE; i++)
  {
    chosenString[0] = WMTOOLS::get_base_encoding_from_stamina_capacity(fld.myTeam[i].staminaCapacityBound);
    LOG_FLD(1, "JTbS10 SENDING out player " << i << " stamina " << fld.myTeam[i].staminaCapacityBound << " as char " << WMTOOLS::get_stamina_capacity_from_base_encoding(WMTOOLS::get_base_encoding_from_stamina_capacity(fld.myTeam[i].staminaCapacityBound))<<" chosenString="<<chosenString<<" encoded=");
    strcat(strbuf, chosenString); 
  }
  // append conflict information
  for (int i=0; i<TEAM_SIZE; i++)
  {
    if ( ivDirectOpponentAssignmentConflict[i] != -1 )
    {
       confBuf[1] = ourPlayerToChar(i);
       confBuf[2] = playerNumToChar(ivDirectOpponentAssignmentConflict[i]);
       strcat(strbuf, confBuf);
    }
  }
  strcat(strbuf,"\"");

  //find out if i can send
  if ( fld.getPM() == PM_play_on )
  {
    if (soccerServerBeforeV15)
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
  }
  else 
  {
    // no play on
    if (soccerServerBeforeV15)
    {
      LOG_FLD(1,<<"ModDirectOpponentAssignment: I am sending as INFO2 ... ["
        <<fld.sayInfo.lastInPlayOn[MSGT_INFO]<<"/"
        <<fld.sayInfo.lastInPlayOn[MSGT_ADVICE]<<"] ... " <<strbuf);
      ivLastSentDirectOpponentAssignments = fld.getTime();
      sendMsg(MSG::MSG_SAY,MSGT_INFO,strbuf);
      return;
    }
    else
    {
      LOG_FLD(1,<<"ModDirectOpponentAssignment: I am sending as FREEFORM ... " <<strbuf);
      ivLastSentDirectOpponentAssignments = fld.getTime();
      sendMsg(MSG::MSG_SAY,MSGT_FREEFORM,strbuf);
      return;
    }
  }
  LOG_FLD(1,<<"ModDirectOpponentAssignment: I am sending NOT ["
    <<fld.sayInfo.lastInPlayOn[MSGT_INFO]<<"/"
    <<fld.sayInfo.lastInPlayOn[MSGT_ADVICE]<<"] ... " <<strbuf);
}

bool ModDirectOpponentAssignment10::behave() 
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
  bool soccerServerBeforeV15 = false; // TOPIC FOR A BACHELOR THESIS -> Structured Communication among
                                      //                                Autonomous Agents in Multi-Agent Systems
  int aliveCounter = 0;
  for (int i=0; i<TEAM_SIZE; i++)
    if (    fld.hisTeam[i].alive
         && ivOpponentPositionArray[i].num > 0
         && fabs( fld.hisTeam[i].pos_x ) < FIELD_BORDER_X
         && fabs( fld.hisTeam[i].pos_y ) < FIELD_BORDER_Y )
      aliveCounter ++;
  if ( (    currentPM != PM_play_on  
         && (   fld.getTime() - ivLastDirectOpponentComputation > 5 ) 
         && (   fld.getTime() > 0
             || (soccerServerBeforeV15 == false && aliveCounter == TEAM_SIZE) )
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
    LOG_FLD(1,<<"ModDirectOpponentAssignment: I am computing the assignment now!");

//TG08
//    if (aliveCounter == TEAM_SIZE)
      this->createDirectOpponentAssignment();
  }
  
  // JTS check for assignment conflicts
  if (!( currentPM == PM_kick_off_l || currentPM == PM_kick_off_r || currentPM == PM_before_kick_off))
    checkConflicts();
  
  // (EVTL.) SENDING THE ASSIGNMENT
  if ( 1 )
  {
    int pauseBetweenSendings = 20;
    //if (fld.getPM() != PM_play_on) pauseBetweenSendings = 100;
    //if (   fld.getPM() != PM_play_on )
    //  pauseBetweenSendings = 5;
    if (   ivLastDirectOpponentComputation >= 0
        && ivLastChangeInComputedDirectOpponents > ivLastSentDirectOpponentAssignments
        && ( fld.getTime() - ivLastSentDirectOpponentAssignments > pauseBetweenSendings)
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
       || (soccerServerBeforeV15 == false && fld.getTime() <= 0 )
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
  
  return true;
}

bool ModDirectOpponentAssignment10::onChangePlayerType
       (bool ownTeam,int unum,int type) 
{
  return true;
}

bool ModDirectOpponentAssignment10::onRefereeMessage(bool PMChange) {

  if ( fld.getPM() == 17) //RM_goal_l) 
    ivCurrentScoreLeft++;
  if ( fld.getPM() == 18) //RM_goal_r) 
    ivCurrentScoreRight++;
  return true;
}

bool ModDirectOpponentAssignment10::onKeyboardInput(const char *str) {

  return false;
}

bool ModDirectOpponentAssignment10::onHearMessage(const char *str) {

  return false;
}

const char ModDirectOpponentAssignment10::modName[]="ModDirectOpponentAssignment10";

bool ModDirectOpponentAssignment10::doWePlayAgainstWrightEagle()
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
