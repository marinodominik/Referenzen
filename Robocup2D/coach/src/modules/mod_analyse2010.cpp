/* Author: 
 *         <Original Version> Manuel "Sputnick" Nickschas, 02/2002
 *         <Extensions>       Thomas Gabel, 04/2008
 *         <Extensions (*)>   Haiko Schol, 12/2008 
 *
 * This module analyses the opponent team and tries to find out the opponent
 * player types.
 * 
 * (*)
 * This module analyses the opponent team and tries to find out the opponent
 * player types. Player decay is also used for the analysis. This method is
 * inspired by the approach taken by AT Humboldt.
 * 
 */

#include "angle.h"
#include "options.h"
#include "logger.h"
#include <sstream>

#include "mod_analyse2010.h"
#include "messages.h"

/** The number of hints needed to be sure enough to exclude a player type */
#define HINTS_TO_BE_SURE 2
#define HINTS_TO_RECONSIDER 30

/** These weights balance the importance of different hints. */
#define KICK_RANGE_WEIGHT 4
#define PLAYER_DECAY_WEIGHT 10
#define INERTIA_MOMENT_WEIGHT 10
#define PLAYER_SPEED_MAX_WEIGHT 20
#define PLAYER_MAX_SPEED_PROGRESS_WEIGHT 10
#define SINGLE_PLAYER_USAGE_WEIGHT 20
#define PLAYER_FUZZY_WEIGHT 3
#define PLAYER_DECAY09_WEIGHT 20

#define START_PL 0  //DON'T MODIFY! Feature not working!
#define STILLSTANDING_PLAYER_THRESHOLD 0.02

const double ModAnalyse2010::MOVEMENT_EPSILON(0.1);


bool ModAnalyse2010::init(int argc,char **argv) {
  pltypes=PlayerParam::player_types;

  oldfld=fld;
  ivRecentTimeBehaveEntered = -1;
  changeAnnounced=false;
  goalieAnnounced=false;
  
  maxKickRange=0;
  
  for(int p=0;p<pltypes;p++) {
    double kr=fld.plType[p].kickable_margin+fld.plType[p].player_size;
    kickRanges[p]=kr+ServerParam::ball_size;
    if(kr>maxKickRange) maxKickRange=kr;
  }
  maxKickRange+=ServerParam::ball_size;

  for(int i=START_PL;i<2*TEAM_SIZE;i++) {
    successMentioned[i]=false;
    onlyOneLeft[i]=false;
  }
  
  Player *player;
  for(int p=START_PL;p<2*TEAM_SIZE;p++) 
  {
    if ((player=getPlByIndex(p,&fld))==NULL) 
      continue;
    player->type = -1; //important: for soccer server in version 12 (and
                       //above, by default each player type can be used
                       //only once - also the default player. hence, it is
                       //NOT a good idea to initially guess that any player is
                       //a default player (i.e. of type 0). therefore, we
                       //initialize all players with an unknown type (-1).
  }  

  for(int i=START_PL;i<2*TEAM_SIZE;i++)
    for(int j=0;j<pltypes;j++)
      possTypes[i][j] = 0;
  
  return true;
}

bool ModAnalyse2010::destroy() {
  
  return true;
}

Player *ModAnalyse2010::getPlByIndex(int i,Field *field) {
  if(i<TEAM_SIZE) {
    if(field->myTeam[i].alive) return &field->myTeam[i];
    return NULL;
  } else {
    if(field->hisTeam[i-TEAM_SIZE].alive) return &field->hisTeam[i-TEAM_SIZE];
    return NULL;
  }
}

bool ModAnalyse2010::behave() 
{
  if (ivRecentTimeBehaveEntered == fld.getTime())
  {
    LOG_FLD(0,"ModAnalyse2010: RETURN IMMEDIATELY. Multiple enterings of behave at t="<<ivRecentTimeBehaveEntered);
    return false;
  }
  ivRecentTimeBehaveEntered = fld.getTime();  

  static std::ostringstream oss;
  
LOG_FLD(0,"ModAnalyse2010: play mode at t="<<fld.getTime()<<" is "
<<fld.getPM()<<" (oldPM="<<oldpm<<") /// oldTime="<<oldfld.getTime()<<" curTime="<<fld.getTime()<<" timeDiff="<<(fld.getTime()-oldfld.getTime()));
  
  
  
  if (fld.getPM()==PM_play_on)
    updateSpeedProgressInformationArray();
  
  if(oldpm==PM_play_on && fld.getPM()==PM_play_on) 
  {

    
    checkCollisions();
    checkKick();
    checkPlayerDecay();
    //checkInertiaMoment();  Don't check this, this module causes trouble
    checkPlayerSpeedMax();
    checkMaxSpeedProgress();
    checkInertiaMoment08();
    checkSpeedProgress();
//    checkForAlreadyAssignedTypes();
    checkPlayerDecay09();

    showPossTypeTable();

    for(int i=START_PL;i<2*TEAM_SIZE;i++)
      for(int j=0;j<pltypes;j++)
        possTypes[i][j] = (int)(0.99*(float)(possTypes[i][j]));
    
    int cnt;bool flg=false;
    int myBestTypeMatchForPlayer[TEAM_SIZE], hisBestTypeMatchForPlayer[TEAM_SIZE];
    int myUsedTypes[pltypes], hisUsedTypes[pltypes];
    for (int t=0; t<pltypes; t++) 
    {  myUsedTypes[t] = 0; hisUsedTypes[t] = 0; }
    for (int p=0; p<TEAM_SIZE; p++) 
    {
      myBestTypeMatchForPlayer[p]  = -1;
      hisBestTypeMatchForPlayer[p] = -1;
      //unsere Spieler
      long minExclusionPoints = -1;
      for (int t=0; t<pltypes; t++) 
      {
        long currentExclusionPoints = possTypes[p][t];
        if (    minExclusionPoints < 0
             || currentExclusionPoints < minExclusionPoints )
        {
          minExclusionPoints = currentExclusionPoints;
          myBestTypeMatchForPlayer[p] = t; 
        }
      }
      //seine Spieler
      minExclusionPoints = -1;
      for (int t=0; t<pltypes; t++) 
      {
        long currentExclusionPoints = possTypes[p+TEAM_SIZE][t];
        if (    minExclusionPoints < 0
             || currentExclusionPoints < minExclusionPoints )
        {
          minExclusionPoints = currentExclusionPoints;
          hisBestTypeMatchForPlayer[p] = t; 
        }
      }
    }
    for (int p=0; p<TEAM_SIZE; p++) 
    {
      if (fld.myTeam[p].alive)
        myUsedTypes [  myBestTypeMatchForPlayer[p]   ] ++ ;
      if (fld.hisTeam[p].alive)
        hisUsedTypes[  hisBestTypeMatchForPlayer[p]  ] ++ ;
    }
    bool myTypeAnalysisIsConsistent = true,
         hisTypeAnalysisIsConsistent = true;
    for (int t=0; t<pltypes; t++) 
    {
      if (myUsedTypes[t]  > 1) myTypeAnalysisIsConsistent = false;
      if (hisUsedTypes[t] > 1) hisTypeAnalysisIsConsistent = false;
    }    
    showPossTypeTable();
    LOG_FLD(0,<<"ModAnalyse2010: RESULT: myTypeAnalysisIsConsistent="<<myTypeAnalysisIsConsistent
      <<" hisTypeAnalysisIsConsistent="<<hisTypeAnalysisIsConsistent);
      
    bool broadCastIsAdvised =    myTypeAnalysisIsConsistent
                              && hisTypeAnalysisIsConsistent;
      
    if ( broadCastIsAdvised )
    {
      for (int p=0; p<TEAM_SIZE; p++)
      {
        //I don't need to broadcast this info for my own team, because
        //I get this information immediately from the ModChange* module.
        //via announcePlayerChange().
        //if (fld.myTeam[p].type != myBestTypeMatchForPlayer[p] )
        //  RUN::announcePlayerChange(true,p+1,myBestTypeMatchForPlayer[p]);
        if (fld.hisTeam[p].type != hisBestTypeMatchForPlayer[p] )
          RUN::announcePlayerChange(false,p+1,hisBestTypeMatchForPlayer[p]);
      } 
    }


    if (0)
    { int p;
      //#######################################
      cnt=0;
      for (int t=0; t<pltypes; t++) 
      {
	      if (fld.myTeam[p].possibleTypes[t] >= 0) 
        {
          fld.myTeam[p].possibleTypes[t] += possTypes[p][t];
	        if (fld.myTeam[p].type != t) 
          { 
	          if (fld.myTeam[p].possibleTypes[t] > HINTS_TO_BE_SURE) 
            {
              LOG_FLD(1,<<"ModAnalyse2010: **** I am sure: Own player #"<<p+1<<" is not of type "
                <<t<<"! ("<<fld.myTeam[p].possibleTypes[t]<<" Points)");
              //cout<<"\nModAnalyse2010: **** I am sure: Own player #"<<p+1<<" is not of type "
              //  <<t<<"! ("<<fld.myTeam[p].possibleTypes[t]<<" Points)";
	            if (fld.myTeam[p].possibleTypes[t] == PLAYER_FUZZY_WEIGHT)
                fld.myTeam[p].possibleTypes[t]=-2;
              else
                fld.myTeam[p].possibleTypes[t]=-1;
              cnt++;
	          }
          } 
          else
          {
            if (fld.myTeam[p].possibleTypes[t] > HINTS_TO_RECONSIDER) 
            {
              LOG_FLD(0,<<"ModAnalyse2010: RECONSIDER own player #"<<p+1<<", not of type "<<t<<" ("
                <<(fld.myTeam[p].possibleTypes[t])<<")!");
              //std::cerr << "\nModAnalyse2010: RECONSIDER own player #"<<p+1<<", not of type "<<t<<"!";
              //std::cout << "\nModAnalyse2010: RECONSIDER own player #"<<p+1<<", not of type "<<t<<"!";
              
              if (fld.myTeam[p].possibleTypes[t] == PLAYER_FUZZY_WEIGHT)
                fld.myTeam[p].possibleTypes[t]=-2;
              else
                fld.myTeam[p].possibleTypes[t]=-1;
              cnt++;
              //RUN::announcePlayerChange(true,p+1,-1);

              //TG2010: type t can now be considered for other players again
              if (getNumberOfRemainingTypesForPlayer(p) == 0)
                for (int t2=0; t2<pltypes; t2++)
                   fld.myTeam[p].possibleTypes[t2] = 0;
              for (int p2=0; p2<TEAM_SIZE; p2++)
                 fld.myTeam[p2].possibleTypes[t] = 0;

            }
          }
        }
        else 
        if (fld.myTeam[p].possibleTypes[t] == -1)
        {
          cnt++;
        }
        else 
        if (fld.myTeam[p].possibleTypes[t] == -2)
        {
          if (possTypes[p][t] == 0)
          {
            LOG_FLD(0,<<"ModAnalyse2010: CANCEL SPEED PROGRESS ASSUMPTION for own player #"<<p+1<<", eventually really of type "<<t<<"!");
            //std::cerr<<"\nModAnalyse2010: CANCEL SPEED PROGRESS ASSUMPTION for own player #"<<p+1<<", eventually really of type "<<t<<"!";
            fld.myTeam[p].possibleTypes[t] = 0;
          }
          else 
          if (possTypes[p][t] > PLAYER_FUZZY_WEIGHT)
          {
            LOG_FLD(0,<<"ModAnalyse2010: OVERRIDE FUZZY ASSUMPTION for own player #"<<p+1<<", definitely not of type "<<t<<"!");
            //std::cerr<<"\nModAnalyse2010: OVERRIDE FUZZY ASSUMPTION for own player #"<<p+1<<", definitely not of type "<<t<<"!";
            fld.myTeam[p].possibleTypes[t] = -1;
            cnt ++;
          }
          else
            cnt++;
        }
      }
    
    
      if ( cnt==pltypes )
      {
        for (int t=0; t<pltypes; t++)
        {
          if ( fld.myTeam[p].possibleTypes[t] == -2 )
          {
            LOG_FLD(0,<<"ModAnalyse2010: CANCEL FUZZY ASSUMPTION for own player #"<<p+1<<", eventually really of type "<<t<<"!");
            //std::cerr<<"\nModAnalyse2010: CANCEL FUZZY ASSUMPTION for own player #"<<p+1<<", eventually really of type "<<t<<"!";
            fld.myTeam[p].possibleTypes[t] = 0;
          }
          cnt -- ;
        }
      }
      
      if ( cnt==pltypes ) 
      {
        LOG_ERR(0,<<"ERROR: ModAnalyse2010: Something went really wrong, there is no possible type for "
                  <<" own player #"<<p+1);
        LOG_FLD(0,<<"ERROR: ModAnalyse2010: Something went really wrong, there is no possible type for "
                  <<" own player #"<<p+1);
        successMentioned[p]=false;
        onlyOneLeft[p]=false;
        for(int t=0;t<pltypes;t++) 
        {
          possTypes[p][t]=0;
          fld.myTeam[p].possibleTypes[t] = 0;
          flg=true;
        }
      }
      if ( cnt==pltypes-1 ) 
      {
        if(!successMentioned[p]) 
        {
          int t;
          for (t=0; t<pltypes; t++) 
            if ( fld.myTeam[p].possibleTypes[t] >= 0 ) 
              break;
          LOG_DEF(0,<<"ModAnalyse2010: ****** I am sure: My player #"<<p+1
                    <<" must be of type "<<t<<"!");
	        successMentioned[p] = true;
          flg = true;
          onlyOneLeft[p]=true;
        }
      }

      //~~~~~~~BEGINN FUER GEGENSPIELER~~~~~~~~~~~~~~~
      cnt=0;
      for(int t=0;t<pltypes;t++) 
      {
        if (fld.hisTeam[p].possibleTypes[t]>=0) 
        {
          fld.hisTeam[p].possibleTypes[t] += possTypes[p+TEAM_SIZE][t];
          if (fld.hisTeam[p].type!=t) 
          {
            if (fld.hisTeam[p].possibleTypes[t]>HINTS_TO_BE_SURE) 
            {
              LOG_FLD(1,<<"ModAnalyse2010: **** I am sure: Opponent player #"<<p+1<<" is not of type "
                        <<t<<"! ("<<fld.hisTeam[p].possibleTypes[t]<<" Points)");
              if (fld.hisTeam[p].possibleTypes[t] == PLAYER_FUZZY_WEIGHT)
                fld.hisTeam[p].possibleTypes[t]=-2;
              else
                fld.hisTeam[p].possibleTypes[t]=-1;
              cnt++;
            }
          } 
          else
          {
            if (fld.hisTeam[p].possibleTypes[t]>HINTS_TO_RECONSIDER) 
            {
              LOG_FLD(0,<<"ModAnalyse2010: RECONSIDER opp player #"<<p+1<<", not of type "<<t<<" ("
                <<(fld.hisTeam[p].possibleTypes[t])<<")!");
              //std::cerr << "\nModAnalyse2010: RECONSIDER opp player #"<<p+1<<", not of type "<<t<<"!";
              //std::cout << "\nModAnalyse2010: RECONSIDER opp player #"<<p+1<<", not of type "<<t<<"!";

              if (fld.hisTeam[p].possibleTypes[t] == PLAYER_FUZZY_WEIGHT)
                fld.hisTeam[p].possibleTypes[t]=-2;
              else
                fld.hisTeam[p].possibleTypes[t]=-1;
              cnt++;

              RUN::announcePlayerChange(false,p+1,-1);
            }
          }
        }
        else if (fld.hisTeam[p].possibleTypes[t] == -1)
        {
          cnt++;
        }
        else if (fld.hisTeam[p].possibleTypes[t] == -2)
        {
          if (possTypes[p][t] == 0)
          {
              LOG_FLD(0,<<"ModAnalyse2010: CANCEL SPEED PROGRESS ASSUMPTION for opp player #"<<p+1<<", eventually really of type "<<t<<"!");
              //std::cerr<<"\nModAnalyse2010: CANCEL SPEED PROGRESS ASSUMPTION for opp player #"<<p+1<<", eventually really of type "<<t<<"!";
              fld.hisTeam[p].possibleTypes[t] = 0;
          }
          else 
          if (possTypes[p][t] > PLAYER_FUZZY_WEIGHT)
          {
            LOG_FLD(0,<<"ModAnalyse2010: OVERRIDE FUZZY ASSUMPTION for opp player #"<<p+1<<", definitely not of type "<<t<<"!");
            //std::cerr<<"\nModAnalyse2010: OVERRIDE FUZZY ASSUMPTION for opp player #"<<p+1<<", definitely not of type "<<t<<"!";
            fld.hisTeam[p].possibleTypes[t] = -1;
            cnt ++;
          }
          else
            cnt++;
        }
      }
      if ( cnt==pltypes )
      {
        for (int t=0; t<pltypes; t++)
        {
          if ( fld.hisTeam[p].possibleTypes[t] == -2 )
          {
            LOG_FLD(0,<<"ModAnalyse2010: CANCEL FUZZY ASSUMPTION for opp player #"<<p+1<<", eventually really of type "<<t<<"!");
            //std::cerr<<"\nModAnalyse2010: CANCEL FUZZY ASSUMPTION for opp player #"<<p+1<<", eventually really of type "<<t<<"!";
            fld.hisTeam[p].possibleTypes[t] = 0;
          }
          cnt -- ;
        }
      }
      if (cnt==pltypes) 
      {
        LOG_ERR(0,<<"ERROR: ModAnalyse2010: Something went really wrong, there is no possible type for "
                  <<" opp player #"<<p+1);
        flg=true;
        RUN::announcePlayerChange(false,p+1,-1);
      }
      if(cnt==pltypes-1) 
      {
        if (!successMentioned[p+TEAM_SIZE]) 
        {
          int t;
          for (t=0;t<pltypes;t++) 
            if(fld.hisTeam[p].possibleTypes[t]>=0) 
              break;
          LOG_DEF(0,<<"ModAnalyse2010: ****** I am sure: Opponent player #"
                    <<p+1<<" must be of type "<<t<<"!");
          successMentioned[p+TEAM_SIZE]=true;
          flg=true;
          onlyOneLeft[p+TEAM_SIZE]=true;
          if (t!=fld.hisTeam[p].type) 
          {
            RUN::announcePlayerChange(false,p+1,t);
          }
        }
      }
    }//ENDE DER SCHLEIFE UEBER ALLE SPIELER
    
    flg=true; //DEBUGGING OUTPUT
    showPossTypeTable();

    if (flg) 
    {
      LOG_DEF(0,<<"ModAnalyse2010: Revised table of known player types:");
      LOG_DEF(0,<<"ModAnalyse2010:");
      LOG_DEF(0,<<"ModAnalyse2010:  Pl | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | A | B |");
      LOG_DEF(0,<<"ModAnalyse2010: -------------------------------------------------");
      if(START_PL<TEAM_SIZE) 
      {
        oss.seekp(0);
        oss << "ModAnalyse2010: Own |";
        for (int p=0;p<TEAM_SIZE;p++) 
        {
          if (!fld.myTeam[p].alive) 
          {
            oss << " - |";
            continue;
          }
          if (p<START_PL) 
          {
            oss << " x |";
            continue;
          }
          if (myBestTypeMatchForPlayer[p] > -1) 
          {
            oss << " "<<myBestTypeMatchForPlayer[p]<<" |";
            continue;
          }
          oss << " ? |";
        }
        LOG_DEF(0,<<oss.str());
        for (int p=0;p<TEAM_SIZE;p++)
          LOG_DEF(0,<<"ModAnalyse2010: remaining types for own player "<<p+1<<": "
            <<getNumberOfRemainingTypesForPlayer(p));
      }
      oss.seekp(0);
      oss << "ModAnalyse2010: Opp |";
      for(int p=0;p<TEAM_SIZE;p++) 
      {
        if (!fld.hisTeam[p].alive) 
        {
          oss << " - |";
          continue;
        }
        if (p<START_PL-TEAM_SIZE)
        {
          oss << " x |";
          continue;
        }
        if (hisBestTypeMatchForPlayer[p] > -1) 
        {
          oss << " "<<hisBestTypeMatchForPlayer[p]<<" |";
          continue;
        }
        if (   fld.hisTeam[p].type < 0 
            && fld.hisTeam[p].possibleTypes[0]>=0) 
        {
          oss << " * |";
          continue;
        }
        if (fld.hisTeam[p].type<0) 
        {
          oss << " ? |";
          continue;
        }
        oss << " "<<fld.hisTeam[p].type<<" |";
      }
      oss << ends;
      LOG_DEF(0,<<oss.str());
      for (int p=0;p<TEAM_SIZE;p++)
        LOG_DEF(0,<<"ModAnalyse2010: remaining types for opp player "<<p+1<<": "
          <<getNumberOfRemainingTypesForPlayer(p+TEAM_SIZE));
    }
  }
  /* Announce changes to players, if necessary (and possible) */
  if (    fld.getTime() > 0 
       && (!changeAnnounced || !goalieAnnounced) ) 
  {
    using namespace MSG;
    bool soccerServerBeforeV15 = false; // TOPIC FOR A BACHELOR THESIS -> Structured Communication among
                                        //                                Autonomous Agents in Multi-Agent Systems
    //LOG_MSG(0,<<"Telling the players...");
    oss.seekp(0);
    if (soccerServerBeforeV15) // INFO AND ADVICE NO LONGER USABLE IN SOCCER SERVER VERSION 15
    {
      oss << fld.getTime() << " (true) ";
    }
    oss << "\"pt";
    int goalie=-1;
    for(int i=0;i<TEAM_SIZE;i++) 
    {
      if (fld.hisTeam[i].type<0) oss << "_";
      else if (fld.hisTeam[i].type>9)
      {
        switch (fld.hisTeam[i].type)
        {
          case 10: oss << "A"; break;
          case 11: oss << "B"; break;
          case 12: oss << "C"; break;
          case 13: oss << "D"; break;
          case 14: oss << "E"; break;
          case 15: oss << "F"; break;
          case 16: oss << "G"; break;
          case 17: oss << "H"; break;
          case 18: oss << "I"; break;
          case 19: oss << "J"; break;
          case 20: oss << "K"; break;
          case 21: oss << "L"; break;
          case 22: oss << "M"; break;
          case 23: oss << "N"; break;
          case 24: oss << "O"; break;
          default: break;
        }
      }
      else oss << fld.hisTeam[i].type;
      if(fld.hisTeam[i].goalie) goalie=fld.hisTeam[i].number;
    }
    if(goalie>0) oss << " g"<<goalie;
    else oss << " g_";
    oss << "\"";
    oss <<ends;
    if(fld.getPM()==PM_play_on) 
    {
      if (soccerServerBeforeV15)
      {

        if(fld.getTime()==1 || fld.getTime()>=50)
        {
          if (   fld.sayInfo.lastInPlayOn[MSGT_INFO]>=0
              && fld.getTime()-fld.sayInfo.lastInPlayOn[MSGT_INFO]<=ServerParam::clang_win_size)
          {
            if (    fld.sayInfo.lastInPlayOn[MSGT_ADVICE]<0
                 || fld.getTime()-fld.sayInfo.lastInPlayOn[MSGT_ADVICE]>ServerParam::clang_win_size)
            {
              if ( fld.getTime()-fld.sayInfo.lastInPlayOn[MSGT_INFO]>=100)
              {
                LOG_DEF(0,<<"ModAnalyse2010: Send opp types via MSGT_ADVICE at t="
                  <<fld.getTime()<<": "<<oss.str());
                sendMsg(MSG_SAY,MSGT_ADVICE,oss.str());
	              changeAnnounced=true;
	              if(goalie>0) goalieAnnounced=true;
            }
	        }
          }
          else
          {
            LOG_DEF(0,<<"ModAnalyse2010: Send opp types via MSGT_INFO[1] at t="
              <<fld.getTime()<<": "<<oss.str());
            sendMsg(MSG_SAY,MSGT_INFO,oss.str());
            if(goalie>0) goalieAnnounced=true;
	          changeAnnounced=true;
	      }
        }
      }
    } 
    else 
    {
      // no play on
      if (soccerServerBeforeV15)
      {
        LOG_DEF(0,<<"ModAnalyse2010: Send opp types via MSGT_INFO[2] at t="
          <<fld.getTime()<<": "<<oss.str());
        sendMsg(MSG_SAY,MSGT_INFO,oss.str());
      }
      else
      {
        LOG_FLD(1,<<"ModDirectOpponentAssignment: I am sending as FREEFORM ... " <<oss.str());
        sendMsg(MSG::MSG_SAY,MSGT_FREEFORM,oss.str());
      }
      if(goalie>0) goalieAnnounced=true;
        changeAnnounced=true;
    }
    if(!goalieAnnounced && fld.getTime()>=10) 
    {
      LOG_FLD(1,<<"Opponent does not seem to have a goalie!");
      goalieAnnounced=true;
    }
  }
  
  oldfld=fld;oldpm=fld.getPM();
  return true;
}

/** check which players may have collided */
void ModAnalyse2010::checkCollisions() 
{
  Player *player,*tmpplayer;
  double size,tmpsize;
  int type,tmptype;
  for(int p=START_PL;p<2*TEAM_SIZE;p++) 
  {
    mayCollide[p]=false;
    if((player=getPlByIndex(p,&fld))==NULL) 
      continue;
    if (player->type==-1) type=0;
    else type=player->type;  // worst case if unknown
    size=fld.plType[type].player_size;
    bool flg=false;
    for(int p2=0;p2<TEAM_SIZE*2;p2++) 
    {   // Check collision
      if(p2==p) continue;
      if((tmpplayer=getPlByIndex(p2,&fld))==NULL) continue;
      if(tmpplayer->type==-1) tmptype=0;else tmptype=tmpplayer->type;
      tmpsize=fld.plType[tmptype].player_size;
      if((tmpplayer->pos()-player->pos()).norm()<size+tmpsize+0.2) 
      {
        flg=true;
        break;
      }
    }
    if(flg) 
    {
      mayCollide[p]=true;
      if(p<TEAM_SIZE) 
      {
        LOG_FLD(1,<<"Own player #"<<p+1<<" collided with other player!");
      } 
      else 
      {
        LOG_FLD(1,<<"Opp player #"<<p+1-TEAM_SIZE<<" collided with other player!");
      }
      continue;
    }
    if ((player->pos()-fld.ball.pos()).norm()<size+ServerParam::ball_size+0.1) 
    {
      if(p<TEAM_SIZE) 
      {
        LOG_FLD(1,<<"Own player #"<<p+1<<" collided with ball!");
      } 
      else 
      {
        LOG_FLD(1,<<"Opp player #"<<p-TEAM_SIZE+1<<" collided with ball!");	
      }
      mayCollide[p]=true;
    }
  }
}      

/** Ball gekickt? Kickrange gross genug? */
bool ModAnalyse2010::checkKick() 
{
  
  ballKicked=false;  
  kickingPlayer=-1;

  Player *player;
  
  Vector u = oldfld.ball.vel();
  Vector newpos=oldfld.ball.pos()+u;
  Vector newvel=ServerParam::ball_decay*u;
  double rmax=ServerParam::ball_rand*oldfld.ball.vel().norm();
  
  if (    (fabs(fld.ball.pos().getX()-newpos.getX())>rmax)
       || (fabs(fld.ball.pos().getY()-newpos.getY())>rmax) )
    ballKicked=true;
  if (    (fabs(fld.ball.vel().getX()-newvel.getX())>ServerParam::ball_decay*rmax)
       || (fabs(fld.ball.vel().getX()-newvel.getX())>ServerParam::ball_decay*rmax) )
    ballKicked=true;

  int cnt=0;int kickpl=-1;
  if(ballKicked) 
  {
    for(int i=0;i<TEAM_SIZE*2;i++) 
      canKick[i]=false;
    //LOG_FLD(0,<<"ModAnalyse2010: Ball has been kicked!");
    for (int i=0;i<TEAM_SIZE*2;i++) 
    {
      if ((player=getPlByIndex(i,&oldfld))!=NULL) 
      {
        if ((oldfld.ball.pos()-player->pos()).norm()<=maxKickRange) 
        {
          canKick[i]=true;
          cnt++;
          kickpl=i;
        }
      }
    }
    if (!cnt) 
    {
      LOG_FLD(1,<<"ModAnalyse2010: Ball has been kicked, but I don't know by whom... Collision?");
    } 
    else 
    {
      if (cnt==1) 
      {
        if (mayCollide[kickpl]) 
        {   // ignore in case of possible collision
          LOG_FLD(2,<<"ModAnalyse2010: Not sure if player kicked or collided -> ignoring...");
        }
        else 
        {
          kickingPlayer=kickpl;
          if(kickpl<TEAM_SIZE) 
          {
            LOG_FLD(2,<<"ModAnalyse2010: Ball was kicked by own player #"<<kickpl+1);
          } 
          else 
          {
            LOG_FLD(2,<<"ModAnalyse2010: Ball was kicked by opponent player #"<<kickpl+1-TEAM_SIZE);
          }
	
          /* check kickrange... */
	
          double kr = (oldfld.ball.pos()-getPlByIndex(kickpl,&oldfld)->pos()).norm();
          for (int t=0;t<pltypes;t++) 
          {
      	    if (kr>kickRanges[t]+.001) 
            {
      	      possTypes[kickpl][t] += KICK_RANGE_WEIGHT;
      	      if (kickpl<TEAM_SIZE) 
              {
                LOG_FLD(2,<<"ModAnalyse2010: *** Own player #"<<kickpl+1<<" probably not of type "
                          <<t<<" (kickrange)");
      	      } 
              else 
              {
                LOG_FLD(2,<<"ModAnalyse2010: *** Opp player #"<<kickpl-TEAM_SIZE+1
                          <<" probably not of type " <<t<<" (kickrange)");
      	      }
      	    }
      	  }
	      }
      } 
      else 
      {
        LOG_FLD(2,<<"ModAnalyse2010: Ball was kicked, but there are several players in kickrange.");
      }
    }
  }
  return true;
}

/** Player Decay testen */
bool ModAnalyse2010::checkPlayerDecay() 
{
  double randx,randy,rmax,decay;
  Player *oldplayer,*player;
  for(int p=START_PL;p<2*TEAM_SIZE;p++) 
  {
    if (mayCollide[p] || canKick[p]) 
      continue;
    if ((oldplayer=getPlByIndex(p,&oldfld))==NULL) 
      continue;
    if ((player=getPlByIndex(p,&fld))==NULL) 
      continue;
    if ((player->pos()-oldplayer->pos()).norm()<0.000001) 
      continue; // kleine Bewegungen ignorieren
    if (    p!=kickingPlayer 
         && fabs((oldplayer->bodyAng-player->bodyAng).get_value())<0.001 )
      continue;
    //cout << "\nThinking...";
    rmax=ServerParam::player_rand*oldplayer->vel().norm();
    for(int t=0;t<pltypes;t++) 
    {
      decay=fld.plType[t].player_decay;
      randx=fabs((player->vel().getX()-decay*oldplayer->vel().getX())/decay);
      randy=fabs((player->vel().getY()-decay*oldplayer->vel().getY())/decay);
      //LOG_ERR(0,<<"randx = "<<randx<<", randy = "<<randy);
      if(randx>rmax+0.0000001 || randy>rmax+0.0000001) 
      {
        if (p<TEAM_SIZE) 
        {
          //LOG_FLD(2,<<"ModAnalyse2010: *** Own player #"<<p+1<<" probably not of type "
          //	  <<t<<" (player_decay)");
          LOG_FLD(2,<<"ModAnalyse2010: *** Own player #"<<p+1<<" not type "
                    <<t<<" (player_decay, rmax="<<rmax<<", randx="<<randx
                    <<", randy="<<randy<<")");
        } 
        else
        {
          //LOG_FLD(2,<<"ModAnalyse2010: *** Opp player #"<<p+1-TEAM_SIZE<<" probably not of type "
          //  <<t<<" (player_decay)");
          LOG_FLD(2,<<"ModAnalyse2010: *** Opp player #"<<p-TEAM_SIZE+1<<" not type "
      	    <<t<<" (player_decay, rmax="<<rmax<<", randx="<<randx<<", randy="<<randy<<")");
        }
        possTypes[p][t]+=PLAYER_DECAY_WEIGHT;
      }
    }
  }
  return true;
}     

/** ATTENTION: This Module does not seem to work correctly - it sometimes makes false assumptions! */
bool ModAnalyse2010::checkInertiaMoment() {
  Player *player,*oldplayer;
  for(int p=START_PL;p<TEAM_SIZE*2;p++) {
    if((player=getPlByIndex(p,&fld))==NULL) continue;
    if((oldplayer=getPlByIndex(p,&oldfld))==NULL) continue;
    double actangle=fabs(oldplayer->bodyAng.diff(player->bodyAng));
    if(actangle<0.000001) continue;
    double rmax=oldplayer->vel().norm()*ServerParam::player_rand;
    for(int t=0;t<pltypes;t++) {
      double ang=fabs(((1.0+rmax)*ServerParam::maxmoment)/(1.0+fld.plType[t].inertia_moment*
						      oldplayer->vel().norm()));
      if(actangle>ang+0.000001) {
	if(p<TEAM_SIZE) {
	  LOG_FLD(0,<<"ModAnalyse2010: *** Own player #"<<p+1<<" probably not of type "
		  <<t<<" (inertia) act="<<actangle<<" max="<<ang);
	} else {
	  LOG_FLD(0,<<"ModAnalyse2010: *** Opp player #"<<p-TEAM_SIZE+1<<" probably not of type "
		  <<t<<" (inertia) act="<<actangle<<" max="<<ang);
	}
	possTypes[p][t]+=INERTIA_MOMENT_WEIGHT;
      }
    }
  }
  return true;
}

bool ModAnalyse2010::checkPlayerSpeedMax() 
{
  Player *player,*oldplayer;
  for(int p=START_PL;p<TEAM_SIZE*2;p++) 
  {
    if (mayCollide[p]) 
      continue;
    if ((player=getPlByIndex(p,&fld))==NULL) 
      continue;
    if ((oldplayer=getPlByIndex(p,&oldfld))==NULL) 
      continue;
    double actxdist=fabs(player->pos().getX()-oldplayer->pos().getX());
    double actydist=fabs(player->pos().getY()-oldplayer->pos().getY());
    //double actspeed=(player->pos()-oldplayer->pos()).norm();
    for(int t=0;t<pltypes;t++) 
    {
      double maxspeed=fld.plType[t].player_speed_max;
      double rmax=ServerParam::player_rand*fld.plType[t].player_speed_max;
      //double maxspeed=
      double actspeed
        = sqrt((actxdist-rmax)*(actxdist-rmax)+(actydist-rmax)*(actydist-rmax));
      //LOG_DEF(0,<<"player #"<<p<<"type "<<t<<" speed="<<actspeed<<" max="<<maxspeed);
      if (actspeed>maxspeed+.0001) 
      {
        if (p<TEAM_SIZE) 
        {
          LOG_FLD(2,<<"ModAnalyse2010: *** Own player #"<<p+1<<" probably not of type "
            <<t<<" (player_speed_max)" <<"act="<<actspeed<<" max="<<maxspeed);
        } 
        else 
        {
          LOG_FLD(2,<<"ModAnalyse2010: *** Opp player #"<<p-TEAM_SIZE+1<<" probably not of type "
            <<t<<" (player_speed_max)" <<"act="<<actspeed<<" max="<<maxspeed);
        }
        possTypes[p][t] += PLAYER_SPEED_MAX_WEIGHT;
      }
    }
  }
  return true;
}



bool ModAnalyse2010::onRefereeMessage(bool PMChange) {
  
  return true;
}

bool ModAnalyse2010::onKeyboardInput(const char *str) {

  return false;
}

bool ModAnalyse2010::onHearMessage(const char *str) {

  return false;
}

bool ModAnalyse2010::onChangePlayerType(bool ownTeam,int unum,int type) {
  if(ownTeam) 
  {
    successMentioned[unum-1]=false;
    onlyOneLeft[unum-1]=false;
    for(int t=0;t<pltypes;t++) {
      //possTypes[unum-1][t]=0;
      //fld.myTeam[unum-1].possibleTypes[t]=0;
    }
    return true;
  }
  if(type<0) 
  {
    successMentioned[unum-1+TEAM_SIZE]=false;
    onlyOneLeft[unum-1+TEAM_SIZE]=false;
    for(int t=0;t<pltypes;t++) 
    {
      possTypes[unum-1+TEAM_SIZE][t]=0;
      fld.hisTeam[unum-1].possibleTypes[t]=0;
    }
  } 
  else 
  {
    //fld.hisTeam[unum-1].type=type;
  }
  /** tell this the players! */
  
  changeAnnounced=false;
    
  return true;
}


//END OF CLASS ModAnalyse2010

bool
ModAnalyse2010::updateSpeedProgressInformationArray()
{
  Player *oldPlayer,*curPlayer;
  
  for(int p=START_PL; p < 2*TEAM_SIZE; p++)
  {
    //consider current speed progression
    SpeedProgressionInformation 
        & curSpPrInfo       = ivSpeedProgressInformationArray[p]
                              .ivCurrentSpeedProgress;
    //check for skipping
    if ( mayCollide[p] || canKick[p] ) 
    {
      //invalidate history
      curSpPrInfo.ivLength = -1;
      continue;
    }
    if ( (oldPlayer = getPlByIndex(p,&oldfld)) == NULL ) continue;
    if ( (curPlayer = getPlByIndex(p,&fld))    == NULL ) continue;
    //determine velocity
    double curSpeed=(curPlayer->pos()-oldPlayer->pos()).norm();
    //do something useful
    if ( curSpeed < STILLSTANDING_PLAYER_THRESHOLD ) //TODO: Parameter einstellen 
    {
      //a new speed accelaration episode may start
      ivSpeedProgressInformationArray[p]
        .ivTimeOfRecentDashSequenceStart = fld.getTime() - 1; 
      curSpPrInfo.ivLength = 0;
    }
    else
    {
      //player is already running
      bool addCurrentSpeed = false;
      if ( curSpPrInfo.ivLength == -1 )
        addCurrentSpeed = false;
      else
      if ( curSpPrInfo.ivLength == 0 )
      { 
        addCurrentSpeed = true;
        ivSpeedProgressInformationArray[p].ivNumberOfHistoriesForStep[0] ++ ;
      }
      else
      {
        double recentSpeed
          = curSpPrInfo.ivSpeedProgress[ curSpPrInfo.ivLength - 1 ];
        //has the player accelarated?
        if ( curSpeed >= recentSpeed ) addCurrentSpeed = true;
        //has the current accelaraion episode already ended?
        //then, we will have to wait for a new episode to be started
        if (   ivSpeedProgressInformationArray[p]
                 .ivTimeOfRecentDashSequenceStart 
             + curSpPrInfo.ivLength + 2  <  fld.getTime()
         )
          addCurrentSpeed = false; 
      }
      if ( curSpPrInfo.ivLength == SPEED_PROGRESS_MAX )
        addCurrentSpeed = false;
      //do adding of information!
      if ( addCurrentSpeed == true )
      {
        //increment counter
        ivSpeedProgressInformationArray[p]
          .ivNumberOfHistoriesForStep[curSpPrInfo.ivLength] ++ ;
        //extend current speed progress information
        curSpPrInfo.ivSpeedProgress[ curSpPrInfo.ivLength ] = curSpeed;
        //check for updating the largest speed progress information!
        if (   curSpPrInfo.ivSpeedProgress[ curSpPrInfo.ivLength ]
             > ivSpeedProgressInformationArray[p].ivLargestSpeedProgressSoFar
                 .ivSpeedProgress[curSpPrInfo.ivLength] )
        {
          ivSpeedProgressInformationArray[p]
            .ivLargestSpeedProgressSoFar.ivSpeedProgress[ curSpPrInfo.ivLength ]
            = curSpPrInfo.ivSpeedProgress[ curSpPrInfo.ivLength ];
          ivSpeedProgressInformationArray[p].ivLargestSpeedProgressSoFar
            .ivNumberOfUpdates[ curSpPrInfo.ivLength ] ++ ;
          ivSpeedProgressInformationArray[p]
            .ivTimeOfLastLargestSpeedProgressUpdate = fld.getTime();          
          double recentMaxSpeed = 0.0;
          for (int l=0; l<SPEED_PROGRESS_MAX; l++)
          {
            if (      ivSpeedProgressInformationArray[p]
                            .ivLargestSpeedProgressSoFar
                            .ivSpeedProgress[l]
                    < recentMaxSpeed
                 || ivSpeedProgressInformationArray[p]
                            .ivNumberOfHistoriesForStep[l]
                    < 5  ) //TODO: Parameter einstellen!
              break;
            else
            {
              ivSpeedProgressInformationArray[p]
                .ivLargestSpeedProgressSoFar.ivLength = l+1;
              recentMaxSpeed = ivSpeedProgressInformationArray[p]
                                 .ivLargestSpeedProgressSoFar
                                 .ivSpeedProgress[l];
            }
          } 
/*//DEBUG
cout<< "pl_"<<p<<"\t"<<curPlayer->number<<"\t";
for (int i=0; i<SPEED_PROGRESS_MAX; i++)
  cout<<ivSpeedProgressInformationArray[p]
            .ivLargestSpeedProgressSoFar.ivSpeedProgress[ i ]<<"\t";
cout<<endl;
cout<< "pl_"<<p<<"\t"<<curPlayer->number<<"\t";
for (int i=0; i<SPEED_PROGRESS_MAX; i++)
      cout<<""<<ivSpeedProgressInformationArray[p].ivLargestSpeedProgressSoFar
            .ivNumberOfUpdates[ i]
      <<"("<<ivSpeedProgressInformationArray[p].ivNumberOfHistoriesForStep[i]<<"->"
      <<ivSpeedProgressInformationArray[p].ivLargestSpeedProgressSoFar.ivLength<<")\t";
cout<<endl;*/
        }
        //increment length
        curSpPrInfo.ivLength ++ ;
      }
    }
  }
  return true;
}

bool
ModAnalyse2010::checkSpeedProgress()
{
  //try to exploit the information gathered
  for (int p=START_PL; p < 2*TEAM_SIZE; p++)
  {
    if ( getNumberOfRemainingTypesForPlayer( p ) < 2 ) 
      continue;
#if LOGGING && BASIC_LOGGING
    int realType = (p<11) ? p : p-11;
#endif
    double minError = INT_MAX, maxError = 0.0;
    for (int t=0; t<pltypes; t++) 
    {
      if (    //too short episodic history
              ivSpeedProgressInformationArray[p]
                .ivLargestSpeedProgressSoFar.ivLength < 3
              //unreliable data: initial dash too small
              //[0.432 = 100*dash_power_rate*eff_max - 10%noise] 
           || ivSpeedProgressInformationArray[p]
                .ivLargestSpeedProgressSoFar.ivSpeedProgress[0] < 0.396
           || ivSpeedProgressInformationArray[p]
                .ivLargestSpeedProgressSoFar.ivSpeedProgress[1] < 0.5386
           || ivSpeedProgressInformationArray[p]
                .ivLargestSpeedProgressSoFar.ivSpeedProgress[2] < 0.5898
         )
      {
        ivSpeedProgressInformationArray[p].ivErrorForPlayerType[t] = -1.0;
        continue;
      }
      //compare the observed speed progress to the expected one
      double error = 0.0, delta;
      for (int l=0; l < ivSpeedProgressInformationArray[p]
                          .ivLargestSpeedProgressSoFar.ivLength; l++ )
      {
        delta =   ivSpeedProgressInformationArray[p]
                    .ivLargestSpeedProgressSoFar.ivSpeedProgress[l]
                - fld.plType[t].max_likelihood_max_speed_progress[l];
        error += delta * delta;
      }
      error = sqrt(error);
      ivSpeedProgressInformationArray[p].ivErrorForPlayerType[t] = error;
      if (error < minError) minError= error;
      if (error > maxError) maxError = error;
    }
    int exclusionCounter = 0;
    const double ERROR_SHARE_THRESHOLD = 0.3; //TODO: Parameter
    for (int t=0; t<pltypes; t++) 
    {
      if ( p < TEAM_SIZE && fld.myTeam[p].possibleTypes[t] < 0 ) continue;
      if ( p >=TEAM_SIZE && fld.hisTeam[p-TEAM_SIZE].possibleTypes[t] < 0) continue;      
      double error = ivSpeedProgressInformationArray[p].ivErrorForPlayerType[t];
      if ( error > ERROR_SHARE_THRESHOLD*(maxError-minError) + minError ) 
        exclusionCounter ++ ;
    }
    //avoid excluding all!
    if ( exclusionCounter >= getNumberOfRemainingTypesForPlayer( p ) ) 
      continue;
    for (int t=0; t<pltypes; t++) 
    {
      double error = ivSpeedProgressInformationArray[p].ivErrorForPlayerType[t];
      /*cout<<"t="<<fld.getTime()<<", pl_"<<p<<" (type="
        <<realType<<"): error for type "<<t<<" is "
        <<error<<" (minE="<<minError<<", maxE="<<maxError<<")"<<endl;*/        
      if ( error < 0.0 )
      {
        //cout<<"t="<<fld.getTime()<<", pl_"<<p<<": INSUFFICIENT INFORMATION."<<endl;
        break;
      }
      if ( error > ERROR_SHARE_THRESHOLD*(maxError-minError) + minError ) 
      {
        possTypes[p][t] += PLAYER_FUZZY_WEIGHT;
        //cout<<"t="<<fld.getTime()<<", pl_"<<p<<" (type="
          //<<realType<<") is probably NOT of type "<<t<<"."<<endl;
        if (p<TEAM_SIZE) 
        {
          LOG_FLD(0,<<"ModAnalyse2010: *** Own player #"<<p+1<<" probably not of type "
                    <<t<<" (speed progress) "
                    <<"errShare="<<((error-minError)/(maxError-minError))
                    <<" wech:"<<exclusionCounter<<" of "<<getNumberOfRemainingTypesForPlayer(p));
        } 
        else 
        {
          LOG_FLD(0,<<"ModAnalyse2010: *** Opp player #"<<p-TEAM_SIZE+1<<" probably not of type "
                    <<t<<" (speed progress) "
                    <<"errShare="<<((error-minError)/(maxError-minError))<<" wech:"<<exclusionCounter<<" of "<<getNumberOfRemainingTypesForPlayer(p));
        }
        /*//this block is for debugging only
        //can be used if all opponent players use types incrementing with
        //their player numbers!
        if ( realType == t )
        {
          LOG_FLD(0,<<"ModAnalyse2010: CRITICAL MISMATCH ("
            <<((error-minError)/(maxError-minError))<<")!!!";
          int num=0;
          for (int q=0; q<ivSpeedProgressInformationArray[p]
                          .ivLargestSpeedProgressSoFar.ivLength; q++ )
          {
            LOG_FLD(0, << "ModAnalyse2010:"
                       << ivSpeedProgressInformationArray[p]
                          .ivNumberOfHistoriesForStep[q]<<"("
                       << ivSpeedProgressInformationArray[p]
                          .ivLargestSpeedProgressSoFar.ivSpeedProgress[q]<<") ";
            num += ivSpeedProgressInformationArray[p]
                      .ivNumberOfHistoriesForStep[q];
          }
          LOG_FLD(0,<<"ModAnalyse2010: ==> "<<num);
        }*/
      }
    }
    LOG_FLD(0,<<"ModAnalyse2010: pl_"<<p<<" (type="
        <<realType<<"[only in DEBUG-Mode]): I exclded "
        <<exclusionCounter<<" types.");
    /*//show which players were excluded
    for (int t=0; t<pltypes; t++)
      cout<< t<<":"<<fld.myTeam[p].possibleTypes[t]<<"  ";
    cout<<endl;*/
    /*
    //debug output: show speed progress infos
    if (exclusionCounter==0)
    {
          int num=0;
          for (int q=0; q<ivSpeedProgressInformationArray[p]
                          .ivLargestSpeedProgressSoFar.ivLength; q++ )
          {
            cout << ivSpeedProgressInformationArray[p]
                      .ivNumberOfHistoriesForStep[q]<<"("
                 <<ivSpeedProgressInformationArray[p]
                   .ivLargestSpeedProgressSoFar.ivSpeedProgress[q]<<") ";
            num += ivSpeedProgressInformationArray[p]
                      .ivNumberOfHistoriesForStep[q];
          }
          cout<<" ==> "<<num<<endl;
    }*/
  }
  return true;
}

bool
ModAnalyse2010::checkMaxSpeedProgress()
{
  //try to exploit the information gathered
  for (int p=START_PL; p < 2*TEAM_SIZE; p++)
  {
    //check for skipping
    if ( mayCollide[p] || canKick[p] ) 
      continue;
    if (    fld.getTime() 
         != ivSpeedProgressInformationArray[p]
              .ivTimeOfLastLargestSpeedProgressUpdate ) 
      continue;
    //int realType = (p<11) ? p : p-11;
    for (int t=0; t<pltypes; t++)
    {
      for (int l=0; l<SPEED_PROGRESS_MAX; l++)
      {
        double theoreticalMax
          = fld.plType[t].max_speed_progress[l];
        double observedMax
          = ivSpeedProgressInformationArray[p]
                .ivLargestSpeedProgressSoFar.ivSpeedProgress[l];
        if ( observedMax > theoreticalMax + STILLSTANDING_PLAYER_THRESHOLD )
        {
          //cout << "t="<<fld.getTime()<<": PLAYER "<<p<<" CANNOT be of type "
          //  <<t<<" in step "<<l<<"."<<endl;
          possTypes[p][t] += PLAYER_MAX_SPEED_PROGRESS_WEIGHT;
          if (p<TEAM_SIZE) 
          {
            LOG_FLD(0,<<"ModAnalyse2010: *** Own player #"<<p+1<<" probably not of type "
                      <<t<<" (max speed progress) l="
                      <<l<<" observed="<<observedMax<<" theoMax="<<theoreticalMax
                      <<"(v="<<possTypes[p][t]<<")");
          } 
          else 
          {
            LOG_FLD(0,<<"ModAnalyse2010: *** Opp player #"<<p-TEAM_SIZE+1<<" probably not of type "
                      <<t<<" (max speed progress) l="
                      <<l<<" observed="<<observedMax<<" theoMax="<<theoreticalMax
                      <<"(v="<<possTypes[p][t]<<")");
          }
          /*
          //real type is only a debug feature: can be checked against if
          //opponent team uses player types with incrementing numbers
          if ( realType == t )
          {
            cout << "CRITICAL MISMATCH for player "<<p
              <<" at l="<<l<<". observed="<<observedMax<<" theoMax="<<theoreticalMax<<endl;
          }*/
        }
      }
    }
  }  
  return true;
}

bool ModAnalyse2010::checkInertiaMoment08() 
{
  Player *player,*oldplayer;
  for (int p=START_PL; p<TEAM_SIZE*2; p++) 
  {
    if ( (player=getPlByIndex(p,&fld)) == NULL ) 
      continue;
    if ( (oldplayer=getPlByIndex(p,&oldfld)) == NULL ) 
      continue;
    double turnAngle = fabs(oldplayer->bodyAng.diff(player->bodyAng));
    if ( turnAngle < 0.000001 ) 
      continue;
    //int realType = (p<11) ? p : p-11;
    for ( int t=0; t<pltypes; t++) 
    {
      double actualVelocity =  oldplayer->vel().norm() //already multiplied by
                                                      //player_decay, BUT
                            * (1.0-ServerParam::player_rand);
                                                      //vel may be reduced                          
      double maxTurnAngle = fabs(   ( ServerParam::maxmoment )
                                 / ( 1.0 +   fld.plType[t].inertia_moment
                                           * actualVelocity ) );
      if ( turnAngle > (1.0+ServerParam::player_rand)*maxTurnAngle+0.001 ) 
      {
        /*
        //real type can be used only when soccer server has assigned
        //player types (increasing numbers)
        if ( realType == t )
        {
          cout << "CRITICAL MISMATCH for player "<<p
            <<" with turnAngle="<<turnAngle<<" maxTurnAngle="<<maxTurnAngle
            <<" at vel="<<actualVelocity<<", SHARE: "<<turnAngle/maxTurnAngle<<endl;
        }*/
        if (p<TEAM_SIZE) 
        {
          LOG_FLD(0,<<"ModAnalyse2010: *** Own player #"<<p+1<<" probably not of type "
                    <<t<<" (inertia) act="<<turnAngle<<" max="<<maxTurnAngle<<" v="<<actualVelocity);
        } 
        else 
        {
          LOG_FLD(0,<<"ModAnalyse2010: *** Opp player #"<<p-TEAM_SIZE+1<<" probably not of type "
                    <<t<<" (inertia) act="<<turnAngle<<" max="<<maxTurnAngle<<" v="<<actualVelocity);
        }
        possTypes[p][t]+=INERTIA_MOMENT_WEIGHT;
      }
    }
  }
  return true;
}

bool
ModAnalyse2010::checkForAlreadyAssignedTypes()
{
  Player *player;
  for (int p=START_PL; p<TEAM_SIZE; p++) 
  {
    if ( (player=getPlByIndex(p,&fld)) == NULL ) 
      continue;
    if ( player->type >= 0 )
    {
      for (int o=START_PL; o<TEAM_SIZE; o++) 
        if ( o!=p && fld.myTeam[p].possibleTypes[player->type] != -1 )
        {
          //cout<<"ModAnalyse2010: *** Own player #"<<o+1<<" cannot be of type "
          //          <<player->type<<" (single use), as own player #"<<p+1<<" is already"<<endl;
          possTypes[o][player->type] += SINGLE_PLAYER_USAGE_WEIGHT;  
          LOG_FLD(0,<<"ModAnalyse2010: *** Own player #"<<o+1<<" cannot be of type "
                    <<player->type<<" (single use), as own player #"<<p+1<<" is already (v="
                    <<possTypes[o][player->type]<<")");
        }
    }
  }
  for (int p=TEAM_SIZE; p<2*TEAM_SIZE; p++) 
  {
    if ( (player=getPlByIndex(p,&fld)) == NULL ) 
      continue;
    if ( player->type >= 0 )
    {
      for (int o=TEAM_SIZE; o<2*TEAM_SIZE; o++) 
        if ( o!=p && fld.hisTeam[p-TEAM_SIZE].possibleTypes[player->type] != -1 )
        {
          //cout<<"ModAnalyse2010: *** Opp player #"<<o+1<<" cannot be of type "
          //          <<player->type<<" (single use), as opp player #"<<p+1<<" is already"<<endl;
          possTypes[o][player->type] += SINGLE_PLAYER_USAGE_WEIGHT;  
          LOG_FLD(0,<<"ModAnalyse2010: *** Opp player #"<<o+1<<" cannot be of type "
                    <<player->type<<" (single use), as opp player #"<<p+1<<" is already (v="
                    <<possTypes[o][player->type]<<")");
        }
    }
  }
  return true;
}

int 
ModAnalyse2010::getNumberOfRemainingTypesForPlayer( int plIdx )
{
  int returnValue = 0;
  if ( plIdx < TEAM_SIZE )
  {
    //my team
    for ( int t=0; t<pltypes; t++)
      if ( fld.myTeam[plIdx].possibleTypes[t] >= 0 ) returnValue ++ ;    
  }
  else
  {
    //my team
    for ( int t=0; t<pltypes; t++)
      if ( fld.hisTeam[plIdx-TEAM_SIZE].possibleTypes[t] >= 0 ) returnValue ++ ;    
  }
  return returnValue;
}

bool 
ModAnalyse2010::checkPlayerDecay09()
{
  for (int playerIndex = START_PL; playerIndex < 2*TEAM_SIZE; playerIndex++) 
  {
    if ( mayCollide[playerIndex] == true )
    {
      LOG_FLD(0,<<"ModAnalyse2010: === checkPlayerDecay09 === ABORTED for player "<<playerIndex
        <<" because of COLLISION.");
      continue;
    }
    
    LOG_FLD(0,<<"ModAnalyse2010: === checkPlayerDecay09 ===");
    Player* player = getPlByIndex(playerIndex, &fld);
    if (player == NULL) continue;
    Player* oldplayer = getPlByIndex(playerIndex, &oldfld);
    if (oldplayer == NULL) continue;
    double posDiff = (player->pos() - oldplayer->pos()).norm();

    // ignore small movements
    if (posDiff < MOVEMENT_EPSILON)
      continue;

    // **TODO** make sure player position is inside the field boundaries
    // (may be guaranteed by PM_play_on)

    double decay(player->vel().norm() / posDiff);
    double minDiff(100.0);
    int playerType(-1);
    for (int i = 0; i < PlayerParam::player_types; ++i) 
    {
      double diff = fabs(decay - fld.plType[i].player_decay);
      if (diff < minDiff) 
      {
        minDiff = diff;
        playerType = i;
      }
    }

    if (playerType != -1) 
    {
      for ( int allOtherTypes = 0; allOtherTypes < pltypes; allOtherTypes ++ )
        if ( allOtherTypes != playerType )
          possTypes[playerIndex][allOtherTypes] += PLAYER_DECAY09_WEIGHT;
      if (playerIndex<TEAM_SIZE) 
      {
        LOG_FLD(0,<<"ModAnalyse2010: *** Own player #"<<playerIndex+1<<" is presumably of type "
                  <<playerType<<" (player decay 09, direct assignment) ");
      }
      else
      {
        LOG_FLD(0,<<"ModAnalyse2010: *** Opponent player #"<<playerIndex-TEAM_SIZE+1<<" is presumably of type "
                  <<playerType<<" (player decay 09, direct assignment) ");
      } 
    }
  }
  return true;
}

void 
ModAnalyse2010::showPossTypeTable()
{
  for (int playerIndex = START_PL; playerIndex < 2*TEAM_SIZE; playerIndex++) 
  {
    LOG_FLD(0,<<(((playerIndex+1)>11)?(playerIndex-TEAM_SIZE+1):(playerIndex+1))<<"\t"
              <<possTypes[playerIndex][0]<<"\t"
              <<possTypes[playerIndex][1]<<"\t"
              <<possTypes[playerIndex][2]<<"\t"
              <<possTypes[playerIndex][3]<<"\t"
              <<possTypes[playerIndex][4]<<"\t"
              <<possTypes[playerIndex][5]<<"\t"
              <<possTypes[playerIndex][6]<<"\t"
              <<possTypes[playerIndex][7]<<"\t"
              <<possTypes[playerIndex][8]<<"\t"
              <<possTypes[playerIndex][9]<<"\t"
              <<possTypes[playerIndex][10]<<"\t"
              <<possTypes[playerIndex][11]<<"\t"
              <<possTypes[playerIndex][12]<<"\t"
              <<possTypes[playerIndex][13]<<"\t"
              <<possTypes[playerIndex][14]<<"\t"
              <<possTypes[playerIndex][15]<<"\t"
              <<possTypes[playerIndex][16]<<"\t"
              <<possTypes[playerIndex][17]<<"\t"
    );
  }
  for (int playerIndex = 0; playerIndex < TEAM_SIZE; playerIndex++) 
  {
    LOG_FLD(0,<<"MMM "<<playerIndex+1<<"\t"
              <<fld.myTeam[playerIndex].possibleTypes[0]<<"\t"
              <<fld.myTeam[playerIndex].possibleTypes[1]<<"\t"
              <<fld.myTeam[playerIndex].possibleTypes[2]<<"\t"
              <<fld.myTeam[playerIndex].possibleTypes[3]<<"\t"
              <<fld.myTeam[playerIndex].possibleTypes[4]<<"\t"
              <<fld.myTeam[playerIndex].possibleTypes[5]<<"\t"
              <<fld.myTeam[playerIndex].possibleTypes[6]<<"\t"
              <<fld.myTeam[playerIndex].possibleTypes[7]<<"\t"
              <<fld.myTeam[playerIndex].possibleTypes[8]<<"\t"
              <<fld.myTeam[playerIndex].possibleTypes[9]<<"\t"
              <<fld.myTeam[playerIndex].possibleTypes[10]<<"\t"
              <<fld.myTeam[playerIndex].possibleTypes[11]<<"\t"
              <<fld.myTeam[playerIndex].possibleTypes[12]<<"\t"
              <<fld.myTeam[playerIndex].possibleTypes[13]<<"\t"
              <<fld.myTeam[playerIndex].possibleTypes[14]<<"\t"
              <<fld.myTeam[playerIndex].possibleTypes[15]<<"\t"
              <<fld.myTeam[playerIndex].possibleTypes[16]<<"\t"
              <<fld.myTeam[playerIndex].possibleTypes[17]<<"\t"
    );
  }
  for (int playerIndex = 0; playerIndex < TEAM_SIZE; playerIndex++) 
  {
    LOG_FLD(0,<<"HHH "<<playerIndex+1<<"\t"
              <<fld.hisTeam[playerIndex].possibleTypes[0]<<"\t"
              <<fld.hisTeam[playerIndex].possibleTypes[1]<<"\t"
              <<fld.hisTeam[playerIndex].possibleTypes[2]<<"\t"
              <<fld.hisTeam[playerIndex].possibleTypes[3]<<"\t"
              <<fld.hisTeam[playerIndex].possibleTypes[4]<<"\t"
              <<fld.hisTeam[playerIndex].possibleTypes[5]<<"\t"
              <<fld.hisTeam[playerIndex].possibleTypes[6]<<"\t"
              <<fld.hisTeam[playerIndex].possibleTypes[7]<<"\t"
              <<fld.hisTeam[playerIndex].possibleTypes[8]<<"\t"
              <<fld.hisTeam[playerIndex].possibleTypes[9]<<"\t"
              <<fld.hisTeam[playerIndex].possibleTypes[10]<<"\t"
              <<fld.hisTeam[playerIndex].possibleTypes[11]<<"\t"
              <<fld.hisTeam[playerIndex].possibleTypes[12]<<"\t"
              <<fld.hisTeam[playerIndex].possibleTypes[13]<<"\t"
              <<fld.hisTeam[playerIndex].possibleTypes[14]<<"\t"
              <<fld.hisTeam[playerIndex].possibleTypes[15]<<"\t"
              <<fld.hisTeam[playerIndex].possibleTypes[16]<<"\t"
              <<fld.hisTeam[playerIndex].possibleTypes[17]<<"\t"
    );
  }
}

const char ModAnalyse2010::modName[]="ModAnalyse2010";

