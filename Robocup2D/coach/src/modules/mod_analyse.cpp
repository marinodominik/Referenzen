/* Author: Manuel "Sputnick" Nickschas, 02/2002
 *
 * This module analyses the opponent team and tries to find out the opponent
 * player types.
 *
 */

#include "angle.h"
#include "options.h"
#include "logger.h"
#include <sstream>

#include "mod_analyse.h"
#include "messages.h"

/** The number of hints needed to be sure enough to exclude a player type */
#define HINTS_TO_BE_SURE 1
#define HINTS_TO_RECONSIDER 15

/** These weights balance the importance of different hints. */
#define KICK_RANGE_WEIGHT 2
#define PLAYER_DECAY_WEIGHT 5
#define INERTIA_MOMENT_WEIGHT 5
#define PLAYER_SPEED_MAX_WEIGHT 10

#define START_PL 0  //DON'T MODIFY! Feature not working!

bool ModAnalyse::init(int argc,char **argv) {
  pltypes=PlayerParam::player_types;

  oldfld=fld;
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
  
  return true;
}

bool ModAnalyse::destroy() {
  
  return true;
}

Player *ModAnalyse::getPlByIndex(int i,Field *field) {
  if(i<TEAM_SIZE) {
    if(field->myTeam[i].alive) return &field->myTeam[i];
    return NULL;
  } else {
    if(field->hisTeam[i-TEAM_SIZE].alive) return &field->hisTeam[i-TEAM_SIZE];
    return NULL;
  }
}

bool ModAnalyse::behave() {

  static std::ostringstream oss;
  
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
	      LOG_FLD(1,<<"ModAnalyse: **** I am sure: Own player #"<<p+1<<" is not of type "
		      <<t<<"! ("<<fld.myTeam[p].possibleTypes[t]<<" Points)");
	      fld.myTeam[p].possibleTypes[t]=-1;cnt++;
	    }
	  } else {
	    if(fld.myTeam[p].possibleTypes[t]>HINTS_TO_RECONSIDER) {
	      LOG_FLD(0,<<"ModAnalyse: RECONSIDER own player #"<<p+1<<", not of type "<<t<<"!");
	      std::cerr << "\nModAnalyse: RECONSIDER own player #"<<p+1<<", not of type "<<t<<"!";
	      fld.myTeam[p].possibleTypes[t]=-1;cnt++;
	      //RUN::announcePlayerChange(true,p+1,-1);
	    }
	  }
	}
	else {cnt++;}
      }
      if(cnt==pltypes) {
	LOG_ERR(0,<<"ERROR: ModAnalyse: Something went really wrong, there is no possible type for "
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
	  LOG_DEF(0,<<"ModAnalyse: ****** I am sure: My player #"<<p+1<<" must be of type "<<t<<"!");
	  successMentioned[p]=true;flg=true;onlyOneLeft[p]=true;
	}
      }

      cnt=0;
      for(int t=0;t<pltypes;t++) {
	if(fld.hisTeam[p].possibleTypes[t]>=0) {
	  fld.hisTeam[p].possibleTypes[t]+=possTypes[p+TEAM_SIZE][t];
	  if(fld.hisTeam[p].type!=t) {
	    if(fld.hisTeam[p].possibleTypes[t]>HINTS_TO_BE_SURE) {
	      LOG_FLD(1,<<"ModAnalyse: **** I am sure: Opponent player #"<<p+1<<" is not of type "
		      <<t<<"! ("<<fld.hisTeam[p].possibleTypes[t]<<" Points)");
	      fld.hisTeam[p].possibleTypes[t]=-1;cnt++;
	    }
	  } else {
	    if(fld.hisTeam[p].possibleTypes[t]>HINTS_TO_RECONSIDER) {
	      LOG_FLD(0,<<"ModAnalyse: RECONSIDER opp player #"<<p+1<<", not of type "<<t<<"!");
	      std::cerr << "\nModAnalyse: RECONSIDER opp player #"<<p+1<<", not of type "<<t<<"!";
	      fld.hisTeam[p].possibleTypes[t]=-1;cnt++;
	      RUN::announcePlayerChange(false,p+1,-1);
	    }
	  }
	}
	else {cnt++;}
      }
      if(cnt==pltypes) {
	LOG_ERR(0,<<"ERROR: ModAnalyse: Something went really wrong, there is no possible type for "
		<<" opp player #"<<p+1);
	flg=true;
	RUN::announcePlayerChange(false,p+1,-1);
      }
      if(cnt==pltypes-1) {
	if(!successMentioned[p+TEAM_SIZE]) {
	  int t;
	  for(t=0;t<pltypes;t++) if(fld.hisTeam[p].possibleTypes[t]>=0) break;
	  LOG_DEF(0,<<"ModAnalyse: ****** I am sure: Opponent player #"<<p+1
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
      LOG_DEF(0,<<"ModAnalyse: Revised table of known player types:");
      LOG_DEF(0,<<"ModAnalyse:");
      LOG_DEF(0,<<"ModAnalyse:  Pl | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | A | B |");
      LOG_DEF(0,<<"ModAnalyse: -------------------------------------------------");
      if(START_PL<TEAM_SIZE) {
	oss.seekp(0);
	oss << "ModAnalyse: Own |";
	for(int p=0;p<TEAM_SIZE;p++) {
	  if(!fld.myTeam[p].alive) {oss << " - |";continue;}
	  if(p<START_PL) {
	    oss << " x |";continue;
	  }
	  if(onlyOneLeft[p]) {
	    int j;
	    for(j=0;j<pltypes;j++) if(fld.myTeam[p].possibleTypes[j]>=0) break;
	    oss << " "<<j<<" |";continue;
	  }
	  oss << " ? |";
	}
	LOG_DEF(0,<<oss.str());
      }
      oss.seekp(0);
      oss << "ModAnalyse: Opp |";
      for(int p=0;p<TEAM_SIZE;p++) {
	if(!fld.hisTeam[p].alive) {oss << " - |";continue;}
	if(p<START_PL-TEAM_SIZE) {oss << " x |";continue;}
	if(fld.hisTeam[p].type<0 && fld.hisTeam[p].possibleTypes[0]>=0) {
	  oss << " * |";continue;}
	if(fld.hisTeam[p].type<0) {oss << " ? |";continue;}
	oss << " "<<fld.hisTeam[p].type<<" |";
      }
      oss << ends;
      LOG_DEF(0,<<oss.str());
      
    }
  }
  /* Announce changes to players, if necessary (and possible) */
  if(fld.getTime()>0 && (!changeAnnounced || !goalieAnnounced)) {
    using namespace MSG;
    //LOG_MSG(0,<<"Telling the players...");
    oss.seekp(0);
    oss << fld.getTime() << " (true) ";
    oss << "\"pt";
    int goalie=-1;
    for(int i=0;i<TEAM_SIZE;i++) {
      if(fld.hisTeam[i].type<0) oss << "_";
      else oss << fld.hisTeam[i].type;
      if(fld.hisTeam[i].goalie) goalie=fld.hisTeam[i].number;
    }
    if(goalie>0) oss << " g"<<goalie;
    else oss << " g_";
    oss << "\"";
    oss <<ends;
    if(fld.getPM()==PM_play_on) {
      if(fld.getTime()==1 || fld.getTime()>=50) {
	if(fld.sayInfo.lastInPlayOn[MSGT_INFO]>=0 &&
	   fld.getTime()-fld.sayInfo.lastInPlayOn[MSGT_INFO]<=ServerParam::clang_win_size) {
	  if(fld.sayInfo.lastInPlayOn[MSGT_ADVICE]<0 ||
	     fld.getTime()-fld.sayInfo.lastInPlayOn[MSGT_ADVICE]>ServerParam::clang_win_size) {
	    if(fld.getTime()-fld.sayInfo.lastInPlayOn[MSGT_INFO]>=100) {
	      sendMsg(MSG_SAY,MSGT_ADVICE,oss.str());
	      changeAnnounced=true;
	      if(goalie>0) goalieAnnounced=true;
	    }
	  }
	} else {
	  sendMsg(MSG_SAY,MSGT_INFO,oss.str());
	  if(goalie>0) goalieAnnounced=true;
	  changeAnnounced=true;
	}
      }
    } else {
      sendMsg(MSG_SAY,MSGT_INFO,oss.str());
      if(goalie>0) goalieAnnounced=true;
      changeAnnounced=true;
    }
    if(!goalieAnnounced && fld.getTime()>=10) {
      LOG_FLD(1,<<"Opponent does not seem to have a goalie!");
      goalieAnnounced=true;
    }
  }
  
  oldfld=fld;oldpm=fld.getPM();
  return true;
}

/** check which players may have collided */
void ModAnalyse::checkCollisions() {
  Player *player,*tmpplayer;
  double size,tmpsize;
  int type,tmptype;
  for(int p=START_PL;p<2*TEAM_SIZE;p++) {
    mayCollide[p]=false;
    if((player=getPlByIndex(p,&fld))==NULL) continue;
    if(player->type==-1) type=0;else type=player->type;  // worst case if unknown
    size=fld.plType[type].player_size;
    bool flg=false;
    for(int p2=0;p2<TEAM_SIZE*2;p2++) {   // Check collision
      if(p2==p) continue;
      if((tmpplayer=getPlByIndex(p2,&fld))==NULL) continue;
      if(tmpplayer->type==-1) tmptype=0;else tmptype=tmpplayer->type;
      tmpsize=fld.plType[tmptype].player_size;
      if((tmpplayer->pos()-player->pos()).norm()<size+tmpsize+0.2) {
	flg=true;break;
      }
    }
    if(flg) {
      mayCollide[p]=true;
      if(p<TEAM_SIZE) {
	LOG_FLD(1,<<"Own player #"<<p+1<<" collided with other player!");
      } else {
	LOG_FLD(1,<<"Opp player #"<<p+1-TEAM_SIZE<<" collided with other player!");
      }
      continue;
    }
    if((player->pos()-fld.ball.pos()).norm()<size+ServerParam::ball_size+0.1) {
      if(p<TEAM_SIZE) {
	LOG_FLD(1,<<"Own player #"<<p+1<<" collided with ball!");
      } else {
	LOG_FLD(1,<<"Opp player #"<<p-TEAM_SIZE+1<<" collided with ball!");	
      }
      mayCollide[p]=true;
    }
  }
}      

/** Ball gekickt? Kickrange gross genug? */
bool ModAnalyse::checkKick() {
  
  ballKicked=false;  
  kickingPlayer=-1;

  Player *player;
  
  Vector u = oldfld.ball.vel();
  Vector newpos=oldfld.ball.pos()+u;
  Vector newvel=ServerParam::ball_decay*u;
  double rmax=ServerParam::ball_rand*oldfld.ball.vel().norm();
  
  if((fabs(fld.ball.pos().getX()-newpos.getX())>rmax) ||
     (fabs(fld.ball.pos().getY()-newpos.getY())>rmax)) ballKicked=true;
  if((fabs(fld.ball.vel().getX()-newvel.getX())>ServerParam::ball_decay*rmax) ||
     (fabs(fld.ball.vel().getX()-newvel.getX())>ServerParam::ball_decay*rmax)) ballKicked=true;

  int cnt=0;int kickpl=-1;
  if(ballKicked) {
    for(int i=0;i<TEAM_SIZE*2;i++) canKick[i]=false;
    //LOG_FLD(0,<<"ModAnalyse: Ball has been kicked!");
    for(int i=0;i<TEAM_SIZE*2;i++) {
      if((player=getPlByIndex(i,&oldfld))!=NULL) {
	if((oldfld.ball.pos()-player->pos()).norm()<=maxKickRange) {
	  canKick[i]=true;cnt++;kickpl=i;
	}
      }
    }
    if(!cnt) {
      LOG_FLD(1,<<"ModAnalyse: Ball has been kicked, but I don't know by whom... Collision?");
    } else {
      if(cnt==1) {
	if(mayCollide[kickpl]) {   // ignore in case of possible collision
	  LOG_FLD(2,<<"ModAnalyse: Not sure if player kicked or collided -> ignoring...");
	} else {
	  kickingPlayer=kickpl;
	  if(kickpl<TEAM_SIZE) {
	    LOG_FLD(2,<<"ModAnalyse: Ball was kicked by own player #"<<kickpl+1);
	  } else {
	    LOG_FLD(2,<<"ModAnalyse: Ball was kicked by opponent player #"<<kickpl+1-TEAM_SIZE);
	  }
	
	  /* check kickrange... */
	
	  double kr=(oldfld.ball.pos()-getPlByIndex(kickpl,&oldfld)->pos()).norm();
	  for(int t=0;t<pltypes;t++) {
	    if(kr>kickRanges[t]+.001) {
	      possTypes[kickpl][t]+=KICK_RANGE_WEIGHT;
	      if(kickpl<TEAM_SIZE) {
		LOG_FLD(2,<<"ModAnalyse: *** Own player #"<<kickpl+1<<" probably not of type "
			<<t<<" (kickrange)");
	      } else {
		LOG_FLD(2,<<"ModAnalyse: *** Opp player #"<<kickpl-TEAM_SIZE+1
			<<" probably not of type " <<t<<" (kickrange)");
	      }
	    }
	  }
	}
      } else {
	LOG_FLD(2,<<"ModAnalyse: Ball was kicked, but there are several players in kickrange.");
      }
    }
  }
  return true;
}

/** Player Decay testen */
bool ModAnalyse::checkPlayerDecay() {
  double randx,randy,rmax,decay;
  Player *oldplayer,*player;
  for(int p=START_PL;p<2*TEAM_SIZE;p++) {
    if(mayCollide[p] || canKick[p]) continue;
    if((oldplayer=getPlByIndex(p,&oldfld))==NULL) continue;
    if((player=getPlByIndex(p,&fld))==NULL) continue;
    if((player->pos()-oldplayer->pos()).norm()<0.000001) continue; // kleine Bewegungen ignorieren
    if(p!=kickingPlayer && fabs((oldplayer->bodyAng-player->bodyAng).get_value())<0.001)
      continue;
    //cout << "\nThinking...";
    rmax=ServerParam::player_rand*oldplayer->vel().norm();
    for(int t=0;t<pltypes;t++) {
      decay=fld.plType[t].player_decay;
      randx=fabs((player->vel().getX()-decay*oldplayer->vel().getX())/decay);
      randy=fabs((player->vel().getY()-decay*oldplayer->vel().getY())/decay);
      //LOG_ERR(0,<<"randx = "<<randx<<", randy = "<<randy);
      if(randx>rmax+0.0000001 || randy>rmax+0.0000001) {
	if(p<TEAM_SIZE) {
	  //LOG_FLD(2,<<"ModAnalyse: *** Own player #"<<p+1<<" probably not of type "
	  //	  <<t<<" (player_decay)");
	  LOG_FLD(2,<<"ModAnalyse: *** Own player #"<<p+1<<" not type "
	  	  <<t<<" (player_decay, rmax="<<rmax<<", randx="<<randx<<", randy="<<randy<<")");
	} else {
	  //LOG_FLD(2,<<"ModAnalyse: *** Opp player #"<<p+1-TEAM_SIZE<<" probably not of type "
	  //  <<t<<" (player_decay)");
	  LOG_FLD(2,<<"ModAnalyse: *** Opp player #"<<p-TEAM_SIZE+1<<" not type "
	  	  <<t<<" (player_decay, rmax="<<rmax<<", randx="<<randx<<", randy="<<randy<<")");
	}
	possTypes[p][t]+=PLAYER_DECAY_WEIGHT;
      }
    }
  }
  return true;
}     

/** ATTENTION: This Module does not seem to work correctly - it sometimes makes false assumptions! */
bool ModAnalyse::checkInertiaMoment() {
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
	  LOG_FLD(0,<<"ModAnalyse: *** Own player #"<<p+1<<" probably not of type "
		  <<t<<" (inertia) act="<<actangle<<" max="<<ang);
	} else {
	  LOG_FLD(0,<<"ModAnalyse: *** Opp player #"<<p-TEAM_SIZE+1<<" probably not of type "
		  <<t<<" (inertia) act="<<actangle<<" max="<<ang);
	}
	possTypes[p][t]+=INERTIA_MOMENT_WEIGHT;
      }
    }
  }
  return true;
}

bool ModAnalyse::checkPlayerSpeedMax() {
  Player *player,*oldplayer;
  for(int p=START_PL;p<TEAM_SIZE*2;p++) {
    if(mayCollide[p]) continue;
    if((player=getPlByIndex(p,&fld))==NULL) continue;
    if((oldplayer=getPlByIndex(p,&oldfld))==NULL) continue;
    double actxdist=fabs(player->pos().getX()-oldplayer->pos().getX());
    double actydist=fabs(player->pos().getY()-oldplayer->pos().getY());
    //double actspeed=(player->pos()-oldplayer->pos()).norm();
    for(int t=0;t<pltypes;t++) {
      double maxspeed=fld.plType[t].player_speed_max;
      double rmax=ServerParam::player_rand*fld.plType[t].player_speed_max;
      //double maxspeed=
      double actspeed=sqrt((actxdist-rmax)*(actxdist-rmax)+(actydist-rmax)*(actydist-rmax));
      //LOG_DEF(0,<<"player #"<<p<<"type "<<t<<" speed="<<actspeed<<" max="<<maxspeed);
      if(actspeed>maxspeed+.0001) {
	if(p<TEAM_SIZE) {
	  LOG_FLD(2,<<"ModAnalyse: *** Own player #"<<p+1<<" probably not of type "
		  <<t<<" (player_speed_max)" <<"act="<<actspeed<<" max="<<maxspeed);
	} else {
	  LOG_FLD(2,<<"ModAnalyse: *** Opp player #"<<p-TEAM_SIZE+1<<" probably not of type "
		  <<t<<" (player_speed_max)" <<"act="<<actspeed<<" max="<<maxspeed);
	}
	possTypes[p][t]+=PLAYER_SPEED_MAX_WEIGHT;
      }
    }
  }
  return true;
}



bool ModAnalyse::onRefereeMessage(bool PMChange) {
  
  return true;
}

bool ModAnalyse::onKeyboardInput(const char *str) {

  return false;
}

bool ModAnalyse::onHearMessage(const char *str) {

  return false;
}

bool ModAnalyse::onChangePlayerType(bool ownTeam,int unum,int type) {
  if(ownTeam) {
    successMentioned[unum-1]=false;
    onlyOneLeft[unum-1]=false;
    for(int t=0;t<pltypes;t++) {
      possTypes[unum-1][t]=0;
      fld.myTeam[unum-1].possibleTypes[t]=0;
    }
    return true;
  }
  if(type<0) {
    successMentioned[unum-1+TEAM_SIZE]=false;
    onlyOneLeft[unum-1+TEAM_SIZE]=false;
    for(int t=0;t<pltypes;t++) {
      possTypes[unum-1+TEAM_SIZE][t]=0;
      fld.hisTeam[unum-1].possibleTypes[t]=0;
      
    }
  } else {
    //fld.hisTeam[unum-1].type=type;

  }
  /** tell this the players! */
  
  changeAnnounced=false;
    
  return true;
}

const char ModAnalyse::modName[]="ModAnalyse";
