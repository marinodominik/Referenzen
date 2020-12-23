/* Author: Manuel "Sputnick" Nickschas, 02/2002
 *
 * ModChange changes player types in own team.
 * See mod_change.h for description.
 *
 */

#include "defs.h"
#include "coach.h"
#include "logger.h"
#include "valueparser.h"
#include "str2val.h"
#include "mod_change.h"

const char ModChange::modName[]="ModChange";

bool ModChange::init(int argc,char **argv) {

  ValueParser vp(Options::coach_conf,"ModChange");
  vp.get("fastest",fastestPlayers,3);
  vp.get("wait_for_team",waitForTeam,1);
  //for(int i=0;i<3;i++) {
  //  cout << " " << fastestPlayers[i];
  //}

  queueCnt=0;totalSubs=0;prepDone=false;
  for(int i=0;i<PlayerParam::player_types;i++) typesOnFld[i]=0;
  left_goals=right_goals=0;
  emerg_change=false;
  return true;
}

bool ModChange::destroy() {
  cout << "\nModChange destroy!";

  return true;
}

bool ModChange::behave() {

  /*************************************************************************/
  /** Check queue for succeeded substitutions and remove completed entries */

  if(queueCnt) sendQueue();
  
  MsgChangePlayerType *msg=(MsgChangePlayerType*)MSG::msg[MSG::MSG_CHANGE_PLAYER_TYPE];
  int i,j;
  for(i=0;i<queueCnt;i++) {
    if(subQueue[i].time>=0) {
      for(int j=msg->subsCnt-1;j>=0;j--) {
	if(msg->subsDone[j].done!=1) continue;
	if(msg->subsDone[j].unum!=subQueue[i].unum || msg->subsDone[j].type!=subQueue[i].type)
	  continue;
	msg->subsDone[j].done=2;
	subQueue[i].done=1;
	if(fld.getTime()>0) totalSubs++;
	break;
      }
      if(subQueue[i].done==0) {
	if(fld.getTime()-subQueue[i].time>5) {
	  LOG_ERR(0,<< "Could not change player " << subQueue[i].unum << " to type "
		  <<subQueue[i].type<<" within 5 cycles!");
	  subQueue[i].done=-1;
	}
      }
    }
  }
  int newcnt=queueCnt;
  for(i=0,j=0;i<queueCnt;i++) {
    if(subQueue[i].done!=0) {
      newcnt--;continue;
    }
    subQueue[j].unum=subQueue[i].unum;subQueue[j].type=subQueue[i].type;
    subQueue[j].time=subQueue[i].time;subQueue[j].done=subQueue[i].done;
    j++;
  }
  queueCnt=newcnt;

  /***************************************************************************/
  /**                                                                        */

  if(!prepDone) prepareForGame();

  for(int i=0;i<PlayerParam::player_types;i++) typesOnFld[i]=0;
  for(int i=0;i<11;i++) {
    if(!fld.myTeam[i].alive) continue;
    typesOnFld[fld.myTeam[i].type]++;
  }

  //cout <<"\n";
  //for(int i=0;i<PlayerParam::player_types;i++)
  //  cout << "["<<typesOnFld[i]<<"]";

  if(emerg_change) {
    if(fld.getTime()>=2800 && left_goals<=right_goals) {
      changePlayer(9,emerg_type);
      changePlayer(10,emerg_type);
      changePlayer(11,emerg_type);
      //std::cerr << "\n\n\nCHANGING!!!\n\n\n";
      emerg_change=false;
    }
  }
    
  return true;
}

bool ModChange::onRefereeMessage(bool PMChange) {
  if(fld.getPM()!=PM_play_on && fld.getPM()<PM_MAX) sendQueue();
  if(fld.getPM()==RM_goal_l) left_goals++;
  if(fld.getPM()==RM_goal_r) right_goals++;
  return true;
}

bool ModChange::onKeyboardInput(const char *str) {
  const char *sdum;
  int unum,type;

  if(strskip(str,"chng ",sdum)) {
    str=sdum;
    if(str2val(str,unum,str) && str2val(str,type,str)) {
      changePlayer(unum,type);
    }
    return true;
  }
  return false;
}

bool ModChange::changePlayer(int unum,int type) {
  if(queueCnt>=11) {
    LOG_ERR(0,<<"Too many substitutions during this play_on cycle!");return false;
  }
  //std::cerr << "\nchange: "<< unum <<" -> "<<type;
  subQueue[queueCnt].unum=unum;
  subQueue[queueCnt].type=type;
  subQueue[queueCnt].time=-1;
  subQueue[queueCnt++].done=0;
  subsToDo++;
  if(fld.getPM()!=PM_play_on && fld.getPM()<PM_MAX) sendQueue();
  return true;
}

bool ModChange::sendQueue() {
  if(queueCnt) {
    if(fld.getPM()!=PM_play_on && fld.getPM()<PM_MAX) {
      for(int i=0;i<queueCnt;i++) {
	if(!subQueue[i].done && subQueue[i].time<0) {
	  for(int j=0;j<11;j++) {
	    if(fld.myTeam[j].number==subQueue[i].unum && fld.myTeam[j].alive) {
	      sendMsg(MSG::MSG_CHANGE_PLAYER_TYPE,subQueue[i].unum,subQueue[i].type);
	      subQueue[i].time=fld.getTime();
	      break;
	    }
	  }
	}
      }
    }
  }
  return true;
}

int ModChange::cmp_real_player_speed_max(const PlayerType *t1,const PlayerType *t2) {
  if(t1->real_player_speed_max > t2->real_player_speed_max) return 1;
  if(t1->real_player_speed_max < t2->real_player_speed_max) return -1;
  return 0;
  
}

int ModChange::cmp_stamina10m(const PlayerType *t1,const PlayerType *t2) {
   if(t1->stamina_10m > t2->stamina_10m) return 1;
   if(t1->stamina_10m < t2->stamina_10m) return -1;
   return 0;
}

void ModChange::prepareForGame() {
  if(waitForTeam) {
    int cnt=0;
    for(int i=0;i<11;i++) if(fld.myTeam[i].alive) cnt++;
    if(cnt<11) return;
  }
#if 0
  Value bestkick=-7;
  int besttype;
  for(int t=0;t<7;t++) {
    if(fld.plType[t].kickable_margin>bestkick) {
      bestkick = fld.plType[t].kickable_margin;
      besttype=t;
    }
  }
  prepDone=true;
  changePlayer(1,besttype);
  std::cerr << "besttype: "<<besttype;
  return;
#endif  
  /* first create an array that we can sort later on */

  int max;
  PlayerType pt[PlayerParam::player_types];
  for(int t=1;t<PlayerParam::player_types;t++) {
    pt[t-1]=fld.plType[t];
  }
  

#if 0
  /* based on average stamina demand */
  for(int i=0;i<PlayerParam::player_types;i++) 
    for(int j=i+1;j<PlayerParam::player_types;j++) 
      if(pt[i].stamina_inc_max<34 ||
	 pt[i].real_player_speed_max<pt[j].real_player_speed_max) {
	PlayerType dum=pt[i];pt[i]=pt[j];pt[j]=dum;
      }
  for(max=0;max<4;max++) {
    if(pt[max].real_player_speed_max>=pt[max].player_speed_max-.02) continue;
    cerr << "\nWARNING: Only "<<max<<" player types reach speed "<<pt[max].player_speed_max<<"!";
    break;
  }
  for(int i=0;i<max;i++)
    for(int j=0;j<max;j++)
      if(pt[i].stamina_10m<pt[j].stamina_10m) {
	PlayerType dum=pt[i];pt[i]=pt[j];pt[j]=dum;
      }
  if(max<4) {
    for(int i=max;i<4;i++)
      for(int j=i+1;j<4;j++)
	if(pt[i].stamina_10m<pt[j].stamina_10m) {
	  PlayerType dum=pt[i];pt[i]=pt[j];pt[j]=dum;
	}
  }

  /* OK, best stamina_10m values for midfielders... */
  int best_10m=pt[0].id;

  /* Same for stamina_20m */
  for(int i=0;i<max;i++)
    for(int j=0;j<max;j++)
      if(pt[i].stamina_20m<pt[j].stamina_20m) {
	PlayerType dum=pt[i];pt[i]=pt[j];pt[j]=dum;
      }
  if(max<4) {
    for(int i=max;i<4;i++)
      for(int j=i+1;j<4;j++)
	if(pt[i].stamina_20m<pt[j].stamina_20m) {
	  PlayerType dum=pt[i];pt[i]=pt[j];pt[j]=dum;
	}
  }
  int c=0;
  int best_20m[4];
  for(int t=0;t<4;t++) {
    if(pt[t].id==best_10m) continue;
    best_20m[c++]=pt[t].id;
  }

  cerr << "\nbest_10m   = "<<best_10m
       << "\nbest_20m[0]= "<<best_20m[0]
       << "\nbest_20m[1]= "<<best_20m[1]
       << "\nbest_20m[2]= "<<best_20m[2];
  
  for(int t=0;t<7;t++) {
    cerr << "\nid: "<<pt[t].id<<" real_player_speed_max: "<<pt[t].real_player_speed_max;
  }
#else
#define MIN_STAMINA_INC_MAX 36
  /* based on stamina_inc_max */
  int maxpt=PlayerParam::player_types-1;
  for(int i=0;i<maxpt;i++) {
    if(pt[i].stamina_inc_max<MIN_STAMINA_INC_MAX) {
      for(int j=i+1;j<PlayerParam::player_types-1;j++)
	pt[j-1]=pt[j];
      maxpt--;i--;
    }
  }
  if(maxpt<4) cerr << "\nWARNING: Only "<<maxpt<<" player types have stamina_inc_max>"
		   <<MIN_STAMINA_INC_MAX;
	
  for(int i=0;i<maxpt;i++) 
    for(int j=i+1;j<maxpt;j++) 
      if(pt[i].real_player_speed_max<pt[j].real_player_speed_max) {
	PlayerType dum=pt[i];pt[i]=pt[j];pt[j]=dum;
      }
  for(max=0;max<4;max++) {
    if(pt[max].real_player_speed_max>=pt[max].player_speed_max-.02) continue;
    cerr << "\nWARNING: Only "<<max<<" player types reach speed "<<pt[max].player_speed_max<<"!";
    break;
  }

  cerr << "\nBest hetero players:";
  for(int i=0;i<maxpt;i++) {
    cerr << "\nType "<<pt[i].id<<": speed_max="<<pt[i].real_player_speed_max<<", stamina_inc_max="
	 <<pt[i].stamina_inc_max;
  }
  emerg_change=false;
  if(maxpt>=3) {
    changePlayer(9,pt[1].id);
    changePlayer(10,pt[0].id);
    changePlayer(11,pt[0].id);
    
    changePlayer(6,pt[1].id);
    changePlayer(7,pt[0].id);
    changePlayer(8,pt[1].id);

    if(pt[2].stamina_inc_max>50) {
      changePlayer(2,pt[2].id);
      changePlayer(4,pt[2].id);
      changePlayer(5,pt[2].id);
    }
  } else if(maxpt==2) {
    changePlayer(9,pt[0].id);
    changePlayer(10,pt[0].id);
    changePlayer(11,pt[1].id);
    
    changePlayer(6,pt[0].id);
    changePlayer(7,pt[1].id);
    changePlayer(8,pt[1].id);
  } else if(maxpt==1) {
    changePlayer(6,pt[0].id);
    changePlayer(9,pt[0].id);
    changePlayer(10,pt[0].id);
  }
  else if(maxpt==0) {
    for(int t=1;t<PlayerParam::player_types;t++) {
      pt[t-1]=fld.plType[t];
    }
    for(int i=0;i<PlayerParam::player_types;i++) 
      for(int j=i+1;j<PlayerParam::player_types;j++) 
	if(pt[i].real_player_speed_max<pt[j].real_player_speed_max) {
	  PlayerType dum=pt[i];pt[i]=pt[j];pt[j]=dum;
	}
    for(int t=0;t<PlayerParam::player_types;t++)
      if(pt[t].real_player_speed_max>1.15) {
	emerg_type=pt[t].id;
	emerg_change=true;
	break;
      }
    cerr << "\nWARNING: No good player types, will change after half-time if necessary!";
  }
    
#endif
  //changePlayer(1,best_10m);
  
  //changePlayer(2,best_20m[1]);
  //changePlayer(3,best_20m[1]);
  //changePlayer(4,best_20m[2]);
  //changePlayer(5,best_20m[1]);
  
  prepDone=true;
}
