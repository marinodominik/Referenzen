/* Author: Manuel "Sputnick" Nickschas
 *
 * See messages.h for description.
 *
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "coach.h"
#include "messages.h"
#include "param.h"
#include "logger.h"
#include "str2val.h"
#include "wmtools.h"

using namespace MSG;

const char *MSG::msgString[]={
  "nil",
  "init","say","ear","eye","look","change_player_type","team_names",
  "change_mode","move","check_ball","start","recover",
  "server_param","player_param","player_type","see_global","hear", "situation", "reseed_hetro",
  "clang","done","think"};

AbstractMessage *MSG::msg[MSG_NUM];
char MSG::msgBuffer[MAX_MSG_SIZE];

void MSG::initMsg() {

//  cout << "\nInitialising message parser...";
  
  msg[MSG_INIT] = new MsgInit();
  msg[MSG_SAY] = new MsgSay();
  msg[MSG_EAR] = new MsgEar();
  msg[MSG_EYE] = new MsgEye();
  msg[MSG_LOOK] = new MsgLook();
  msg[MSG_CHANGE_PLAYER_TYPE] = new MsgChangePlayerType();
  msg[MSG_TEAM_NAMES] = new MsgTeamNames();
  msg[MSG_CHANGE_MODE] = new MsgChangeMode();
  msg[MSG_MOVE] = new MsgMove();
  msg[MSG_CHECK_BALL] = new MsgCheckBall();
  msg[MSG_START] = new MsgStart();
  msg[MSG_RECOVER] = new MsgRecover();
  msg[MSG_SERVER_PARAM] = new MsgServerParam();
  msg[MSG_PLAYER_PARAM] = new MsgPlayerParam();
  msg[MSG_PLAYER_TYPE] = new MsgPlayerType();
  msg[MSG_SEE_GLOBAL] = new MsgSeeGlobal();
  msg[MSG_HEAR] = new MsgHear();
  msg[MSG_SETSIT] = new MsgSetSit();
  msg[MSG_RESEED_HETRO] = new MsgReseedHetro();
  msg[MSG_CLANG] = new MsgClang();
  msg[MSG_DONE] = new MsgDone();
  msg[MSG_THINK] = new MsgThink();
  msg[MSG_USER] = new MsgUser();
}

void MSG::destructMsg() {
//  cout << "\nCleaning up message parser...";
  for(int i=0;i<MSG_NUM;i++) delete msg[i];
}

int MSG::parseMsg(const char *str) {
  const char *strorig = str;
  int msgtype = MSG_ERR;
  bool ackflg = false;

  if(str[0]!='(') {
    LOG_ERR(0,<< "PARSER: Got invalid message: " << strorig);
    return MSG_ERR;
  }

  str++;
  
  if(!strncmp(str,"ok ",3)) {
    ackflg=true;str+=3;
  }

  if(!strncmp(str,"warning ",8)) {
    LOG_MSG(0,<< "WARNING from server: " << strorig);
    return MSG_ERR;
  }

  if(!strncmp(str,"error ",6)) {
    LOG_MSG(0,<< "ERROR from server: " << strorig);
    return MSG_ERR;
  }

  for(int i=1;i<MSG_NUM;i++) {
    if(!strncmp(str,msgString[i],strlen(msgString[i]))) {
      msgtype = i;break;
    }
  }
  if(msgtype==MSG_ERR) {
    LOG_MSG(0, << "PARSER: Got unknown message from server: " << strorig);
    return MSG_ERR;
  }

  if(!ackflg) {
    if(msg[msgtype]->parseMsg(strorig)) {
      msg[msgtype]->lastRcvd=fld.getTime();
      return msgtype;
    }
  } else {
    if(msg[msgtype]->parseMsgAck(strorig)) {
      if(msg[msgtype]->lastSent!=fld.getTime()) {
	LOG_MSG(0,<< "PARSER: ACK for message type " << msgString[msgtype]
		<< " not in same cycle!");
      }
      msg[msgtype]->lastAck=fld.getTime();
      return -msgtype;
    }
  }
  LOG_MSG(0,<< "PARSER: Could not parse message: " << strorig);
  return MSG_ERR;
}

bool MSG::sendMsg(int type) {
  return msg[type]->sendMsg();
}

bool MSG::sendMsg(int type,int p0,const char *str) {
  return msg[type]->sendMsg(p0,str);
}

bool MSG::sendMsg(int type,int p0,string str) {
  return msg[type]->sendMsg(p0,str);
}

bool MSG::sendMsg(int type,string str) {
  return msg[type]->sendMsg(str);
}

bool MSG::sendMsg(int type,const char *str,const char *str2) {
  return msg[type]->sendMsg(str,str2);
}

bool MSG::sendMsg(int type,int p0) {
  return msg[type]->sendMsg(p0);
}

bool MSG::sendMsg(int type,int p0, int p1) {
  return msg[type]->sendMsg(p0,p1);
}

bool MSG::sendMsg(int type,const char *p0, double p1,double p2) {
  return msg[type]->sendMsg(p0,p1,p2);
}

bool MSG::sendMsg(int type,const char *p0, double p1,double p2,double p3) {
  return msg[type]->sendMsg(p0,p1,p2,p3);
}

bool MSG::sendMsg(int type,const char *p0, double p1,double p2,double p3,double p4,double p5) {
  return msg[type]->sendMsg(p0,p1,p2,p3,p4,p5);
}

bool MSG::sendMsg(int type,const char *string, int p0, int p1) {
  return msg[type]->sendMsg(string,p0,p1);
}

bool MSG::recvMsg(char *buf) {
  return AbstractMessage::getMsgFromServer(buf);
}

int MSG::recvAndParseMsg() {
  //static char buf[MAX_MSG_SIZE];
  //cout << "\n" << msgBuffer << flush; 
  if(!recvMsg(msgBuffer)) return MSG_ERR;
  //LOG_MSG(0,<<"### TG ### "<<msgBuffer);
  //if(!strskip(msgBuffer,"(see_global")) cout << "\n" << msgBuffer << flush;
  return parseMsg(msgBuffer);
}


/*******************************************************************************************
 * Message Implementations
 ******************************************************************************************/

AbstractMessage::AbstractMessage() {
  lastSent=lastRcvd=-1;
  lastAck=-1;
}

AbstractMessage::~AbstractMessage() {

}

bool AbstractMessage::abstractCall(int i=-1) {
  cerr << "\nERROR: Function #"<<i<<" of AbstractMessage has been called!";
  return false;
}

//--------------------------------------------------------------------------
bool AbstractMessage::sendMsgToServer(const char *str) {
  RUN::sock.send_msg(str,strlen(str)+1);
  if(lastAck!=-1 && lastAck!=lastSent) {
    LOG_MSG(0,<<"sendMsgToServer: Did not get ACK for last message! Now sending:\n"<<str);
  }
  lastSent = fld.getTime();
  return true;
}

bool AbstractMessage::getMsgFromServer(char *buf) {
  int num_bytes;
  if(!RUN::sock.recv_msg(buf,num_bytes,true)) return false;
  buf[num_bytes]='\000';
  return true;
}
//--------------------------------------------------------------------------

bool MsgInit::parseMsg(const char *str) {
  if(onlineCoach) {
    char s;
    if(sscanf(str,"(init %c ok)",&s)) {
      RUN::side='l'== s?RUN::side_LEFT:RUN::side_RIGHT;
      RUN::serverAlive = true;
      return true;
    }
  } else {
    RUN::serverAlive = true;//RUN::side = RUN::side_NONE;
    return true;
  }
  return false;
}

bool MsgInit::sendMsg(const char *version,const char *tname=NULL) {
  if(onlineCoach) sprintf(msgBuffer,"(init %s (version %s))",tname,version);
  else sprintf(msgBuffer,"(init (version %s))",version);
  sendMsgToServer(msgBuffer);
  return true;
}
//--------------------------------------------------------------------------

bool MsgSay::sendMsg(int type,const char *str) {
  if(fld.getPM()==PM_play_on && fld.sayInfo.lastInPlayOn[type]>=0 &&
     fld.getTime()-fld.sayInfo.lastInPlayOn[type]
     < ServerParam::clang_win_size) {
    LOG_MSG(0,<<"ERROR: Cannot send message to players now!");
    return false;
  }
  if(!fld.sayInfo.ack[type]) {
    LOG_MSG(0,<<"WARNING: Got no ACK for last say message of that type!");
  }
  fld.sayInfo.last[type]=fld.getTime();
  if(fld.getPM()==PM_play_on) fld.sayInfo.lastInPlayOn[type]=fld.getTime();
  fld.sayInfo.ack[type]=false;
  if (type == MSGT_FREEFORM)
    sprintf(msgBuffer,"(say (%s %s))",MSG_TYPE_STR[type],str);
  else
    sprintf(msgBuffer,"(say (%s (%s)))",MSG_TYPE_STR[type],str);
  LOG_MSG(1,<<"Sending message to players: "<<msgBuffer);
  //std::cerr << "\nSending message to players: \""<<msgBuffer<<"\"";
  sendMsgToServer(msgBuffer);
  return true;
}

bool MsgSay::sendMsg(int type,string str) {
  if(fld.getPM()==PM_play_on && fld.sayInfo.lastInPlayOn[type]>=0 &&
     fld.getTime()-fld.sayInfo.lastInPlayOn[type]
     < ServerParam::clang_win_size) {
    LOG_MSG(0,<<"ERROR: Cannot send message to players now!");
    return false;
  }
  if(!fld.sayInfo.ack[type]) {
    LOG_MSG(0,<<"WARNING: Got no ACK for last say message of that type!");
  }
  fld.sayInfo.last[type]=fld.getTime();
  if(fld.getPM()==PM_play_on) fld.sayInfo.lastInPlayOn[type]=fld.getTime();
  fld.sayInfo.ack[type]=false;
  if (type == MSGT_FREEFORM)
    sprintf(msgBuffer,"(say (%s %s))",MSG_TYPE_STR[type],str.c_str());
  else
    sprintf(msgBuffer,"(say (%s (%s)))",MSG_TYPE_STR[type],str.c_str());
  LOG_MSG(1,<<"Sending message to players: "<<msgBuffer);
  //std::cerr << "\nSending message to players: \""<<msgBuffer<<"\"";
  sendMsgToServer(msgBuffer);
  return true;
}

bool MsgSay::parseMsgAck(const char *) {
  int oldest=-1,oldtime=INT_MAX;
  for(int i=0;i<MAX_SAY_TYPES;i++) {
    if(fld.sayInfo.ack[i]) continue;
    if(fld.sayInfo.last[i]<oldtime) {
      oldest=i;oldtime=fld.sayInfo.last[i];
    }
  }
  if(oldest==-1) {
    LOG_MSG(0,<<"WARNING: Got senseless ACK for say message!");
    return true;
  }
  fld.sayInfo.ack[oldest]=true;
  return true;
}


//--------------------------------------------------------------------------

bool MsgEye::sendMsg(int mode) {
  lastMode=mode;
  if(mode) sprintf(msgBuffer,"(eye on)");
  else sprintf(msgBuffer,"(eye off)");
  return sendMsgToServer(msgBuffer);
}

bool MsgEye::parseMsgAck(const char *str) {
  int mode=0;
  str+=8;
  if(strskip(str,"on")) mode=1;
  if(mode!=lastMode) {
    LOG_MSG(0,<<"Warning: Got ACK for wrong eye mode!");return false;
  }
  return true;
}
//--------------------------------------------------------------------------

bool MsgEar::sendMsg(int mode) {
  lastMode=mode;
  if(mode) sprintf(msgBuffer,"(ear on)");
  else sprintf(msgBuffer,"(ear off)");
  return sendMsgToServer(msgBuffer);
}

bool MsgEar::parseMsgAck(const char *str) {
  int mode=0;
  str+=8;
  if(strskip(str,"on")) mode=1;
  if(mode!=lastMode) {
    LOG_MSG(0,<<"Warning: Got ACK for wrong ear mode!");return false;
  }
  return true;
}

//--------------------------------------------------------------------------

MsgChangePlayerType::MsgChangePlayerType() {
  subsCnt=0;
  
}

bool MsgChangePlayerType::sendMsg(int unum, int type) {
  if(Options::isTrainer) {
    return sendMsg(Options::teamName,unum,type);
  }
  if(fld.getPM()==PM_play_on) {
    LOG_MSG(0,<<"Error: Cannot substitute player while in play_on mode!");
    return false;
  }
  /* Ueberpruefung sollte das Modul erledigen! */
  //if(onlineCoach && subsDone>=3) {
  //  LOG_MSG(0,<<"Error: Cannot substitute more than 3 players in online coach mode!");
  //  return false;
  //}
  //subsDone++;
  if(subsCnt>=MAX_SUBS_PER_GAME && onlineCoach) {
    LOG_MSG(0,<<"ERROR: Too many substitutions requested within this game, ignoring!");
    return false;
  }
  subsDone[subsCnt].unum=unum;
  subsDone[subsCnt].type=type;
  subsDone[subsCnt++].done=0;
  
  sprintf(msgBuffer,"(change_player_type %d %d)",unum,type);
  //cout << "\nSending: " << msgBuffer << flush;
  return sendMsgToServer(msgBuffer);
}

bool MsgChangePlayerType::sendMsg(const char *string,int unum, int type) {
  if(fld.getPM()==PM_play_on) {
    LOG_MSG(0,<<"Error: Cannot substitute player while in play_on mode!");
    return false;
  }
  /* Ueberpruefung sollte das Modul erledigen! */
  //if(onlineCoach && subsDone>=3) {
  //  LOG_MSG(0,<<"Error: Cannot substitute more than 3 players in online coach mode!");
  //  return false;
  //}
  //subsDone++;
  if(subsCnt>=MAX_SUBS_PER_GAME && onlineCoach) {
    LOG_MSG(0,<<"ERROR: Too many substitutions requested within this game, ignoring!");
    return false;
  }
  subsDone[subsCnt].unum=unum;
  subsDone[subsCnt].type=type;
  subsDone[subsCnt++].done=0;
  
  sprintf(msgBuffer,"(change_player_type %s %d %d)",string,unum,type);
  //cout << "\nSending: " << msgBuffer << flush;
  return sendMsgToServer(msgBuffer);
}

bool MsgChangePlayerType::parseMsgAck(const char *str) {
  int unum,type;
  strskip(str,"(ok change_player_type ",str);
  if(!onlineCoach) {
    strskip(str,Options::teamName,str);
  }
  //strfind(str,' ',str);str++;
  if(!(str2val(str,unum,str) && str2val(str,type,str)) && strfind(str,')',str)) {
    LOG_MSG(0,<<"ERROR: Unknown format for change_player_type ACK message!");
    return false;
  }
  fld.myTeam[unum-1].type=type;
  RUN::announcePlayerChange(true,unum,type);
  for(int i=1;i<=subsCnt;i++) {
    if(subsDone[subsCnt-i].done!=0) continue;
    if(subsDone[subsCnt-i].type==type && subsDone[subsCnt-i].unum==unum) {
      LOG_MSG(0,<<"Player change #"<<subsCnt-i+1<<" succeeded! (player "<<unum<<", type "<<type<<")");
      subsDone[subsCnt-i].time=fld.getTime();
      subsDone[subsCnt-i].done=1;
      return true;
    }
  }
  LOG_MSG(0,<<"ERROR: Got ACK for substitution that I haven't requested...");
  return false;
}

bool MsgChangePlayerType::parseMsg(const char *str) {
  int nr;
  //cout << "\n" << str;
  if(sscanf(str,"(change_player_type %d)",&nr)) {
    RUN::announcePlayerChange(false,nr,-1);
    return true;
  }
  return false;
}

//--------------------------------------------------------------------------

bool MsgTeamNames::sendMsg() {
  return sendMsgToServer("(team_names)");
}

/**
 * Each class for a message type contains a parseMsgAck method.
 * This method is used to scan for the team names. It usually
 * expects a string like "(ok team_names (team l xxx) (team r yyy) )".
 * In case that no team has connected to the server, yet, the string
 * to be parsed should look like "(ok team_names)".
 * @param str is the string to be parsed
 * @return In its current implementation this method always returns
 * true as it is unprobable that syntactical errors are produced by
 * the soccer server.
 */
bool MsgTeamNames::parseMsgAck(const char *str) {
  const char *pnt,*pnt2;
  strskip(str,"(ok team_names ",str); 
  if(strskip(str,"(team l ",pnt)) 
  {
    strfind(pnt,')',pnt2);
    //cout << "==" << pnt << "=="<<pnt2<<"=="<<pnt2-pnt;
    //cout << "**"<<pnt<<"**";
    if (RUN::side==RUN::side_LEFT) 
    {
      strncpy(fld.myTeamName,pnt,pnt2-pnt);
      fld.myTeamName[pnt2-pnt]='\000';
    } 
    else 
    {
      strncpy(fld.hisTeamName,pnt,pnt2-pnt);
      fld.hisTeamName[pnt2-pnt]='\000';
    }
    str=pnt2+1;
  }
  if(strskip(str,"(team r ",pnt)) 
  {
    strfind(pnt,')',pnt2);
    if (RUN::side==RUN::side_RIGHT) 
    {
      strncpy(fld.myTeamName,pnt,pnt2-pnt);
      fld.myTeamName[pnt2-pnt]='\000';
    } 
    else 
    {
      strncpy(fld.hisTeamName,pnt,pnt2-pnt);
      fld.hisTeamName[pnt2-pnt]='\000';
    }
  }
  return true;
}

//--------------------------------------------------------------------------



//--------------------------------------------------------------------------

//--------------------------------------------------------------------------
bool MsgServerParam::parseMsg(const char *str) {
  return ServerParam::parseMsg(str);
}

bool MsgPlayerParam::parseMsg(const char *str) {
  return PlayerParam::parseMsg(str);
}

bool MsgPlayerType::parseMsg(const char *str) {
  const char *strorig = str;
  strskip(str,"(player_type (id ", str);
  int i;
  bool flg = str2val(str, i);
  if(flg){
    fld.plType[i].init();
    if(!fld.plType[i].parseMsg(strorig)) {
      return false;
    }
  } else{
    return false;
  }
  return true;
  //andi  LOG_MSG(0,<< "PARSER: Got message of type player_type without good reason!");
  //andi return false;
}
//--------------------------------------------------------------------------

bool MsgSeeGlobal::parseMsg(const char *str) {

  //fld.timeOfLastSee=fld.curTime;
  int last=fld.getTime();
  bool flg=strskip(str,"(see_global ",str) && parseSeeString(str);
  if(fld.getTime()-last>1) {
    LOG_ERR(0,<<"WARNING: Missed a see_global message! Old: "<< last << ", now: " << fld.getTime());
  }
  return flg;
    
  
}

bool MsgSeeGlobal::parseSeeString(const char *str) {
  long ldum;const char *sdum;
  double px,py,vx,vy;
  Angle ba,na,pt;
  int nr;
  //cout << "\n" << str << "\n";
  bool flg,ownplayer,goalie,pt_flg;
  
  // objects I don't get here should be set to dead:
  for(int i=0;i<11;i++) {
    fld.myTeam[i].alive=false;fld.hisTeam[i].alive=false;
  }
  fld.ball.alive=false;
  
  if (!str2val(str,ldum,str)) return false;
  fld.setTime(ldum);

  for(;;) {
    flg=false;
    pt_flg=false;
    if(strskip(str,"((g l)",sdum) || strskip(str,"((g r)",sdum)) {
      strfind(sdum,')',str);str++;flg=true;
    }
    if(strskip(str,"((b)",sdum)) {
      str=sdum;
      if (!(str2val(str,px,str) && str2val(str,py,str) &&
	    str2val(str,vx,str) && str2val(str,vy,str) &&
	    strskip(str,")",str))) return false;
      fld.ball.setPos(px,-py);fld.ball.setVel(vx,-vy);
      fld.ball.alive=true;flg=true;
    }
    if(strskip(str,"((p \"",sdum)) {
      str=sdum;
      strfind(str,'\"',sdum);
      if(strncmp(str,Options::teamName,sdum-str)==0) ownplayer=true;
      else ownplayer=false;
      str2val(sdum+1,nr,str);
      if(strskip(str,"goalie",sdum)) {
        goalie=true;str=sdum;
      } else {goalie=false;}
          strskip(str,")",str);
          if(!(str2val(str,px,str) && str2val(str,py,str) && str2val(str,vx,str) &&
           str2val(str,vy,str) && str2val(str,ba,str) && str2val(str,na,str))) return false;
          if ( str2val(str,pt,sdum) ) { // pointto data received
           pt_flg = true;
           pt=WMTOOLS::conv_server_angle_to_angle_0_2Pi(pt);
           str=sdum;
      }
      strfind(str,')',str); // skip tackle data
      strskip(str,")",str);
      ba=WMTOOLS::conv_server_angle_to_angle_0_2Pi(ba);
      na=WMTOOLS::conv_server_angle_to_angle_0_2Pi(na);
      nr--;
      if(ownplayer) {
        fld.myTeam[nr].setPos(px,-py);fld.myTeam[nr].setVel(vx,-vy);
        fld.myTeam[nr].bodyAng=(ANGLE)ba;fld.myTeam[nr].neckAng=(ANGLE)na;
        fld.myTeam[nr].goalie=goalie;fld.myTeam[nr].number=nr+1;
        fld.myTeam[nr].alive=true;
        if (pt_flg) {
            fld.myTeam[nr].pointtoFlag=pt_flg;
            fld.myTeam[nr].pointtoDir=(ANGLE)pt;
            // JTS10 extract stamina capacity info from pointto
            fld.myTeam[nr].staminaCapacityBound 
              =   (0.9*fld.myTeam[nr].staminaCapacityBound)
                + (0.1*WMTOOLS::get_stamina_capacity_from_pointto(fld.myTeam[nr].pointtoDir));
            LOG_MSG(0,<<"MsgSeeGlobal: Player "<<nr<<", is pointing: "<<pt_flg
              <<", into dir: "<<pt<<", encodes stamina: "
              <<WMTOOLS::get_stamina_capacity_from_pointto(fld.myTeam[nr].pointtoDir)
              <<", estimated stamina reserve: "
              <<fld.myTeam[nr].staminaCapacityBound);
        }
      } else {
        fld.hisTeam[nr].setPos(px,-py);fld.hisTeam[nr].setVel(vx,-vy);
        fld.hisTeam[nr].bodyAng=(ANGLE)ba;fld.hisTeam[nr].neckAng=(ANGLE)na;
        fld.hisTeam[nr].goalie=goalie;fld.hisTeam[nr].number=nr+1;
        fld.hisTeam[nr].alive=true;
        if (pt_flg) {
            fld.hisTeam[nr].pointtoFlag=pt_flg;
            fld.hisTeam[nr].pointtoDir=(ANGLE)pt;
        }
      }
      flg=true;
    }
    if(!flg) break;
  }
  fld.visState();

  return strskip(str,")",sdum);
}

//--------------------------------------------------------------------------

bool MsgHear::parseMsg(const char *str) {
  long ldum;const char *sdum;
  //cout << "\n" << str;
  if (!strskip(str,"(hear ",str) || !str2val(str,ldum,str)) return false;
  if(ldum!=fld.getTime()) {
    //LOG_ERR(0,<<"WARNING: Wrong cycle number for hear message - missed see_global?");
    //LOG_ERR(0,<<"WARNING: Adjusted current time to value of hear message!");
    fld.setTime(ldum);
  }
  if(strskip(str,"referee ",sdum)) 
  {
    str=sdum;
    for(int i=0;i<RM_MAX;i++) 
    {
      if(strskip(str,playmodeStrings[i],sdum)) 
      {
        fld.setPM(i);
        return true;
      }
    }
    LOG_MSG(0,<<"PARSER: Got unknown hear message from referee!");
    return false;
  } 
  else 
  {
    //LOG_MSG(1,<<"PARSER: hear message from players will not be parsed.");
    //LOG_MSG(1,<<"PARSER: player hear: "<<str);
    return true;
  }
}

bool MsgChangeMode::sendMsg(int param){
  switch (param){
  case PM_before_kick_off:
    sprintf(msgBuffer,"(change_mode before_kickoff)");
    break;
  case PM_kick_off_l:
    sprintf(msgBuffer,"(change_mode kick_off_l)");
    break;
  case PM_kick_off_r:
    sprintf(msgBuffer,"(change_mode kick_off_r)");
    break;
  case PM_kick_in_l:
    sprintf(msgBuffer,"(change_mode kick_in_l)");
    break;
  case PM_kick_in_r:
    sprintf(msgBuffer,"(change_mode kick_in_r)");
    break;
  case PM_corner_kick_l:
    sprintf(msgBuffer,"(change_mode corner_kick_l)");
    break;
  case PM_corner_kick_r:
    sprintf(msgBuffer,"(change_mode corner_kick_r)");
    break;
  case PM_goal_kick_l:
    sprintf(msgBuffer,"(change_mode goal_kick_l)");
    break;
  case PM_goal_kick_r:
    sprintf(msgBuffer,"(change_mode goal_kick_r)");
    break;
  case PM_free_kick_l:
    sprintf(msgBuffer,"(change_mode free_kick_l)");
    break;
  case PM_free_kick_r:
    sprintf(msgBuffer,"(change_mode free_kick_r)");
    break;
  case PM_offside_l:
    sprintf(msgBuffer,"(change_mode offside_l)"); // don't know if this will work
    break;
  case PM_offside_r:
    sprintf(msgBuffer,"(change_mode offside_r)"); // don't know if this will work
    break;
  case PM_indirect_free_kick_l:
    sprintf(msgBuffer,"(change_mode indirect_free_kick_l)");
    break;
  case PM_indirect_free_kick_r:
    sprintf(msgBuffer,"(change_mode indirect_free_kick_r)");
    break;
  case RM_foul_l:
    sprintf(msgBuffer,"(change_mode penalty_kick_l)");
    break;
  case RM_foul_r:
    sprintf(msgBuffer,"(change_mode penalty_kick_r)");
    break;
  case RM_goal_l:
    sprintf(msgBuffer,"(change_mode goal_l)");
    break;
  case RM_goal_r:
    sprintf(msgBuffer,"(change_mode goal_r)");
    break;
  case RM_drop_ball:
    sprintf(msgBuffer,"(change_mode drop_ball)");
    break;
  case RM_half_time:
    sprintf(msgBuffer,"(change_mode first_half_over)");
    break;
    //  case RM_pause:
    //    sprintf(msgBuffer,"(change_mode pause)");
    //    break;
    //  case RM_human_judge:
    //    sprintf(msgBuffer,"(change_mode human_judge)");
    //    break;
  case PM_play_on:
    sprintf(msgBuffer,"(change_mode play_on)");
    break;
    //  case PM_half_time:
    //    sprintf(msgBuffer,"(change_mode before_kickoff)");
    //    break;
  case RM_time_up:
    sprintf(msgBuffer,"(change_mode time_over)");
    break;
  default:
    sprintf(msgBuffer,"(change_mode time_over)");
    break;
  }
  sendMsgToServer(msgBuffer);
  return true;
}

bool MsgChangeMode::parseMsgAck(const char *str){
    return true;
}

/** seems to be for andi's async server only? */
bool MsgSetSit::sendMsg(const char* string,const char *str2){
  sprintf(msgBuffer,"(situation (%s))",string);
  sendMsgToServer(msgBuffer);
  return true;
}

/** seems to be for andi's async server only? */
bool MsgSetSit::sendMsg(string string){
  sprintf(msgBuffer,"(situation (%s))",string.c_str());
  sendMsgToServer(msgBuffer);
  return true;
}

bool MsgSetSit::parseMsgAck(const char *str){
    return true;
}

bool MsgReseedHetro::sendMsg(){
  sprintf(msgBuffer,"(reseed_hetro)");
  sendMsgToServer(msgBuffer);
  return true;
}

bool MsgReseedHetro::parseMsgAck(const char *str){
    return true;
}

bool MsgRecover::sendMsg(){
  sprintf(msgBuffer,"(recover)");
  sendMsgToServer(msgBuffer);
  return true;
}

bool MsgRecover::parseMsgAck(const char *str){
    return true;
}

//--------------------------------------------------------------------------
bool MsgMove::sendMsg(const char *obj, double x, double y) {
  sprintf(msgBuffer,"(move (%s) %f %f)",obj,x,y);
  sendMsgToServer(msgBuffer);
  return true;
}

bool MsgMove::sendMsg(const char *obj, double x, double y,double dir) {
  sprintf(msgBuffer,"(move (%s) %f %f %f)",obj,x,y,dir);
  sendMsgToServer(msgBuffer);
  return true;
}

bool MsgMove::sendMsg(const char *obj, double x, double y,double dir,double velx,double vely) {
  sprintf(msgBuffer,"(move (%s) %f %f %f %f %f)",obj,x,y,dir,velx,vely);
  sendMsgToServer(msgBuffer);
  return true;  
}

bool MsgMove::parseMsgAck(const char*) {
  return true;
}

//--------------------------------------------------------------------------

bool MsgClang::parseMsg(const char *str) {
  return true;
}

bool MsgDone::sendMsg() {
  sprintf(msgBuffer,"(done)");
  //cerr << "Time: " << fld.getTime() << " (done)\n";
  return sendMsgToServer(msgBuffer);
}

bool MsgDone::parseMsgAck(const char *str) {
  return true;
}

bool MsgThink::parseMsg(const char *str) {
  if(!strcmp(str,"(think)")) {
    //cerr << "Time: " << fld.getTime() << " (think)\n";
    return true;
  } 
  return false;
}

//--------------------------------------------------------------------------

//--------------------------------------------------------------------------
