/* Author: Manuel "Sputnick" Nickschas, 12/2001
 *
 * This file contains message parsing and structs.
 * 
 */

#ifndef _MESSAGES_H_
#define _MESSAGES_H_

#include "defs.h"
#include "field.h"
#include "options.h"

class AbstractMessage {
 private:
  bool abstractCall(int nr);
  
 public:  
  long lastSent;
  long lastRcvd;
  long lastAck;

  bool sendMsgToServer(const char *);    
  static bool getMsgFromServer(char *);
  
  virtual bool sendMsg() {return abstractCall(1);}
  virtual bool sendMsg(const char* string,const char *str2=NULL) {return abstractCall(2);}
  virtual bool sendMsg(int param) {return abstractCall(3);}
  virtual bool sendMsg(int param,const char *str) {return abstractCall(8);}
  virtual bool sendMsg(int param,string str) {return abstractCall(8);}
  virtual bool sendMsg(string str) {return abstractCall(8);}
  virtual bool sendMsg(int param,int p2) {return abstractCall(4);}
  virtual bool sendMsg(int param,int p2,int p3) {return abstractCall(5);}
  virtual bool sendMsg(const char*,double p2,double p3) {return abstractCall(8);}
  virtual bool sendMsg(const char*,double p2,double p3,double p4) {return abstractCall(8);}
  virtual bool sendMsg(const char*,double p2,double p3,double p4,double p5,double p6) {return abstractCall(8);}
  virtual bool sendMsg(const char *str, int p2, int p3){return abstractCall(8);}
  virtual bool parseMsg(const char* string=NULL) {return abstractCall(6);}
  virtual bool parseMsgAck(const char* string=NULL) {return abstractCall(7);}

  AbstractMessage();
  virtual ~AbstractMessage();
};

namespace MSG {

  const int MAX_MSG_SIZE = 8192;
  const int MAX_SUBS_PER_GAME = 50;
  
  extern char msgBuffer[MAX_MSG_SIZE];
  
  extern const char *msgString[];
  extern AbstractMessage *msg[];

  /** These are the message types. Negative value means ACK for that type... */
  enum {
    MSG_ERR,
    MSG_INIT,MSG_SAY,MSG_EAR,MSG_EYE,MSG_LOOK,MSG_CHANGE_PLAYER_TYPE,MSG_TEAM_NAMES,
    MSG_CHANGE_MODE,MSG_MOVE,MSG_CHECK_BALL,MSG_START,MSG_RECOVER,
    MSG_SERVER_PARAM,MSG_PLAYER_PARAM,MSG_PLAYER_TYPE,MSG_SEE_GLOBAL,MSG_HEAR,
    MSG_SETSIT,MSG_RESEED_HETRO,MSG_CLANG,MSG_DONE,MSG_THINK,MSG_USER,
    MSG_NUM
  };

  void initMsg();                        /** initialises message structures and parser */
  void destructMsg();                    /** "Destructor" to get rid of the created message objects */
  int parseMsg(const char *string);      /** returns message type */
  bool sendMsg(int type);
  bool sendMsg(int type,int p0,const char *string);
  bool sendMsg(int type,int p0,string string);
  bool sendMsg(int type,string string);
  bool sendMsg(int type,const char *string,const char *string2=NULL); 
  bool sendMsg(int type,int p0);
  bool sendMsg(int type,int p0, int p1);
  bool sendMsg(int type,const char*,double p1,double p2);
  bool sendMsg(int type,const char*,double p1,double p2,double p3);
  bool sendMsg(int type,const char*,double p1,double p2,double p3,double p4,double p5);
  bool sendMsg(int type,const char *string,int p0,int p1);
  bool recvMsg(char *);
  int recvAndParseMsg();
}

/**************************************************************************
 * Message classes - one class for every message type!
 *************************************************************************/

class MsgInit : public AbstractMessage {
  bool parseMsg(const char *str);
  bool sendMsg(const char *version,const char *teamName);
};

class MsgSay: public AbstractMessage {

 public:
        bool sendMsg(int type,const char *string);
        bool sendMsg(int type,string string);
  bool parseMsgAck(const char*);
  
};

class MsgEar : public AbstractMessage {
  int lastMode;
 public:
  MsgEar() {lastMode=-1;}

  bool sendMsg(int mode);
  bool parseMsgAck(const char*);
};

class MsgEye : public AbstractMessage {
  int lastMode;
 public:
  MsgEye() {lastMode=-1;}

  bool sendMsg(int mode);
  bool parseMsgAck(const char*);
};

class MsgLook : public AbstractMessage {
  bool parseObjString(const char *str);
  
};

class MsgChangePlayerType : public AbstractMessage {
 public:
  int subsCnt;
  struct {
    int unum,type,time,done;
  } subsDone[MSG::MAX_SUBS_PER_GAME];
  
  //  int subsToDo;    // number of substitutions in the queue
  //  static int unumQueue[20];
  //  static int typeQueue[20];
  
 public:
  MsgChangePlayerType();
  bool sendMsg(int unum, int type);
  bool sendMsg(const char *string,int unum, int type);
  bool parseMsg(const char*);
  bool parseMsgAck(const char*);
  
};

class MsgTeamNames : public AbstractMessage {

 public:
  bool sendMsg();
  bool parseMsgAck(const char*);
  
};

class MsgChangeMode : public AbstractMessage {
  bool sendMsg(int param);
  bool parseMsgAck(const char *str);

};

class MsgMove : public AbstractMessage {
  bool sendMsg(const char *obj,double x,double y);
  bool sendMsg(const char *obj,double x,double y,double dir);
  bool sendMsg(const char *obj,double x,double y,double dir,double velx,double vely);
  bool parseMsgAck(const char *);
};

class MsgDone: public AbstractMessage {

 public:
  bool sendMsg();
  bool parseMsgAck(const char*);
};

class MsgThink: public AbstractMessage {

 public:
  bool parseMsg(const char *str);
};

class MsgCheckBall : public AbstractMessage {

};

class MsgStart : public AbstractMessage {

};

class MsgRecover : public AbstractMessage {
  bool sendMsg();
  bool parseMsgAck(const char *str);
};

class MsgServerParam : public AbstractMessage {
  bool parseMsg(const char *str);
};

class MsgPlayerParam : public AbstractMessage {
  bool parseMsg(const char *str);
};

class MsgPlayerType : public AbstractMessage {
  bool parseMsg(const char *str);
};

class MsgSeeGlobal : public AbstractMessage {
  bool parseMsg(const char *str);
  bool parseSeeString(const char *str);
};

class MsgHear : public AbstractMessage {
  bool parseMsg(const char *str);
  
};

class MsgUser : public AbstractMessage {

};

class MsgSetSit : public AbstractMessage {
        bool sendMsg(const char* string,const char *str2=NULL);
        bool sendMsg(string string);
  bool parseMsgAck(const char *str);
};

class MsgReseedHetro : public AbstractMessage {
  bool sendMsg();
  bool parseMsgAck(const char *str);
};

class MsgClang : public AbstractMessage {
  bool parseMsg(const char *str);
};

#endif
