#include "intentiontype.h"
#include <stdio.h>


void IntentionType::clone(IntentionType* iT){
  iT->time = time;
  iT->type = type;
  iT->p1 = p1;
  iT->p2 = p2;
  iT->p3 = p3;
  iT->p4 = p4;
  iT->p5 = p5;
}


void IntentionType::toString(char* str) {
  const char *str3=getString(type);
  if(!str3) {
    sprintf(str,"NO VALID INTENTION TYPE SET");
  } else {
    sprintf(str,"%s(time %d): %.2f %.2f %.2f %.2f %.2f", str3, time, p1, p2, p3, p4, p5);
  }
}

const char* IntentionType::getString(int type) {
  switch (type) {
  case DECISION_TYPE_NONE: return "DECISION_TYPE_NONE"; break;
  case DECISION_TYPE_SCORE: return "DECISION_TYPE_SCORE"; break;
  case DECISION_TYPE_PASS: return "DECISION_TYPE_PASS"; break;
  case DECISION_TYPE_LAUFPASS: return "DECISION_TYPE_LAUFPASS"; break;
  case DECISION_TYPE_DRIBBLE: return "DECISION_TYPE_DRIBBLE"; break;
  case DECISION_TYPE_STOPBALL: return "DECISION_TYPE_STOPBALL"; break;
  case DECISION_TYPE_FACE_BALL: return "DECISION_TYPE_FACE_BALL"; break;
  case DECISION_TYPE_EXPECT_PASS: return "DECISION_TYPE_EXPECT_PASS"; break;
  case DECISION_TYPE_LEAVE_OFFSIDE: return "DECISION_TYPE_LEAVE_OFFSIDE"; break;
  case DECISION_TYPE_RUNTO: return "DECISION_TYPE_RUNTO"; break;
  case DECISION_TYPE_INTERCEPTBALL: return "DECISION_TYPE_INTERCEPTBALL"; break;
  case DECISION_TYPE_INTERCEPTSLOWBALL: return "DECISION_TYPE_INTERCEPTSLOWBALL"; break;
  case DECISION_TYPE_CLEARANCE: return "DECISION_TYPE_INTERCEPTSLOWBALL"; break;

  case DECISION_TYPE_OFFSIDETRAP: return "DECISION_TYPE_INTERCEPTSLOWBALL"; break;
  case DECISION_TYPE_DEFEND: return "DECISION_TYPE_INTERCEPTSLOWBALL"; break;
  case DECISION_TYPE_BLOCKBALLHOLDER: return "DECISION_TYPE_INTERCEPTSLOWBALL"; break;
  case DECISION_TYPE_ATTACKBALLHOLDER: return "DECISION_TYPE_INTERCEPTSLOWBALL"; break;
  case DECISION_TYPE_WAITANDSEE: return "DECISION_TYPE_INTERCEPTSLOWBALL"; break;
  case DECISION_TYPE_SEARCH_BALL: return "DECISION_TYPE_INTERCEPTSLOWBALL"; break;
  case DECISION_TYPE_DEFEND_FACE_BALL: return "DECISION_TYPE_INTERCEPTSLOWBALL"; break;
  case DECISION_TYPE_ATTACK: return "DECISION_TYPE_ATTACK"; break;
  case DECISION_TYPE_OFFER_FOR_PASS: return "DECISION_TYPE_OFFER_FOR_PASS"; break;
  case DECISION_TYPE_GOALIECLEARANCE: return "DECISION_TYPE_GOALIECLEARANCE"; break;
  case DECISION_TYPE_LOOKFORGOALIE: return "DECISION_TYPE_LOOKFORGOALIE"; break;
  case DECISION_TYPE_INTROUBLE: return "DECISION_TYPE_INTROUBLE"; break;
  case DECISION_TYPE_IMMEDIATE_SELFPASS: return "DECISION_TYPE_IMMEDIATE_SELFPASS"; break;
  default: return NULL; break;
  }


}
