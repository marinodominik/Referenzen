#ifndef _INTENTION_TYPE_H_
#define _INTENTION_TYPE_H_


#define DECISION_TYPE_NONE 0
#define DECISION_TYPE_SCORE 1
#define DECISION_TYPE_PASS 2
#define DECISION_TYPE_DRIBBLE 3
#define DECISION_TYPE_STOPBALL 4
#define DECISION_TYPE_FACE_BALL 5
#define DECISION_TYPE_EXPECT_PASS 6
#define DECISION_TYPE_LEAVE_OFFSIDE 7
#define DECISION_TYPE_RUNTO 8
#define DECISION_TYPE_INTERCEPTBALL 9
#define DECISION_TYPE_INTERCEPTSLOWBALL 10
#define DECISION_TYPE_CLEARANCE 11
#define DECISION_TYPE_OFFSIDETRAP 12
#define DECISION_TYPE_DEFEND 13
#define DECISION_TYPE_BLOCKBALLHOLDER 14
#define DECISION_TYPE_ATTACKBALLHOLDER 15
#define DECISION_TYPE_WAITANDSEE 16
#define DECISION_TYPE_SEARCH_BALL 17
#define DECISION_TYPE_DEFEND_FACE_BALL 18
#define DECISION_TYPE_ATTACK 19
#define DECISION_TYPE_OFFER_FOR_PASS 20
#define DECISION_TYPE_GOALIECLEARANCE 21
#define DECISION_TYPE_LOOKFORGOALIE 22
#define DECISION_TYPE_INTROUBLE 23
#define DECISION_TYPE_IMMEDIATE_SELFPASS 24
#define DECISION_TYPE_LAUFPASS 25
#define DECISION_TYPE_KICKNRUSH 26
#define DECISION_TYPE_SELFPASS 27

/** neck intention types
    The intention is valid for the current cycle;
    older intentions will be ignored.
    Some types are not implemented yet and marked with '-'.
    Use set_my_neck_intention() and get_my_neck_intention().
*/

#define NECK_INTENTION_NONE 40                      /* do nothing special */
#define NECK_INTENTION_LOCKNECK 41                  /* lock neck to body angle at 0 degrees */
#define NECK_INTENTION_LOOKTOGOAL 42                /* look to goal */
#define NECK_INTENTION_LOOKTOGOALIE 43              /* look to assumed goalie position */
#define NECK_INTENTION_LOOKTOBALL 44                /* look to assumed ball position */
#define NECK_INTENTION_LOOKTOCLOSESTOPPONENT 45     /* -look to closest opponent */
#define NECK_INTENTION_CHECKOFFSIDE 46              /* -look to opponent that defines the offside line */
#define NECK_INTENTION_LOOKINDIRECTION 47           /* look into direction p1 */
#define NECK_INTENTION_PASSINDIRECTION 48           /* dito, but specify that I want to pass */
#define NECK_INTENTION_SCANFORBALL 49               /* -search the ball */
#define NECK_INTENTION_FACEBALL 50                  /* look to assumed ball position if necessary */
#define NECK_INTENTION_EXPECTPASS 51                /* -look into ball direction if necessary */
#define NECK_INTENTION_BLOCKBALLHOLDER 52           /*  */
#define NECK_INTENTION_DIRECTOPPONENTDEFENSE 53     /* defense is based on direct opponent assignments*/
                                                    /* regular looking to direct opp. is required */

/** view intention types
    I am not sure if this will be needed. However, the necessary methods
    (set_my_view_intention(), get_my_view_intention()) have been
    implemented.
*/

#define VIEW_INTENTION_NONE 0

		
class IntentionType{
 public:
  int type;
  int time;
  float p1,p2,p3,p4,p5;
  void clone(IntentionType* iT);
  void toString(char* str);
  static const char *getString(int type);
};

#endif
