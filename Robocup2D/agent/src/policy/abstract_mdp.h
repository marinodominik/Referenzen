#ifndef _ABSTRACT_MDP_H
#define _ABSTRACT_MDP_H

#include "../basics/globaldef.h"

#include "Vector.h"

#define AACTION_TYPE_PASS 0
#define AACTION_TYPE_DRIBBLE 1
#define AACTION_TYPE_SCORE 2
#define AACTION_TYPE_SOLO 3
#define AACTION_TYPE_WAITANDSEE 4
#define AACTION_TYPE_DEADLYPASS 5
#define AACTION_TYPE_SELFPASS 6
#define AACTION_TYPE_GOTO 7
#define AACTION_TYPE_GO2BALL 8
#define AACTION_TYPE_STAY 9
#define AACTION_TYPE_GOHOME 10
#define AACTION_TYPE_IMMEDIATE_SELFPASS 11
#define AACTION_TYPE_LAUFPASS 12
#define AACTION_TYPE_TURN_AND_DASH 13
#define AACTION_TYPE_DRIBBLE_QUICKLY 14
#define AACTION_TYPE_PANIC_KICK 15
#define AACTION_TYPE_BACK_UP 16
#define AACTION_TYPE_KICKNRUSH 17
#define AACTION_TYPE_HOLDTURN 18
#define AACTION_TYPE_FACE_BALL 19
#define AACTION_TYPE_TACKLE 20
#define AACTION_TYPE_TURN_INERTIA 21
#define AACTION_TYPE_GO4PASS 22
#define AACTION_TYPE_NEURO_HASSLE 23

#define SUBTYPE_NONE 0
#define SUBTYPE_PASS 1
#define SUBTYPE_LAUFPASS 2
#define SUBTYPE_PENALTYAREAPASS 3
#define SUBTYPE_SELFPASS 4
#define SUBTYPE_PASS_IN_BACK 5


struct ABall {
  Vector pos;
  Vector vel;
  bool valid;
  int age;
};

struct AObject {
  Vector pos;
  Vector vel;
  double body_angle;
  int number;
  int stamina;
  bool valid;
  int age;
};

struct AState {  // Abstract State
  int my_idx;
  AObject my_team[11];
  AObject op_team[11];
  ABall ball;
};

struct AAction { // Abstract Action
  int subtype;
  int action_type;
  int target_player;
  Vector virtual_ballpos; // for go4pass
  Vector virtual_ballvel; // for go4pass

  Vector target_position;
  Vector actual_resulting_position; // position where the player gets the pass
  Vector actual_resulting_ballvel;  // ball velocity when the player gets the pass
  Vector potential_position; // position that the player might reach

  int acting_player;
  float kick_velocity;
  float kick_dir;
  float V;
  float dash_power;
  float cost; //Kosten fuer diese Aktion
  int advantage; // how many cycles is our team quicker than opponent?
  int time2intercept; // how many cycles to intercept
  int cycles2go;
  int targetplayer_number;
  bool risky_pass;
};


struct AObjectArray {
  int num;
  AObject const* obj[11];
};

#if 0
struct AObjectEditArray {
  int num;
  AObject * obj[11];
};
#endif 

class AbstractMDP {
 public:
  static void init_params();
  static void copy_mdp2astate(AState &state);
  static void model(const AState& state, const AAction& action,  AState &succ_state);
  static void target_model(const AState& state, const AAction& action,  AState &succ_state);
  static void model_fine(const AState& state, const AAction& action,  AState &succ_state);
  // model for joint action (positions) of whole team
  static void model(const AState& state, const AAction action[],  AState &succ_state);
  static void set_action(AAction &action, int type, int actor, int recv, Vector target, 
			 float vel,float dir=0, Vector resulting_pos = Vector(0), Vector resulting_vel = Vector(0), int advantage = 0,
			 int time2intercept = 0, int cycles2go = 0, int targetplayer_number = 0,
			 const bool risky_pass = false);

  static void set_action_go4pass(AAction &action, const Vector vballpos, const Vector vballvel);

  //static Main_Move *aaction2move(AAction action);

  /***************************************************************************/
  /** here come some utilities to access a AState. This are optimized 
      funtions and should be used instead of own "hacked" versions.
  */
  static void get_valid_teammates(const AState &, AObjectArray & arr);
  static void get_valid_opponents(const AState &, AObjectArray & arr);

  /** extracts the nearest <num_of_first> teammates to the given <pos>,
      the extracted AObjects are sorted (by nearest to <pos>)!!! */
  static void get_first_teammates_by_distance_to_pos(int num_of_first, const AState &, const Vector & pos, AObjectArray & arr);
  /** extracts the nearest <num_of_first> oppenents to the given <pos>
      the extracted AObjects are sorted (by nearest to <pos>)!!! */
  static void get_first_opponents_by_distance_to_pos(int num_of_first, const AState &, const Vector & pos, AObjectArray & arr);
  /** extracts the <num_of_first> first teammates.
      they are sorted in ascending order by their x coordinate (e.g. leftmostplayer is first, goalie is last)
  */
  static void get_first_teammates_by_x_coord(int num_of_first, const AState & , AObjectArray & arr);
  /** extracts the <num_of_last> first teammates.
      they are sorted in descending order by their x coordinate (e.g. leftmostplayer is last, goalie is first)
  */
  static void get_last_teammates_by_x_coord(int num_of_last, const AState & , AObjectArray & arr);
 protected:
  static void get_valid_objects(const AObject *obj, AObjectArray & arr);
  //static void get_valid_objects(AObject *obj, AObjectEditArray & arr);
  static void get_first_objects_by_distance_to_pos(int num_of_first, const Vector & pos, AObjectArray &);
  static int passreceiver_max_age;
  static int selfpass_mode;
  static int dribble_duration;
};

#endif 
