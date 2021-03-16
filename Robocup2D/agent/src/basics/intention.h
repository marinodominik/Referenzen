#ifndef _INTENTION_H
#define _INTENTION_H

#include "Vector.h"

#define SELFPASS 1
#define IMMEDIATE_SELFPASS 2
#define PASS 3
#define LAUFPASS 4
#define KICKNRUSH 5
#define SCORE 6
#define OPENING_SEQ 7
#define PANIC_KICK 8
#define TACKLING 9

#define DRIBBLE 10
#define SOLO 11
#define WAITANDSEE 12
#define HOLDTURN 13
#define TINGLETANGLE 14

#define TURN_AND_DASH 20
#define DRIBBLE_QUICKLY 21

#define GOTO 30
#define GO2BALL 31
#define STAY 32
#define GOHOME 33
#define BACKUP 34

#define SUBTYPE_NONE 0
#define SUBTYPE_PASS 1
#define SUBTYPE_LAUFPASS 2
#define SUBTYPE_PENALTYAREAPASS 3
#define SUBTYPE_SELFPASS 4
#define SUBTYPE_PASS_IN_BACK 5


/* we could add more here... */
enum { NECK_REQ_NONE, NECK_REQ_LOOKINDIRECTION, NECK_REQ_PASSINDIRECTION, NECK_REQ_BLOCKBALLHOLDER,
       NECK_REQ_SCANFORBALL, NECK_REQ_FACEBALL, NECK_REQ_DIRECTOPPONENTDEFENSE };
enum { ATT_REQ_NONE };

class Intention{
 private:
 public:
  int subtype;
  long valid_since_cycle;  // cycle at which intention was first initialized
  long valid_at_cycle;  // cycle at which intention is valid
  double V;
  bool risky_pass;
  bool immediatePass;
  int type;
  int target_player;
  double priority; // gives priority/ quality of a pass or laufpass
  Vector resultingpos;
  ANGLE target_body_dir;
  double kick_speed;
  double kick_dir;
  Vector kick_target;
  Vector player_target; // for go2s...
  Vector potential_pos;
  int advantage;
  // ridi: new for 06 (selfpass information
  int attacker_num;
  bool wait_then_pass; // flag um anzuzeigen, dass nicht direkt gekickt werden soll.

  void copyFromIntention(Intention &intn);

  bool is_pass2teammate();
  bool is_selfpass();
  bool is_dribble();

  void set_pass(const Vector &target, const double &speed, const int valid_at, int target_player_number= 0,
		double priority = 0, const Vector ipos = Vector(0), const   Vector potential_position = Vector(0));
  void set_laufpass(const Vector &target, const double &speed, const int valid_at, int target_player_number= 0,
		    double priority = 0, const Vector ipos = Vector(0),const bool is_risky = false,
		    const   Vector potential_position = Vector(0));
  void set_pass(const Vector &target, const double &speed, const double &kickdir, const int valid_at,
		int target_player_number= 0,
		double priority = 0, const Vector ipos = Vector(0), const   Vector potential_position = Vector(0));
  void set_laufpass(const Vector &target, const double &speed, const double &kickdir, const int valid_at,
		    int target_player_number= 0,
		    double priority = 0, const Vector ipos = Vector(0),const bool is_risky = false,
		    const   Vector potential_position = Vector(0));
  void set_opening_seq(const Vector &target, const double &speed, const int valid_at,
		       int target_player_number= 0);
  void set_selfpass(const ANGLE & targetdir, const Vector &target, const double &speed, const int valid_at);
  void set_selfpass(const ANGLE & targetdir, const Vector &target, const double &speed, const int valid_at, const int attackernum);
  void set_immediateselfpass(const Vector &target, const double &speed, const int valid_at);
  void set_dribble(const Vector &target, const int valid_at);
  void set_tingletangle(const Vector &target, const int valid_at);
  void set_dribblequickly(const Vector &target, const int valid_at);
  void set_holdturn(const ANGLE body_dir, const int valid_at);
  void set_score(const Vector &target, const double &speed, const int valid_at);
  void set_kicknrush(const Vector &target, const double &speed, const int valid_at);
  void set_panic_kick(const Vector &target, const int valid_at);
  void set_tackling( const double &speed, const int valid_at );
  void set_backup(const int valid_at);
  void set_waitandsee(const int valid_at);
  void set_turnanddash(const int valid_at);
  void set_goto(const Vector &target, const int valid_at);
  void confirm_intention(const Intention intention, const int valid_at);
  void correct_target(const Vector &target){kick_target = target;};
  void correct_speed(const double &speed){kick_speed = speed;};
  void reset();
  int get_type();
  long valid_at(){return valid_at_cycle;};
  long valid_since();
  void set_valid_since(const long cycle);
  void set_valid_at(const long cycle);
  bool get_kick_info(double &speed, Vector &target, int & target_player_number); // false if intention is not a shoot
  bool get_kick_info(double &speed, Vector &target) { // false if intention is not a shoot
    int dum;
    return get_kick_info(speed,target,dum);
  } 
};

class NeckRequest {
  int valid_at_cycle;
  int type;
  double p0;
 public:
  NeckRequest();
  void set_request(int type, double p0 = 0);
  int get_type();
  double get_param();
  bool is_set();
};

class AttentionToRequest
{
  int ivValidAtCycle;
  int ivAttentionToPlayer;
 public:
  AttentionToRequest();
  void set_request(int plNr);
  int  get_param();
  bool is_set();
};

//TGpr: begin
class PassRequestRequest
{
  int ivValidAtCycle;
  int ivRequestPassFromPlayer;
  int ivRequestPassInDirection; //angle in degree!
 public:
  PassRequestRequest();
  void set_request(int plNr, int passAngleParameter);
  bool get_param(int &plNr, int &passAngleParameter);
  bool is_set();
};
//TGpr: end

#endif 
