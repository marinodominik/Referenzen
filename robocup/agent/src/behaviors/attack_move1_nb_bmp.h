#ifndef _ATTACK_MOVE1_NB_BMC_H_
#define _ATTACK_MOVE1_NB_BMC_H_

#include "base_bm.h"
#include "skills/basic_cmd_bms.h"
#include "skills/neuro_go2pos_bms.h"
#include "skills/face_ball_bms.h"

class Attack_Move1_Nb: public BodyBehavior {
 private:
#define MAXPOS 50
  Vector testpos[MAXPOS];
  float y_variation; // could be made player dependend
  float x_variation; // could be made player dependend
  float mindist2teammate;

  float offside_line;
  void determine_offside_line();
  Vector get_targetpos() ;
  Vector get_targetpos_leftattacker() ;
  Vector get_targetpos_centreattacker() ;
  Vector get_targetpos_rightattacker() ;
  Vector get_targetpos_leftmidfielder() ;
  Vector get_targetpos_centremidfielder() ;
  Vector get_targetpos_rightmidfielder() ;
  bool go2pos_intelligent(Cmd &cmd, const Vector target);
  bool do_waitandsee(Cmd &cmd);
  bool is_mypos_ok(const Vector & targetpos);
  bool shall_I_offer_myself();
  
  bool test_offside(Cmd & cmd);

  NeuroGo2Pos *go2pos;
  FaceBall *face_ball;
  BasicCmd *basiccmd;
  double check_all_positions(Vector & result, Vector * testpos, const int num_testpos,const bool test_wrt_targetpos = false,
			  const Vector targetpos = Vector(0,0) );
  double optimize_position(Vector & result, Vector * testpos, const int num_testpos,const PPlayer &teammate );
  bool ball_is_left() ;
  bool ball_is_right() ;
  bool ball_is_half_left() ;
  bool ball_is_far_left() ;
  static bool activated;

 public:
  static bool initialized;
  static bool init(char const * conf_file, int argc, char const* const* argv) {
    if ( initialized )
      return true;
    initialized= true;
    return (
	    NeuroGo2Pos::init(conf_file,argc,argv) &&
	    BasicCmd::init(conf_file,argc,argv) &&
	    FaceBall::init(conf_file,argc,argv)
	    );
  }
  Attack_Move1_Nb();
  virtual ~Attack_Move1_Nb();
  bool get_cmd(Cmd & cmd);
  static bool do_move();
};


#endif
