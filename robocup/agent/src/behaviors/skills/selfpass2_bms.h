#ifndef _SELFPASS2_BMS_H_
#define _SELFPASS2_BMS_H_

#include "../../basics/Cmd.h"
#include "../base_bm.h"
#include "Vector.h"
#include "angle.h"
#include "basic_cmd_bms.h"
#include "oneortwo_step_kick_bms.h"
#include "options.h"
#include "ws_info.h"
#include "log_macros.h"


#define MAX_STEPS 24 //TG09: war ehedem: 20

class Selfpass2: public BodyBehavior {
  static bool initialized;
  BasicCmd *basic_cmd;
  OneOrTwoStepKick *onetwostepkick;
  

 private:
  typedef struct {
    int valid_at;
    Cmd cmd;
    Vector my_pos;
    Vector my_vel;
    ANGLE my_bodydir;
    bool I_have_ball;
    Vector ball_pos;
    Vector ball_vel;
    bool have_ball;
    Vector op_pos;
    int op_num;
    int op_steps2pos;
  } Simtable;

  Simtable simulation_table[MAX_STEPS];

  void simulate_my_movement(const ANGLE targetdir, const int max_steps, Simtable *simulation_table, 
			    const Vector mypos, const Vector myvel, const ANGLE myang,
			    const double mystamina,
			    const Vector ballpos, const Vector ballvel,
			    const bool with_kick, const bool turn2dir_only = false);
  void get_cmd_to_go2dir(Cmd &tmp_cmd,const ANGLE targetdir,const Vector pos, const Vector vel, 
             const ANGLE bodydir, const int stamina,
             const double inertia_moment = ServerOptions::inertia_moment,
             const double stamina_inc_max = ServerOptions::stamina_inc_max);
  void get_cmd_to_go2dir_real_data(Cmd &tmp_cmd,const ANGLE targetdir);

  void get_cmd_to_go2pos(Cmd &tmp_cmd,const Vector targetpos,const Vector pos, const Vector vel, 
			 const ANGLE bodydir, const int stamina,
			 const PPlayer player);

  void simulate_ops_movement(const Vector mypos, const ANGLE targetdir, Simtable *simulation_table, const bool target_is_ball= false);
  int get_min_cycles2_pos(const Vector mypos, const ANGLE targetdir,const Vector targetpos, 
                          const PPlayer player, const int max_steps, Vector &resulting_pos, bool forceCalculation=false);

  void print_table(Simtable *simulation_table);
  bool determine_kick(int &advantage, const double max_dist, Simtable *simulation_table, const ANGLE targetdir,
		      Vector & targetpos, double & targetspeed, int & steps, Vector &attacking_op,
		      int & attacking_number, 
		      const Vector mypos, const Vector myvel, const ANGLE myang,
		      const double mystamina,
		      const Vector ballpos, const Vector ballvel,
		      const int reduce_dashes=0 );
  bool check_nokick_selfpass(int &advantage, Simtable *simulation_table, Vector & targetpos, int & steps, Vector &attacking_op,
			     int & attacking_number, const Vector ballpos);
  bool at_position(const Vector playerpos, const ANGLE bodydir, const double kick_radius, const Vector targetpos);
  bool are_intermediate_ballpositions_safe(const Vector mypos, const ANGLE targetdir,Vector ballpos, Vector ballvel, const int num_steps);
  void reset_simulation_table(Simtable *simulation_table);
  
  static int  cvSelfpassRiskLevel;//TG09
  static long cvLastSelfpassRiskLevelChange;//TG09
  static void updateSelfpassRiskLevel();//TG09

 public:
  bool is_turn2dir_safe(int &advantage, const ANGLE targetdir, double &speed, Vector &ipos, int &steps,
			Vector &attacking_op, int & op_number, const bool check_nokick_only = false, 
			const int max_dashes=10);
  // checks, if I can turn into target direction, with the ball lying in front of me


  bool is_turn2dir_safe(int &advantage, const ANGLE targetdir, double &speed, Vector &ipos, int &steps,
			Vector &attacking_op, int & op_number, 
			const Vector mypos, const Vector myvel, const ANGLE myang,
			const double mystamina,
			const Vector ballpos, const Vector ballvel,
			const bool check_nokick_only = false, 
			const int max_dashes=10);


  bool is_selfpass_safe(int &advantage, const ANGLE targetdir, double &speed, Vector &ipos, int &steps,
			Vector &attacking_op, int & op_number, const bool check_nokick_only = false, 
			const int reduce_dashes=0);

  bool is_selfpass_safe_max_advance(int &advantage, const double max_dist, const ANGLE targetdir, double &speed, Vector &ipos, int &steps,
			Vector &attacking_op, int & op_number, const bool check_nokick_only = false, 
			const int reduce_dashes=0);

  bool is_selfpass_safe(int &advantage, const ANGLE targetdir, double &speed, Vector &ipos, int &steps,
			Vector &attacking_op, int & op_number, 
			const Vector mypos, const Vector myvel, const ANGLE myang,
			const double mystamina,
			const Vector ballpos, const Vector ballvel,
			const bool check_nokick_only = false, 
			const int reduce_dashes=0);


  bool is_selfpass_safe_max_advance(int &advantage, const double max_dist, const ANGLE targetdir, double &speed, Vector &ipos, int &steps,
			Vector &attacking_op, int & op_number, 
			const Vector mypos, const Vector myvel, const ANGLE myang,
			const double mystamina,
			const Vector ballpos, const Vector ballvel,
			const bool check_nokick_only = false, 
			const int reduce_dashes=0);


  bool is_selfpass_safe_with_kick(int &advantage, const double max_dist, const ANGLE targetdir, double &speed, Vector &ipos, int &steps,
			Vector &attacking_op, int & op_number, 
			const Vector mypos, const Vector myvel, const ANGLE myang,
			const double mystamina,
			const Vector ballpos, const Vector ballvel,
			const int reduce_dashes=0);

  bool is_selfpass_safe_without_kick(int &advantage, const ANGLE targetdir, double &speed, Vector &ipos, int &steps,
			Vector &attacking_op, int & op_number, 
			const Vector mypos, const Vector myvel, const ANGLE myang,
			const double mystamina,
			const Vector ballpos, const Vector ballvel,
			const int reduce_dashes=0);

  bool is_dashonly_safe(int &advantage, const ANGLE targetdir, double &speed, Vector &ipos, int &steps,
			Vector &attacking_op, int & op_number, 
			const Vector mypos, const Vector myvel, const ANGLE myang,
			const double mystamina,
			const Vector ballpos, const Vector ballvel,
			const int reduce_dashes=0);





  bool is_selfpass_safe(int &advantage, const ANGLE targetdir);
  bool is_selfpass_still_safe(int &advantage, const ANGLE targetdir, double & kickspeed, int &op_num);

  bool get_cmd(Cmd &cmd);
  bool get_cmd(Cmd & cmd, const ANGLE targetdir, const Vector targetpos, const double kickspeed);

  static bool init(char const * conf_file, int argc, char const* const* argv);
  Selfpass2();
  virtual ~Selfpass2();
  
  static int getRiskLevel();//TG09
  
  
};

#endif
