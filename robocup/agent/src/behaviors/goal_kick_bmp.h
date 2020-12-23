#ifndef _GOAL_KICK_BMP_H_
#define _GOAL_KICK_BMP_H_

#include "../basics/Cmd.h"
#include "base_bm.h"
#include "log_macros.h"
#include "skills/basic_cmd_bms.h"
#include "skills/face_ball_bms.h"
#include "skills/neuro_go2pos_bms.h"
#include "skills/neuro_intercept_bms.h"
#include "skills/one_step_kick_bms.h"
#include "skills/oneortwo_step_kick_bms.h"
#include "skills/neuro_kick05_bms.h"

#define MAX_GFK_POS 20

class GoalKick : public BodyBehavior {
	static bool initialized;

	BasicCmd *basiccmd;
	NeuroGo2Pos *go2pos;
	FaceBall *faceball;
	OneStepKick *onestepkick;
	OneOrTwoStepKick *onetwokick;
	NeuroKick05 *neurokick;
	NeuroIntercept *intercept;

	static double homepos_tolerance;
	static int homepos_stamina_min;
	static int wait_after_catch;
	static int max_wait_after_catch;

	enum {GFK_MODE,GK_MODE};

	long last_called;
	long seq_started;
	long play_on_cnt;
	int goal_kick_mode;

	bool move_to_home;

	int mode;

	double wished_kick_vel;
	Vector wished_kick_pos;
	Vector kickoff_pos;
	Vector target_pos;
	double target_vel;
	int    target_num;

	Vector bestpos,bestvel;



	bool get_goalie_cmd(Cmd &cmd);
	bool get_player_cmd(Cmd &cmd);

	/* data for goalie free kick */
	static int gfk_pos_num;
	static Vector gfk_kickoff_pos[];
	static Vector gfk_target_pos[];
	static double gfk_kickvel[];
	int gfk_rand_arr[MAX_GFK_POS];

	int gfk_goalie_mode;
	Vector gfk_tpos;

	/* methods for goalie free kick */
	bool get_gfk_player_cmd(Cmd &cmd);
	bool get_gfk_goalie_cmd(Cmd &cmd);
	bool panic_gfk(Cmd &cmd);

	/* data for goal kick */
	int gk_goalie_mode;
	Vector gk_tpos;
	bool gk_left;
	int cyc_cnt;
	bool wing_not_possible;

	bool didMove; // CIJAT OSAKA

	/* methods for goal kick */
	bool get_gk_goalie_cmd(Cmd &cmd);
	bool panic_gk(Cmd &cmd);

	bool scan_field(Cmd &cmd);
	void revalidateTargetPosition( Vector & targetpos ); // currentPosition.x auch ins positive laufen lassen
	bool areDangerousOpponentsNearPosition( Vector pos );

public:
	GoalKick();
	virtual ~GoalKick();

	static bool init(char const * conf_file, int argc, char const* const* argv);

	bool get_cmd(Cmd &cmd);
};

#endif
