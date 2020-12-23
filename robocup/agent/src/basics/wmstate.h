#ifndef _WMSTATE_H_
#define _WMSTATE_H_

#include "basics/Cmd.h"
#include "basics/wmoptions.h"
#include "basics/WM.h"
#include "basics/WMTime.h"
#include "PlayerFlagSet.h"
#include "sensorbuffer.h"
#include "mdpstate.h"
#include "pfilter.h"
#include "server_options.h"

class WMstate {
public:
	void init();

	double my_distance_to_ball();

	long ms_time_between_see_messages();

	void import_msg_fullstate(const Msg_fullstate &);
	void import_msg_fullstate(const Msg_fullstate_v8 &);
	void incorporate_msg_fullstate_wrt_view_area(const Msg_fullstate &);
	void compare_with_wmstate(const WMstate &) const;
	void show_object_info() const;
	void export_mdpstate(MDPstate &) const;
	void export_ws(WS &) const;

	void export_msg_teamcomm(Msg_teamcomm &) const;
	Msg_teamcomm2 export_msg_teamcomm( const Cmd_Say & say ) const;

	void incorporate_cmd_and_msg_sense_body(const Cmd &, const Msg_sense_body &);
	void incorporate_cmd(const Cmd &);

	void incorporate_msg_hear(const Msg_hear &);
	void incorporate_msg_see(const Msg_see &, long ms_time, long ms_time_delay);
	void incorporate_msg_change_player_type(const Msg_change_player_type &);

	void reduce_dash_power_if_possible(Cmd_Body & cmd) const;
protected:
	double determine_KONV(int user_space_play_mode) const;
	void check_statistics() const;
	void export_unknown_player(Vector pos, Player & wstate_p, double KONV ) const;
	void export_ws_player( int team, int number, Player & wstate_p, double KONV ) const;
	void incorporate_msg_teamcomm(const Msg_teamcomm &);
	void incorporate_msg_card(int hear_time, const Msg_card & mc);
	void incorporate_msg_teamcomm(int time, const Msg_teamcomm2 &);
	int server_playmode_to_player_playmode(PlayMode server_playmode) const;
	void update_me_from_msg_see(const Msg_see & see);
	void update_ball_from_msg_see(const Vector & my_pos_before_update, const Msg_see & see);
	void update_players_from_msg_see(const Msg_see & see);
	void handle_inconsistent_objects_in_view_area();
	void stop_ball_in_immediate_vicinity_of_other_players(); //heuristics, shuould be done by the agent himself (in the future)

	double time_diff_ball_to_mdpstate_probability(int tdiff) const;
	double time_diff_player_to_mdpstate_probability(int team, int tdiff) const;

public:
	WMTime time;
	WMTime time_of_last_msg_see;
	long   ms_time_of_last_msg_see;
	long   ms_time_of_last_msg_see_after_sb;
	PlayMode play_mode;
	int    penalty_side;
	int    penalty_count; //only counts own penalties

	int my_score;
	int his_score;
	bool synch_mode_ok; ///< for new synched viewing
	int view_quality; ///< takes values {HIGH,LOG}
	int view_width;  ///< takes values {WIDE,NORMAL,NARROW}
	int next_cycle_view_quality;
	int next_cycle_view_width;

	int my_goalie_number;  ///< value 0 means unknown, value -1 means not existent
	int his_goalie_number; ///< value 0 means unknown, value -1 means not existent

	int my_attentionto;
	int my_attentionto_duration;

	ANGLE my_angle;
	ANGLE my_neck_angle;
	double my_effort;
	double my_recovery;
	double my_stamina_capacity;

	int kick_count;
	int dash_count;
	int turn_count;
	int say_count;
	int turn_neck_count;
	int catch_count;
	int move_count;
	int change_view_count;

	double my_speed_value;
	ANGLE my_speed_angle;
	ANGLE my_neck_angle_rel;

	long timeOfCollision; // JTS added for new collision detection

	struct _teamcomm_statistics {
		_teamcomm_statistics() { teamcomm_count= 0; teamcomm_partial_count= 0; pass_info_count= 0;
		recent_ballholder_info_cycle=-1; recent_ballholder_number=-1; }//TG08
		WMTime send_time_of_last_teamcomm;
		int    sender_of_last_teamcomm;
		WMTime recv_time_of_last_teamcomm;
		int    teamcomm_count;
		int    pass_info_count;
		int    teamcomm_partial_count;
		int    recent_ballholder_info_cycle;//TG08
		int    recent_ballholder_number;//TG08
	};

	struct _wm_player {
		_wm_player() { type= -1;
		pass_info.valid= false;
		pass_request.valid = false; //TGpr
		direct_opponent_conflict_number = -1;
		direct_opponent_number = -1;
		yellow_card = false;
		red_card = false;
		stamina_capacity_bound = ServerOptions::stamina_capacity; 
                tackle_time = -ServerOptions::tackle_cycles; action_time = -1; }
		bool alive;  ///< if a player is not alive, don't rely on the other values


		WMTime time_pos; ///< time of last update
		WMTime time_vel;
		WMTime time_angle;

		Vector pos;
		Vector vel;
		ANGLE body_angle;
		ANGLE neck_angle; //this is an absolute angle!
		double stamina;
		int type;

		bool tackle_flag;  //true if the oppoent is tackling (it's a snapshop from time 'time')
                int  tackle_time;  //probable time of most recent tackle (inferred from viewing) 
                int  action_time;  //recent time the player was seen without tackle_flag true
		bool kick_flag;
		bool pointto_flag; //pointto_dir/dist is only valid, if pointto_flag == true
		int pointto_time;  //last time a pointto information was seen

		ANGLE pointto_dir;
		double pointto_dist;

		struct {
			bool valid;
			int recv_time;
			Vector ball_pos;
			Vector ball_vel;
			int time; //this is the absolute time when ball_pos and ball_vel will be valid!
		} pass_info;

		//TGpr: begin
		struct
		{
			bool valid;
			int pass_in_n_steps;
			int pass_param;
			int received_at;
		} pass_request;
		//TGpr: end

		//JTS 10: yc / rc
		bool yellow_card;
		bool red_card;
		bool fouled;
		int foul_cycles;

		bool unsure_number;
		int direct_opponent_conflict_number; // JTS10 doa conflict information as provided by the coach
		int  direct_opponent_number; //number of direct opponent, as assigned by the coach
		double stamina_capacity_bound; // JTS10 used to store the stamina bound info send by the coach
	};

	int current_direct_opponent_assignment; //TGdoa
	int number_of_direct_opponent_updates_from_coach; //TGdoa

	struct _unknown_players {
		WMTime time;
		static const int max_num= 3;
		int num;
		Vector pos[max_num];
	};

	struct _wm_ball {
		WMTime time_pos;
		WMTime time_vel;
		Vector pos;
		Vector vel;
	};

	struct _wm_me_and_ball {
		WMTime time;
		Vector old_ball_rel_pos;
		Vector approx_ball_rel_pos;
		//Vector my_vel;
		Vector my_move;
		Vector my_old_pos;
		bool probable_collision;
	};

	_teamcomm_statistics teamcomm_statistics;
	_unknown_players unknown_players;
	_wm_player my_team[NUM_PLAYERS+1];
	_wm_player his_team[NUM_PLAYERS+1];
	_wm_ball ball;
	_wm_me_and_ball me_and_ball;
protected:
	ParticleFilter pfilter;

	SayMsg msg;

	int get_obj_id(int team, int number) const;
	int get_obj_team(int obj_id) const;
	int get_obj_number(int obj_id) const;

	static const int ball_id= 0;
	//static const int ball_vel_id= NUM_PLAYERS*2+1;
	static const int obj_id_MAX_NUM= NUM_PLAYERS*2+2;
	int teamcomm_times_buffer[obj_id_MAX_NUM];
	int * teamcomm_times;   //just a hack to allow writing to teamcomm_time in a const method (very ugly!!)

	Vector compute_object_pos(const Vector & observer_pos, ANGLE observer_neck_angle_abs,
			double dist, double dir) const;
	///very sophisticated method to compute velocity of an object (see the implementation for more documentation)
	Vector compute_object_vel(const Vector & observer_pos, const Vector & observer_vel,
			const Vector & object_pos, double dist, double dist_change, double dir_change) const;

	///simulates a collision of the object with an obstacle, position of the obstacle never changes!
	Vector compute_object_pos_after_collision(const Vector & object_pos, const Vector & object_vel, double object_radius, const Vector obstacle_pos, double obstacle_radius) const;
	//Vector compute_object_pos_after_collision(const Vector & object_pos, const Vector & object_vel, double object_radius, const Vector & obstacle_pos, const Vector & obstacle_pos, double obstacle_radius) const;

	///simulates a collision of the object with an obstacle, position of the obstacle never changes!
	Vector compute_object_vel_after_collision(const Vector & object_vel) const { return -0.1*object_vel; }

	///gets a reference to a player
	inline _wm_player * ref_player(int team, int number);
	///gets a constant reference to a player
	inline const _wm_player * const_ref_player(int team, int number) const;

	///gets a constant reference to a player using an object id
	inline const _wm_player * const_ref_player(int obj_id) const;

	///only not too old players are considered (max_age), the result is the minimal distance squared!!!
	bool get_nearest_player_to_pos(int team, const Vector & pos, const PlayerFlagSet & players, int max_age, int & res_team, int & res_number, double & sqr_res_distance) const;


	///
	bool in_feel_area(const Vector & pos, const Vector & object_pos, double tolerance= 0.0) const;

	///
	bool in_view_area(const Vector & pos, const ANGLE & neck_angle, int v_width, const Vector & object_pos) const;

	void show_view_area(const Vector & pos, const ANGLE & neck_angle, int v_width) const;

};

#endif
