#include "basics/wmstate.h"

#include "basics/wmoptions.h"
#include "basics/WMTools.h"
#include "sensorparser.h"
#include "options.h"
#include "geometry2d.h"
#include "log_macros.h"
#include "macro_msg.h"
#include "serverparam.h"
#include "pfilter.h"
#include "pointto10_bmp.h" //JTS10 only the constants from this module are used 


//#define MYLOG_WM(YYY,LLL,XXX)  
//#define MYLOG_COMM(YYY,LLL,XXX) LOG_WM(YYY,LLL,XXX)
#define MYLOG_WM(YYY,LLL,XXX) LOG_WM(YYY,LLL,XXX) 
#define MYLOG_COMM(YYY,LLL,XXX)
#define MYLOG_COMM1(YYY,LLL,XXX) LOG_WM(YYY,LLL,XXX) 

void WMstate::init() {
	play_mode= PM_Null;
	penalty_side= unknown_SIDE;
	penalty_count= 0;
	WM::last_export_KONV= determine_KONV(PM_PlayOn);
	time.time= -1;
	my_score= 0;
	his_score= 0;
	synch_mode_ok= false;
	view_quality= HIGH;
	view_width= NARROW;
	ms_time_of_last_msg_see = -1;
	ms_time_of_last_msg_see_after_sb = 0;
	my_goalie_number= 0;
	his_goalie_number= 0;

	my_attentionto_duration= 0;

	kick_count= 0;
	dash_count= 0;
	turn_count= 0;
	say_count= 0;
	turn_neck_count= 0;
	catch_count= 0;
	move_count= 0;
	change_view_count= 0;
	timeOfCollision = -1;

	msg.valid= false;
	msg.time= -100;

	for (int i=0; i < NUM_PLAYERS+1; i++) {
		my_team[i].alive= false;
		my_team[i].type= 0;
		my_team[i].pos= Vector(35.0,0.0); //otherwise in the beginning our players think they have the ball
		his_team[i].alive= false;
	}

	_wm_player & me= my_team[ WM::my_number ];
	me.alive= true;
	INFO_OUT << ID << "my type= " << me.type;
	WM::set_my_type( ServerParam::get_player_type( me.type ) );

	teamcomm_times= teamcomm_times_buffer;
	for (int i=0; i< obj_id_MAX_NUM; i++)
		teamcomm_times[i]= -100;

	current_direct_opponent_assignment = -1; //TGdoa
	number_of_direct_opponent_updates_from_coach = 0; //TGdoa
}

double WMstate::my_distance_to_ball() {
    return ball.pos.distance( my_team[WM::my_number].pos );
}

long WMstate::ms_time_between_see_messages() {
    long res= 150;
    if ( view_quality == LOW )
        res /= 2;
    if ( view_width == WIDE )
        res *= 2;
    else if ( view_width == NARROW)
        res /= 2;
    return res;
}

void WMstate::import_msg_fullstate(const Msg_fullstate & msg) {
	//cerr << ">>>>>>>>>" << msg;
	time.time= msg.time;
	time.cycle= 0; //not cycle information available
	time.total_cycle++;
	time_of_last_msg_see= time;
	//cout << "\nmsg.time= " << msg.time << "  ,  time.time= " << time.time;

	play_mode= msg.play_mode;
	my_score= msg.my_score;
	his_score= msg.his_score;
	view_quality= msg.view_quality;
	view_width= msg.view_width;

	for (int i= 1; i < NUM_PLAYERS+1; i++) {
		my_team[i].alive= false;
		his_team[i].alive= false;
	}

	ball.time_pos= time;
	ball.time_vel= time;
	//cout << "\nmsg.time= " << msg.time << "  ,  ball.time.time= " << ball.time.time;
	ball.pos= Vector(msg.ball.x, -msg.ball.y);
	ball.vel= Vector(msg.ball.vel_x, -msg.ball.vel_y);

	for (int i= 0; i< msg.players_num; i++) {
		const Msg_fullstate::_fs_player & p = msg.players[i];

		_wm_player * wm_p = ref_player( p.team, p.number );
		wm_p->time_pos= time;
		wm_p->time_vel= time;
		wm_p->time_angle= time;

		wm_p->alive= true;
		wm_p->pos= Vector(p.x,-p.y);
		wm_p->vel= Vector(p.vel_x,-p.vel_y);
		wm_p->body_angle= WMTools::conv_server_angle_to_angle_0_2Pi(p.angle);
		wm_p->neck_angle= WMTools::conv_server_angle_to_angle_0_2Pi(p.neck_angle);
		wm_p->neck_angle+= wm_p->body_angle; // neck_angle is given relative in fullstate (but not in wmstate)!
		wm_p->stamina= p.stamina;
		wm_p->type= 0; //there is no type information in the old fullstate format
		if ( p.team == my_TEAM && p.number == WM::my_number ) {
			my_angle=  WMTools::conv_server_angle_to_angle_0_2Pi(p.angle);
			my_neck_angle_rel = WMTools::conv_server_angle_to_angle_0_2Pi(p.neck_angle);
			my_neck_angle = my_angle + my_neck_angle_rel;
			my_effort= p.effort;
			my_recovery= p.recovery;

			my_speed_value= wm_p->vel.norm();
			//my_speed_angle= my_neck_angle - ANGLE(wm_p->vel.arg()); // !!!!!!!!!!!! to be tested
		}
	}
#if 1 //test  
	static Vector old_pos;
	static int old_pos_time= -10;
	if ( old_pos_time+1 == time.time) {
		MYLOG_WM(time.time,0, << "real my_move" << my_team[WM::my_number].pos - old_pos);
	}
	else {
		MYLOG_WM(time.time,0, << "no real my_move");
	}
	old_pos= my_team[WM::my_number].pos;
	old_pos_time= time.time;
#endif
}

void WMstate::import_msg_fullstate(const Msg_fullstate_v8 & msg) {
	my_goalie_number= -1;
	his_goalie_number= -1;

	//cerr << ">>>>>>>>>" << msg;
	time.time= msg.time;
	time.cycle= 0; //not cycle information available
	time.total_cycle++;
	time_of_last_msg_see= time;
	//cout << "\nmsg.time= " << msg.time << "  ,  time.time= " << time.time;

	play_mode= msg.play_mode;
	my_score= msg.my_score;
	his_score= msg.his_score;
	view_quality= msg.view_quality;
	view_width= msg.view_width;

	for (int i= 1; i < NUM_PLAYERS+1; i++) {
		my_team[i].alive= false;
		his_team[i].alive= false;
	}
	my_team[ WM::my_number ].alive= true;

	ball.time_pos= time;
	ball.time_vel= time;
	//cout << "\nmsg.time= " << msg.time << "  ,  ball.time.time= " << ball.time.time;
	ball.pos= Vector(msg.ball.x, -msg.ball.y);
	ball.vel= Vector(msg.ball.vel_x, -msg.ball.vel_y);

	for (int i= 0; i< msg.players_num; i++) {
		const Msg_fullstate_v8::_fs_player & p = msg.players[i];

		_wm_player * wm_p = ref_player( p.team, p.number );
		wm_p->time_pos= time;
		wm_p->time_vel= time;
		wm_p->time_angle= time;

		wm_p->alive= true;
		wm_p->pos= Vector(p.x,-p.y);
		wm_p->vel= Vector(p.vel_x,-p.vel_y);
		wm_p->body_angle= WMTools::conv_server_angle_to_angle_0_2Pi(p.angle);
		wm_p->neck_angle= WMTools::conv_server_angle_to_angle_0_2Pi(p.neck_angle);
		wm_p->neck_angle+= wm_p->body_angle; // neck_angle is given relative in fullstate (but not in wmstate)!
		wm_p->stamina= p.stamina;
		wm_p->type= p.type;
		//JTS 10 card handling
		if ( ClientOptions::server_version >= 14 ) {
			wm_p->fouled = p.fouled;
			wm_p->foul_cycles = p.foul_cycles;
			if ( p.card == YELLOW_CARD )
				wm_p->yellow_card = true;
			if ( p.card == RED_CARD )
				wm_p->red_card = true;
		}
		//TG14: pointto data
		if (p.point_dist < 0.0)
			wm_p->pointto_flag = false;
		else
		{
			wm_p->pointto_flag = true;
			wm_p->pointto_dist = p.point_dist;
			wm_p->pointto_dir = WMTools::conv_server_angle_to_angle_0_2Pi(p.point_dir);
		}
		//TG14: kick and tackle flags
		wm_p->kick_flag = p.kick_flag;
		wm_p->tackle_flag = p.tackle_flag;
                if (p.tackle_flag && ((time.time - wm_p->tackle_time) > ServerOptions::tackle_cycles)) wm_p->tackle_time= time.time - 1;
                if (p.tackle_flag == false) wm_p->action_time= time.time - 1;

		if ( p.goalie ) {
			if (p.team == my_TEAM)
				my_goalie_number= p.number;
			else{
				his_goalie_number= p.number;
				//INFO_OUT << ID << "\ntime= " << time.time <<" no=" <<WM::my_number << " his goalie number "<<his_goalie_number;
			}
		}

		if ( p.team == my_TEAM && p.number == WM::my_number ) {
			my_angle=  WMTools::conv_server_angle_to_angle_0_2Pi(p.angle);
			my_neck_angle_rel = WMTools::conv_server_angle_to_angle_0_2Pi(p.neck_angle);
			my_neck_angle = my_angle + my_neck_angle_rel;
			my_effort= p.effort;
			my_recovery= p.recovery;
			my_stamina_capacity= p.stamina_capacity;

			my_speed_value= wm_p->vel.norm();
			//my_speed_angle= my_neck_angle - ANGLE(wm_p->vel.arg()); // !!!!!!!!!!!! to be tested
		}
	}
#if 1 //test  
	static Vector old_pos;
	static int old_pos_time= -10;
	if ( old_pos_time+1 == time.time) {
		MYLOG_WM(time.time,0, << "real my_move" << my_team[WM::my_number].pos - old_pos);
	}
	else {
		MYLOG_WM(time.time,0, << "no real my_move");
	}
	old_pos= my_team[WM::my_number].pos;
	old_pos_time= time.time;
#endif
}

void WMstate::incorporate_msg_fullstate_wrt_view_area(const Msg_fullstate & msg) {
	//cerr << ">>>>>>>>>" << msg;
	time.time= msg.time;
	time.cycle= 0; //not cycle information available
	time.total_cycle++;
	time_of_last_msg_see= time;
	//cout << "\nmsg.time= " << msg.time << "  ,  time.time= " << time.time;

	play_mode= msg.play_mode;
	my_score= msg.my_score;
	his_score= msg.his_score;
	view_quality= msg.view_quality;
	view_width= msg.view_width;

	Vector my_pos;
	bool my_pos_known= false;
	for (int i= 0; i< msg.players_num; i++) {
		const Msg_fullstate::_fs_player & p = msg.players[i];
		if ( p.team == my_TEAM && p.number == WM::my_number ) {
			my_pos_known= true;
			my_pos= Vector( p.x, -p.y );
			my_angle=  WMTools::conv_server_angle_to_angle_0_2Pi(p.angle);
			my_neck_angle_rel = WMTools::conv_server_angle_to_angle_0_2Pi(p.neck_angle);
			my_neck_angle = my_angle + my_neck_angle_rel;
			my_effort= p.effort;
			my_recovery= p.recovery;
			break;
		}
	}

	if (! my_pos_known ) {
		ERROR_OUT << ID << "\nWMstate::incorporate_msg_fullstate_wrt_view_area: I'm not listed in fullstate, aborting";
		return;
	}

	Vector pos= Vector(msg.ball.x, -msg.ball.y);
	if ( in_view_area(my_pos,my_neck_angle, view_width, pos)
			|| in_feel_area(my_pos,pos) ) {
		ball.time_pos= time;
		ball.time_vel= time;
		ball.pos= pos;
		ball.vel= Vector(msg.ball.vel_x, -msg.ball.vel_y);
	}

	for (int i= 0; i< msg.players_num; i++) {
		const Msg_fullstate::_fs_player & p = msg.players[i];
		Vector pos= Vector(p.x,-p.y);
		if ( in_view_area(my_pos,my_neck_angle, view_width, pos)
				|| in_feel_area(my_pos,pos)
				|| (p.team == my_TEAM && p.number == WM::my_number)) {
			_wm_player * wm_p = ref_player( p.team, p.number );
			wm_p->time_pos= time;
			wm_p->time_vel= time;
			wm_p->time_angle= time;
			wm_p->alive= true;
			wm_p->pos= Vector(p.x,-p.y);
			wm_p->vel= Vector(p.vel_x,-p.vel_y);
			wm_p->body_angle= WMTools::conv_server_angle_to_angle_0_2Pi(p.angle);
			wm_p->neck_angle= WMTools::conv_server_angle_to_angle_0_2Pi(p.neck_angle);
			wm_p->neck_angle+= wm_p->body_angle; // neck_angle is given relative in fullstate (but not in wmstate)!
			wm_p->stamina= p.stamina;

			if (p.team == my_TEAM && p.number == WM::my_number ) {
				my_speed_value= wm_p->vel.norm();
				//my_speed_angle= my_neck_angle - ANGLE(wm_p->vel.arg()); // !!!!!!!!!!!! to be tested
			}
		}
	}
}

void WMstate::compare_with_wmstate(const WMstate & wm) const {
	MYLOG_WM(time.time, 0, << " ***************  WMstate::compare_with_wmstate ********************** " );
	Vector tmpVec, tmpVec2;
	double eps= 0.1;
	double diff,diff2;

	//ANGLE tmpANG;
	//double tmp;

	if (time.time != wm.time.time) {
		MYLOG_WM(time.time, 0, << " ERROR time= " << time.time << "  [" << wm.time.time << "]" );
	}
	if (play_mode != wm.play_mode) {
		MYLOG_WM(time.time, 0, << " ERROR play_mode= " << 0 << "  [" << 0 << "]" );
	}

	if ( fabs(my_effort - wm.my_effort) > 0.01) {
		MYLOG_WM(time.time, 0, << "my_effort= " << my_effort << "  [" << wm.my_effort << "]" );
	}

	if (my_score !=  wm.my_score) {
		MYLOG_WM(time.time, 0, << " ERROR my_score= " << my_score << "  [" << wm.my_score << "]" );
	}
	if (his_score !=  wm.his_score) {
		MYLOG_WM(time.time, 0, << " ERROR his_score= " << his_score << "  [" << wm.his_score << "]" );
	}

	if ( view_quality != wm.view_quality ) {
		MYLOG_WM(time.time, 0, << " ERROR view_quality= " << view_quality << "  [" << wm.view_quality << "]" );
	}

	if ( view_width != wm.view_width ) {
		MYLOG_WM(time.time, 0, << " ERROR view_width= " << view_width << "  [" << wm.view_width << "]" );
	}


	double a= my_angle.diff(wm.my_angle);
	if ( a* 180.0 / M_PI > 15) {
		MYLOG_WM(time.time,0, "ERROR WRONG BODY ANGLE " << my_angle.get_value() << " [" << wm.my_angle.get_value() << "] diff= " << a* 180.0 / M_PI);
	}

	MYLOG_WM(time.time, 0, << " ball.time= " << ball.time_pos << " prob= " << time_diff_ball_to_mdpstate_probability(time.time-ball.time_pos.time));

	//difference in absolute ball pos
	diff=  ball.pos.distance(wm.ball.pos);
	if ( diff > eps) {
		MYLOG_WM(time.time, 0, << " ball.pos= " << ball.pos << "  [" << wm.ball.pos << "]" << " diff= " << diff );
	}

	const _wm_player & wm_my_p= wm.my_team[WM::my_number];
	const _wm_player & my_p= my_team[WM::my_number];

	//difference in absolute ball vel
	diff=  ball.vel.distance(wm.ball.vel);
	if ( diff > eps) {
		diff2= wm.ball.pos.distance( wm_my_p.pos );
		if ( diff2 > WM::my_kick_radius * 2 ) {
			MYLOG_WM(time.time, 0, << "ball.vel= " << ball.vel << "  [" << wm.ball.vel << "]" << " diff= " << diff);
		}
		else {
			MYLOG_WM(time.time, 0, << "XXX ball.vel= " << ball.vel << "  [" << wm.ball.vel << "]" << " diff= " << diff);
		}
	}


	MYLOG_WM(time.time, 0, << " my_p.time= " << my_p.time_pos << " my_p.pos= " << my_p.pos);

	tmpVec= ball.pos - my_p.pos;
	tmpVec2= wm.ball.pos - wm_my_p.pos;
	diff= tmpVec.norm();
	diff2= tmpVec2.norm();
	MYLOG_WM(time.time,0, << " REL BALL ball pos= " << tmpVec << ", dist= " << diff << " [" << tmpVec2 << ", " << diff2 << "]");
	if ( tmpVec.distance(tmpVec2) > eps ) {
		MYLOG_WM(time.time,0, << " ** Diff in rel ball pos = " << tmpVec.distance(tmpVec2));
	}
	if (diff < ServerOptions::kickable_area && diff2 > ServerOptions::kickable_area) {
		MYLOG_WM(time.time,0, << " ** I thick ball is kickable, but it is not");
	}
	if (diff > ServerOptions::kickable_area && diff2 < ServerOptions::kickable_area) {
		MYLOG_WM(time.time,0, << " ** I thick ball is NOT kickable, but it is");
	}

	if ( my_p.alive != wm_my_p.alive ) {
		MYLOG_WM(time.time, 0, << " my.alive= " << my_p.alive << "  [" << wm_my_p.alive << "]");
	}

	if ( fabs( my_p.stamina - wm_my_p.stamina) > 1) {
		MYLOG_WM(time.time, 0, << " my.stamina= " << my_p.stamina << "  [" << wm_my_p.stamina << "]" );
	}

	//difference in absolute pos
	diff=  my_p.pos.distance(wm_my_p.pos);
	if ( diff > eps) {
		MYLOG_WM(time.time, 0, << " my.pos= " << my_p.pos << "  [" << wm_my_p.pos << "]" << " diff= " << diff << " (TAKE CARE)");
	}
	else {
		MYLOG_WM(time.time, 0, << " my.pos diff= " << diff );
	}

	//difference in absolute vel
	diff=  my_p.vel.distance(wm_my_p.vel);
	if ( diff > eps) {
		if (diff > 5) {
			MYLOG_WM(time.time, 0, << " [[[[[[[[[[[[[[[[[[[[[");
		}
		MYLOG_WM(time.time, 0, << " my.vel= " << my_p.vel << "  [" << wm_my_p.vel << "]" << " diff= " << diff);
	}
	else {
		MYLOG_WM(time.time, 0, << " my.vel diff= " << diff );
	}

#if 0
	diff= ball.pos.distance(wm.ball.pos);
	if ( diff <= ServerOptions::player_size + ServerOptions::ball_size + 0.05)
		MYLOG_WM(time.time,0, << " Collision?  " << diff - ServerOptions::player_size - ServerOptions::ball_size << " -------- !!!!!!!!!!!!!!" );
	if ( tmpVec.distance(tmpVec2) > eps)
		MYLOG_WM(time.time, 0, << " rel ball.pos= " << tmpVec2 << "  [" << tmpVec << "]" << " diff= " << tmpVec.distance(tmpVec2) );

	tmpVec= Vector( p.x, p.y);
	if ( tmpVec.distance( wm_p.pos) > eps)
		MYLOG_WM(time.time, 0, << " my_pos= " << wm_p.pos << "  [" << tmpVec << "]" << " diff= " << tmpVec.distance(wm_p.pos) );

	tmpVec= Vector( p.vel_x, p.vel_y);
	if ( tmpVec.distance( wm_p.vel) > eps)
		MYLOG_WM(time.time, 0, << " my_vel= " << wm_p.vel << "  [" << tmpVec << "]" << " diff= " << tmpVec.distance(wm_p.vel) );

	tmpANG = my_angle -  p.angle;
	//MYLOG_WM(time.time, 0, << "    my_angle = " << my_angle << "   p.angle= " << p.angle << "  diffANG= " << tmpANG << "   diffANG (-PI,PI) = " << tmpANG.get_value_mPI_pPI() );
	if ( fabs( tmpANG.get_value_mPI_pPI() ) > eps*0.1 )
		MYLOG_WM(time.time, 0, << " my_angle= " << my_angle.get_value() << "  [" << p.angle << "]" << " diff_deg= " << tmpANG.get_value_mPI_pPI()*180.0/PI );

	tmpANG = my_neck_angle -  p.neck_angle;
	if ( fabs( tmpANG.get_value_mPI_pPI() ) > eps * 0.1 )
		MYLOG_WM(time.time, 0, << " my_neck_angle= " << my_neck_angle.get_value() << "  [" << p.neck_angle << "]" << " diff_deg= " << tmpANG.get_value_mPI_pPI()*180.0/PI );

	tmpVec= Vector( p.vel_x, p.vel_y);
	tmp= tmpVec.norm();
	if ( fabs ( my_speed_value - tmp ) > eps )
		MYLOG_WM(time.time, 0, << " my_speed_value= " << my_speed_value << "  [" << tmp << "]" << " diff " << fabs( my_speed_value - tmp) );
#endif
	MYLOG_WM(time.time, 0, << " ***************  WMstate::compare_with_wmstate <<<<<<<<<<<<<<<<<<<< end " );
}

void WMstate::show_object_info() const {
	MYLOG_WM(time.time, 0, << " ***************  WMstate::show_object_info ********************** " );

	MYLOG_WM(time.time, 0, << "time= " << time.time);
	MYLOG_WM(time.time, 0, << "play_mode= " << 0);
	MYLOG_WM(time.time, 0, << "my_effort= " << my_effort);
	MYLOG_WM(time.time, 0, << "my_score= " << my_score << " his_score= " << his_score);
	MYLOG_WM(time.time, 0, << "view_quality= " << view_quality << " view_width= " << view_width);

#if LOGGING && WORLD_MODEL_LOGGING
	const _wm_player & my_p= my_team[WM::my_number];
#endif
	MYLOG_WM(time.time, 0, << " my_p.time= " << my_p.time_pos << " my_p.pos= " << my_p.pos);

	Vector tmpVec, tmpVec2;
	tmpVec.init_polar(10,my_angle.get_value());
	tmpVec2.init_polar(5,my_neck_angle.get_value());
	//draw neck angle
	MYLOG_WM(time.time,1, << _2D
			<< VL2D(my_p.pos,my_p.pos+tmpVec,"magenta")
			<< VL2D(my_p.pos,my_p.pos+tmpVec2,"red")
			<< VC2D(my_p.pos,1.0,"magenta")
			<< VC2D(my_p.pos,0.3,"magenta")
			<< VC2D(ball.pos,0.3,"000000")
			<< STRING2D(ball.pos.getX()+0.4,ball.pos.getY(), "b," << time.time - ball.time_pos.time,"000000" ) //ball age
	);



	//comparing knowledge about other players
#if LOGGING && WORLD_MODEL_LOGGING
	char colors[4][7]= {"0000ff","ff0000","00ffff","ffff00"};
#endif
	for (int t=0; t < 2; t++)
		for (int i=0; i < NUM_PLAYERS + 1; i++) {
			const _wm_player * p= const_ref_player(t,i);
			if ( ! p->alive ) {
				//MYLOG_WM(time.time,0, << "player " << TEAM_STR[t] << i << " not alive");
				continue;
			}

#if LOGGING && WORLD_MODEL_LOGGING
			int col_offset= 0;
			if ( time_diff_player_to_mdpstate_probability(t, time.time - p->time_pos.time ) < 0.5 )
				col_offset= 2;
			char * color= colors[t+col_offset];
#endif
			if (i!=0 || time.time - p->time_pos.time < 3) {
				MYLOG_WM(time.time,2, << _2D << VC2D(p->pos,3,color) << VL2D(p->pos,p->pos+p->vel,"00ff00")
						<< STRING2D(p->pos.getX()+3.2,p->pos.getY(),i << "," << time.time - p->time_pos.time,color ) );
			}

			if ( p->time_angle == time ) {
				Vector tmp, tmp2;
				tmp.init_polar(5, p->body_angle.get_value() );
				tmp+= p->pos;
				tmp2.init_polar(3, p->neck_angle.get_value() );
				tmp2+= p->pos;
				MYLOG_WM(time.time,2, << _2D << VL2D( p->pos, tmp, color)
						<< VL2D( p->pos, tmp2, color));
				MYLOG_WM(time.time,0, << "got ANGLES of " << TEAM_STR[t] << i
						<< " body= " << RAD2DEG(p->body_angle.get_value_mPI_pPI())
						<< " neck= " << RAD2DEG(p->neck_angle.get_value_mPI_pPI()));
			}
		}
	MYLOG_WM(time.time, 0, << " ***************  WMstate::show_object_info <<<<<<<<<<<<<<<<<<<< end " );
}

void WMstate::export_mdpstate(MDPstate & mdp) const {
	mdp.play_mode= server_playmode_to_player_playmode( play_mode );

	double KONV= determine_KONV( mdp.play_mode );
	WM::last_export_KONV= KONV;

	if ( his_team[0].alive && his_team[0].time_pos.time >= time.time - 2 ) {
		MYLOG_WM(time.time,0,"!!!!!!!! exporting nearest player");
		const _wm_player & me= my_team[WM::my_number];
		Vector tmp= his_team[0].pos - me.pos;
		mdp.closest_attacker.dir= tmp.arg();
		mdp.closest_attacker.dist= tmp.norm();
		mdp.closest_attacker.number= -1;
		mdp.closest_attacker.seen_at= his_team[0].time_pos.time;
		mdp.closest_attacker.number_of_sensed_opponents= 1;
	}
	else {
		//just don't use the closest attacker information setting him to dist 1000.0
		mdp.closest_attacker.dir= 0;
		mdp.closest_attacker.dist= 1000.0;
		mdp.closest_attacker.number= -1;
		mdp.closest_attacker.seen_at= his_team[0].time_pos.time;
		mdp.closest_attacker.number_of_sensed_opponents= 0;
	}

	//<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

	mdp.StateInfoString[0] = '\0'; //no longer needed!

#if 0
	struct _teamcomm_statistics {
		WMTime send_time_of_last_teamcomm;
		WMTime recv_time_of_last_teamcomm;
		int    sender_of_last_teamcomm;
	};
#endif

	mdp.last_message.heart_at=  teamcomm_statistics.recv_time_of_last_teamcomm.time;
	mdp.last_message.from=      teamcomm_statistics.sender_of_last_teamcomm;
	//last_message.sent_at and last_successful_say_at conincide
	mdp.last_message.sent_at=   teamcomm_statistics.send_time_of_last_teamcomm.time;
	mdp.last_successful_say_at= teamcomm_statistics.send_time_of_last_teamcomm.time;

	mdp.time_current= time.time;
	mdp.time_of_last_update= time_of_last_msg_see.time;

	mdp.my_team_score= my_score;
	mdp.his_team_score= his_score;

	mdp.my_goalie_number = my_goalie_number;
	mdp.his_goalie_number = his_goalie_number;
	//cout << "\n$$$$ my_number= " << WM::my_number << " mdp.his_goalie_number= " << mdp.his_goalie_number;

	mdp.view_quality = view_quality;
	mdp.view_angle= view_width;

	double prob= time_diff_ball_to_mdpstate_probability( time.time - ball.time_pos.time );
	mdp.ball.pos_x= FVal( KONV * ball.pos.getX(), prob, ball.time_pos.time );
	mdp.ball.pos_y= FVal( KONV * ball.pos.getY(), prob, ball.time_pos.time );
	mdp.ball.vel_x= FVal( KONV * ball.vel.getX(), prob, ball.time_vel.time );
	mdp.ball.vel_y= FVal( KONV * ball.vel.getY(), prob, ball.time_vel.time );

	for (int team_id= 0; team_id <2; team_id++) {
		FPlayer * mdp_t = mdp.my_team;
		const _wm_player * wm_t = my_team;
		int which_team= my_TEAM;
		if (team_id>0) {
			mdp_t= mdp.his_team;
			wm_t= his_team;
			which_team= his_TEAM;
		}

		for (int i= 1; i< NUM_PLAYERS+1; i++) {
			FPlayer & mdp_p = mdp_t[i-1];
			const _wm_player & wm_p = wm_t[i];

			mdp_p.alive= wm_p.alive;
			mdp_p.number= i;

			prob= time_diff_player_to_mdpstate_probability(team_id, time.time - wm_p.time_pos.time );
			if (team_id == his_TEAM && i == his_goalie_number)
				prob= 1; //opponents goalie never expires, it's is better to know wrong goalie pos, then to assume there is no goalie at all
			mdp_p.pos_x= FVal( KONV * wm_p.pos.getX(), prob, wm_p.time_pos.time );
			mdp_p.pos_y= FVal( KONV * wm_p.pos.getY(), prob, wm_p.time_pos.time );
			mdp_p.vel_x= FVal( KONV * wm_p.vel.getX(), prob, wm_p.time_pos.time );
			mdp_p.vel_y= FVal( KONV * wm_p.vel.getY(), prob, wm_p.time_pos.time );
			mdp_p.stamina = FVal( wm_p.stamina, prob,wm_p.time_pos.time );

			if ( WM::my_number == i && my_TEAM == which_team ) {
				mdp.me= &mdp_p;
				mdp_p.ang= FVal( my_angle.get_value() , 1.0 , time.time );
				mdp_p.neck_angle= FVal( my_neck_angle.get_value() , 1.0, time.time );
				mdp_p.effort = FVal( my_effort , 1.0, time.time );
				mdp_p.recovery = FVal( my_recovery , 1.0, time.time );
				//MYLOG_WM(time.time,0,"EXPORTING MY STAMINA " << mdp_p.stamina.v);
			}
			else {
				mdp_p.ang= FVal( wm_p.body_angle.get_value() , 1.0 , wm_p.time_angle.time );
				mdp_p.neck_angle= FVal( wm_p.neck_angle.get_value() , 1.0, wm_p.time_angle.time );
				mdp_p.effort = FVal( 0.0,0.0,-10 ); //information not in avaiable
				mdp_p.recovery = FVal( 0.0,0.0,-10 );
			}
			//VERY IMPORTANT !!!
			if ( KONV < 0 ) { //convert angles if I play on the right side
				mdp_p.ang.v+= PI; if ( mdp_p.ang.v > 2*PI) mdp_p.ang.v -= 2*PI;
				mdp_p.neck_angle.v+= PI; if ( mdp_p.neck_angle.v > 2*PI) mdp_p.neck_angle.v -= 2*PI;
			}
		}
	}
}

double WMstate::determine_KONV(int user_space_play_mode) const {
	//KONV bewirkt Transformation von Koordinaten, falls der Spieler auf der rechten Seite spielt
	//oder falls der penalty modus aufs eigene Tor gelost wurde!
	if ( penalty_side == unknown_SIDE )
		return WM::my_side == left_SIDE ? 1.0 : -1.0;

	if ( user_space_play_mode == PM_his_BeforePenaltyKick || user_space_play_mode == PM_his_PenaltyKick ) {
		/* during the penalties of the opponent, we have to convert the coordinats, such that
       our goalie is defending the correct goal. Because the other players are inactive during this period
       they just get switched too */

		if ( penalty_side == right_SIDE )
			return -1.0;

		return 1.0;
	}

	/* here we have to switch the sides if the penalty side is left_SIDE
     this is also done for the goalie, because during our own penalty he cold be a possibly
     shooter (and on the other hand, he has never to defend in this mode!)
	 */
	if ( penalty_side == left_SIDE )
		return -1.0;

	return 1.0;
}

void WMstate::check_statistics() const {
	if ( time.time % 500 != 499 || time.time <= 0 )
		return;  //don't check in every cycle,

	int num_alive_teammates= 0;
	for (int i=1 ; i<= NUM_PLAYERS; i++)
		if ( my_team[i].alive )
			num_alive_teammates++;

	if ( num_alive_teammates > 1 &&  teamcomm_statistics.pass_info_count <= 0 ) {
		ERROR_OUT << ID << "\nthere seems to a problem with the communication:"
				<< "\n  teamcomm_count= " << teamcomm_statistics.teamcomm_count
				<< "  pass_info_count= " << teamcomm_statistics.pass_info_count << std::flush;
	}

	if ( teamcomm_statistics.teamcomm_partial_count > 0 )
		WARNING_OUT << ID << "\nI still get partial teamcomm (count= " << teamcomm_statistics.teamcomm_partial_count << ")";

	if ( his_goalie_number <= 0 )
		WARNING_OUT << ID << "\nI still don't know his_goalie_number";


	INFO_OUT << ID << "\nStatus and statistics at time " << time.time;

	if ( (time.time == 499 || time.time == 5999) && teamcomm_statistics.pass_info_count > 0 ) {
		INFO_STREAM << "\nI hear team communication:"
				<< "\n  teamcomm_count= " << teamcomm_statistics.teamcomm_count
				<< "  pass_info_count= " << teamcomm_statistics.pass_info_count << std::flush;
	}

	INFO_STREAM << "Playertypes\n  my_team= ";
	for (int i=1; i<=NUM_PLAYERS; i++)
		INFO_STREAM << " " << i << ":" << my_team[i].type;
	INFO_STREAM << "\n his_team= ";
	for (int i=1; i<=NUM_PLAYERS; i++)
		INFO_STREAM << " " << i << ":" << his_team[i].type;
	INFO_STREAM << std::flush;
}

void WMstate::export_unknown_player(Vector pos, Player & wstate_p, double KONV ) const {
	wstate_p.time= time.time;
	wstate_p.age= 0;
	wstate_p.age_vel= 10;
	wstate_p.age_ang= 10;
	wstate_p.pos.setX(KONV * pos.getX());
	wstate_p.pos.setY(KONV * pos.getY());
	wstate_p.vel.setX(0.0);
	wstate_p.vel.setY(0.0);
	wstate_p.tackle_flag= false;
        wstate_p.tackle_time= -ServerOptions::tackle_cycles;
        wstate_p.action_time= -1;
	wstate_p.kick_flag= false;
	wstate_p.pointto_flag= false;
	wstate_p.pass_info.valid= false;
	wstate_p.alive= true;   //it should be always true at this point anyway, but just to be sure!
	wstate_p.ang= ANGLE(0);
	wstate_p.neck_ang= ANGLE(0);
	wstate_p.neck_ang_rel= ANGLE(0);
	wstate_p.effort = 1.0; //information not avaiable, assume standard type
	wstate_p.recovery = 0.0;
	wstate_p.stamina_capacity = ServerOptions::stamina_capacity; //information not avaiable, assume standard capacity < JTS change to use ServerOptions
	wstate_p.red_card = false;
	wstate_p.yellow_card = false;

	//VERY IMPORTANT !!!
	if ( KONV < 0 ) { //convert angles if I play on the right side
		wstate_p.ang += ANGLE(PI);
		wstate_p.neck_ang+= ANGLE(PI);
		if (wstate_p.pointto_flag)
			wstate_p.pointto_dir+= ANGLE(PI);
		//no need to change the rel angle here !!!
	}

	//type settings
	//Msg_player_type const* player_type= ServerParam::get_player_type( 0 );
	Msg_player_type const* player_type= ServerParam::get_worst_case_opponent_type();
	if (player_type) {
		wstate_p.radius           = player_type->player_size;
		//wstate_p.speed_max        = player_type->player_speed_max;
		wstate_p.speed_max        = player_type->real_player_speed_max;
		wstate_p.dash_power_rate  = player_type->dash_power_rate;
		wstate_p.decay            = player_type->player_decay;
		wstate_p.stamina_inc_max  = player_type->stamina_inc_max;
		wstate_p.inertia_moment   = player_type->inertia_moment;
		wstate_p.stamina_demand_per_meter= player_type->stamina_demand_per_meter;

		wstate_p.kick_radius      = player_type->kickable_margin + wstate_p.radius + ServerOptions::ball_size;
		wstate_p.kick_rand_factor = player_type->kick_rand;

		wstate_p.effort = player_type->effort_max; //information not avaiable, assume worst case
	}
}

void WMstate::export_ws_player( int team, int number, Player & wstate_p, double KONV ) const {
	const WMstate::_wm_player * wm_p_pointer= const_ref_player(team,number);
	if ( ! wm_p_pointer ) {
		ERROR_OUT << ID << "\nno such player";
		return;
	}


	if ( team == my_TEAM )
		wstate_p.team= MY_TEAM;
	else if ( team == his_TEAM )
		wstate_p.team= HIS_TEAM;
	else
		wstate_p.team= UNKNOWN_TEAM;

	const _wm_player & wm_p = *wm_p_pointer;

	wstate_p.alive= wm_p.alive;

	if ( ! wm_p.alive ) {
		if ( WM::my_number == number && team == my_TEAM )
			WARNING_OUT << ID << "\nI'm not alive, but I should be";
		return; //don't need to set anything for an not alive player
	}

	wstate_p.number= number;
	wstate_p.type = wm_p.type;

	wstate_p.time= wm_p.time_pos.time;
	wstate_p.age = time.time - wm_p.time_pos.time;
	wstate_p.age_vel= time.time - wm_p.time_vel.time;
	wstate_p.age_ang= time.time - wm_p.time_angle.time;

	wstate_p.pos.setX(KONV * wm_p.pos.getX());
	wstate_p.pos.setY(KONV * wm_p.pos.getY());
	wstate_p.vel.setX(KONV * wm_p.vel.getX());
	wstate_p.vel.setY(KONV * wm_p.vel.getY());
	wstate_p.stamina = wm_p.stamina;


	wstate_p.tackle_flag= wm_p.tackle_flag;
        wstate_p.tackle_time= wm_p.tackle_time;
        wstate_p.action_time= wm_p.action_time;
	wstate_p.kick_flag= wm_p.kick_flag;
	wstate_p.pointto_flag= wm_p.pointto_flag;
	if (wstate_p.pointto_flag)
	{
		wstate_p.pointto_dir = wm_p.pointto_dir;
		// JTS08 always update pointto_age and confidence
		wstate_p.pointto_age = time.time - wm_p.pointto_time;
	}

	wstate_p.direct_opponent_number = wm_p.direct_opponent_number;
	//JTS10 add conflict information as well as stamina cap bounds
	wstate_p.direct_opponent_conflict_number = wm_p.direct_opponent_conflict_number;
	wstate_p.stamina_capacity_bound = wm_p.stamina_capacity_bound;

	if ( !wm_p.pass_info.valid || wm_p.pass_info.time < time.time ) {
		//wm_p.pass_info.valid= false;
		wstate_p.pass_info.valid= false;
	}
	else {
		wstate_p.pass_info.valid= true;
		wstate_p.pass_info.age= time.time - wm_p.pass_info.recv_time;
		wstate_p.pass_info.ball_vel= KONV * wm_p.pass_info.ball_vel;
		wstate_p.pass_info.ball_pos= KONV * wm_p.pass_info.ball_pos;
		wstate_p.pass_info.abs_time= wm_p.pass_info.time;
		wstate_p.pass_info.rel_time= wm_p.pass_info.time - time.time;
	}

	if ( WM::my_number == number && team == my_TEAM ) {
		wstate_p.alive= true;   //it should be always true at this point anyway, but just to be sure!
		wstate_p.number= WM::my_number;
		wstate_p.ang= my_angle;
		wstate_p.neck_ang= my_neck_angle;
		wstate_p.neck_ang_rel= my_neck_angle - my_angle;
		wstate_p.effort = my_effort;
		wstate_p.recovery = my_recovery;
		wstate_p.stamina_capacity = my_stamina_capacity;
	}
	else {
		wstate_p.ang= wm_p.body_angle;
		wstate_p.neck_ang= wm_p.neck_angle;
		wstate_p.neck_ang_rel= wm_p.neck_angle - wm_p.body_angle;
		wstate_p.effort = 1.0; //information not avaiable, assume homogenous case
		wstate_p.recovery = 0.0;
		wstate_p.stamina_capacity = ServerOptions::stamina_capacity; //information not avaiable, assume default value
	}

	//TGpr: begin
	if ( wm_p.pass_request.valid )
	{
		wstate_p.pass_request.valid           = wm_p.pass_request.valid;
		wstate_p.pass_request.pass_in_n_steps = wm_p.pass_request.pass_in_n_steps;
		wstate_p.pass_request.pass_param      = wm_p.pass_request.pass_param;
		wstate_p.pass_request.pass_param_as_angle = (wstate_p.pass_request.pass_param-8)*10;
		wstate_p.pass_request.received_at     = wm_p.pass_request.received_at;
	}
	else
	{
		wstate_p.pass_request.valid = false;
	}
	//TGpr: end

	//JTS 10: insert yellow / red card info and fouls
	wstate_p.yellow_card = wm_p.yellow_card;
	wstate_p.red_card = wm_p.red_card;
	wstate_p.fouled = wm_p.fouled;
	wstate_p.foul_cycles = wm_p.foul_cycles;

	//VERY IMPORTANT !!!
	if ( KONV < 0  ) { //convert angles if I play on the right side
		wstate_p.ang += ANGLE(PI);
		wstate_p.neck_ang+= ANGLE(PI);
		if (wstate_p.pointto_flag)
			wstate_p.pointto_dir+= ANGLE(PI);
		//no need to change the rel angle here !!!
	}

	//type settings
	Msg_player_type const* player_type;
	if ( wm_p.type >= 0 )
		player_type= ServerParam::get_player_type( wm_p.type );
	else
		player_type= ServerParam::get_worst_case_opponent_type();

	if (player_type) {
		wstate_p.radius           = player_type->player_size;
		//wstate_p.speed_max        = player_type->player_speed_max;
		wstate_p.speed_max        = player_type->real_player_speed_max; //this value should be more realistic then the 'player_speed_max' value, which is just an upper bound!
		wstate_p.dash_power_rate  = player_type->dash_power_rate;
		wstate_p.decay            = player_type->player_decay;
		wstate_p.stamina_inc_max  = player_type->stamina_inc_max;
		wstate_p.inertia_moment   = player_type->inertia_moment;
		wstate_p.stamina_demand_per_meter= player_type->stamina_demand_per_meter;
		//JTS 10: new parameter
		wstate_p.foul_detect_probability = player_type->foul_detect_probability;
		wstate_p.kick_power_rate  = player_type->kick_power_rate;
		wstate_p.catchable_area_l_stretch= player_type->catchable_area_l_stretch;
		wstate_p.kick_radius      = player_type->kickable_margin + wstate_p.radius + ServerOptions::ball_size;
		wstate_p.kick_rand_factor = player_type->kick_rand;
		if ( WM::my_number != number || team != my_TEAM ) //TG09
			wstate_p.effort = player_type->effort_max; //information not in avaiable, assume worst case
	}
	else {
		if ( ClientOptions::server_version >= 8.0 )
			ERROR_OUT << ID << "\nunknown type";
	}

	//try to foresee the position of the
	if ( team == his_TEAM && WMoptions::foresee_opponents_positions)
		if ( wstate_p.age > 0 && wstate_p.age < 10 ) {
			int age= 3; //don't foresee more then 3 cycles!
			if (wstate_p.age < age )
				age= wstate_p.age;
			//double decay= wstate_p.decay;
			if ( wstate_p.vel.sqr_norm() > SQUARE(2.0) )
				ERROR_OUT << "something wrong with player velocity";
			else
				for (int i=0; i< age; i++) {
					wstate_p.pos += wstate_p.vel;
					// wstate_p.vel *= decay; // ridi 04: do NO decaying, assume that op. keeps its speed!
				}
		}
}

void WMstate::export_ws(WS & wstate) const {
	wstate.play_mode= server_playmode_to_player_playmode( play_mode );

	double KONV= determine_KONV(wstate.play_mode);
	WM::last_export_KONV= KONV;

	wstate.penalty_count= penalty_count;

	wstate.current_direct_opponent_assignment = current_direct_opponent_assignment; //TGdoaa
	wstate.number_of_direct_opponent_updates_from_coach = number_of_direct_opponent_updates_from_coach; //TGdoa

	wstate.last_message.heart_at=  teamcomm_statistics.recv_time_of_last_teamcomm.time;
	wstate.last_message.from=      teamcomm_statistics.sender_of_last_teamcomm;
	//last_message.sent_at and last_successful_say_at conincide
	wstate.last_message.sent_at=   teamcomm_statistics.send_time_of_last_teamcomm.time;
	wstate.last_successful_say_at= teamcomm_statistics.send_time_of_last_teamcomm.time;

	wstate.time= time.time;
	wstate.time_of_last_update= time_of_last_msg_see.time;
	wstate.ms_time_of_see = ms_time_of_last_msg_see;
	wstate.ms_time_of_see_delay= ms_time_of_last_msg_see_after_sb;

	wstate.my_team_score= my_score;
	wstate.his_team_score= his_score;

	wstate.my_goalie_number = my_goalie_number;
	wstate.his_goalie_number = his_goalie_number;

	wstate.synch_mode_ok = synch_mode_ok;
	wstate.view_quality = view_quality;
	wstate.view_angle= view_width;

	wstate.my_attentionto= my_attentionto;
	//cout << "\nattentionto= " << wstate.my_attentionto;

	//double prob= time_diff_ball_to_mdpstate_probability( time.time - ball.time.time );
	wstate.ball.time= ball.time_pos.time;
	wstate.ball.age = time.time - ball.time_pos.time;
	wstate.ball.age_vel = time.time - ball.time_vel.time;

	wstate.ball.pos.setX(KONV * ball.pos.getX());
	wstate.ball.pos.setY(KONV * ball.pos.getY());
	wstate.ball.vel.setX(KONV * ball.vel.getX());
	wstate.ball.vel.setY(KONV * ball.vel.getY());

	if ( ball.time_pos.after_reset() )
		if ( ball.time_pos.after_reset2() )
			wstate.ball.invalidated= 2;
		else
			wstate.ball.invalidated= 1;
	else
		wstate.ball.invalidated= 0;

	for (int i= 0; i< NUM_PLAYERS+1; i++) { //just to be sure, no empty player is alive
		wstate.my_team[i].alive= false;
		wstate.his_team[i].alive= false;
	}

	wstate.my_team_num= 0;
	for (int i= 1; i< NUM_PLAYERS+1; i++) {
		Player & wstate_p= wstate.my_team[wstate.my_team_num];
		export_ws_player(my_TEAM, i, wstate_p, KONV);
		if ( wstate_p.alive )
			wstate.my_team_num++;
	}

	wstate.his_team_num= 0;
	for (int i= 1; i< NUM_PLAYERS+1; i++) {
		Player & wstate_p= wstate.his_team[wstate.his_team_num];
		export_ws_player(his_TEAM, i, wstate_p, KONV);
		if ( wstate_p.alive )
			wstate.his_team_num++;
	}

	if ( !msg.valid || msg.time < time.time ) {
		wstate.msg.valid= false;
	}
	else
		wstate.msg= msg;

	//if ( his_team[0].alive && his_team[0].time.time >= time.time - 1 ) {
	if ( his_team[0].alive && his_team[0].time_pos.time == time.time  ) {
		export_ws_player(his_TEAM, 0, wstate.his_team[wstate.his_team_num], KONV);
		wstate.his_team_num++;
	}
#if 0
	for (int i= 1; i< NUM_PLAYERS+1; i++) {
		MYLOG_COMM1(time.time,0,"Exporting MY player "<<i <<" pos: "<<my_team[i].pos
				<<" age: "<<my_team[i].age<<" at time: "<<wstate.time);
	}
#endif

}

Msg_teamcomm2 WMstate::export_msg_teamcomm( const Cmd_Say & say ) const {
    Msg_teamcomm2 tc; //completely rewritten 11.6.02

	int MAX_CAPACITY= 3;
	static const int HIGH_PRIORITY   = 1;
	static const int NORMAL_PRIORITY = 2;
	static const int LOW_PRIORITY    = 3;


#if 1
	if ( say.is_cmd_set() && say.msg_valid() ) {
		say.get_msg(tc.msg);
		MAX_CAPACITY -= 2;  //a small message could still fit into the message
	}
#endif

	/* pass_info and ball_info are exclusive (as long as you can only communicate 10 characters!)
     but this is checked in the encoding method later on!
	 */
#if 1
	//pass info
	if ( say.is_cmd_set() && say.pass_valid() ) {
		if (MAX_CAPACITY >= 2 ) {
			//cout << "\nINNNNNNNNNNNNNN" << std::flush;
			tc.pass_info.valid=  say.get_pass(tc.pass_info.ball_pos,
					tc.pass_info.ball_vel,
					tc.pass_info.time);
			tc.pass_info.ball_pos *= WM::last_export_KONV;   //always convert 'user space' data to the 'world model' coordinates
			tc.pass_info.ball_vel *= WM::last_export_KONV;   //always convert 'user space' data to the 'world model' coordinates
			tc.pass_info.time -= time.time; //reduce to relative time
			if ( tc.pass_info.valid )
				MAX_CAPACITY -= 2; //no more then one additional object is possible
		}
		else {
			//WARNING_OUT << ID << "too much to say (pass_info)";
		}
	}
#endif

#if 1
	//ball_holder_info
	if ( say.is_cmd_set() && say.ball_holder_valid() ) {
		if ( MAX_CAPACITY >= 1 ) {
			tc.ball_holder_info.valid= say.get_ball_holder( tc.ball_holder_info.pos );
			tc.ball_holder_info.pos *= WM::last_export_KONV;   //always convert 'user space' data to the 'world model' coordinates
			if ( tc.ball_holder_info.valid )
				MAX_CAPACITY -= 1; //no more then two additional objects are possible
		}
		else {
			//WARNING_OUT << ID << "too much to say (ball_holder_info)";
		}
	}
#endif

#if 1
	//ball info
	if ( say.is_cmd_set() && say.ball_valid() ) {
		if ( MAX_CAPACITY >= 2 ) {
			//cout << "\nINNNNNNNNNNNNNN" << std::flush;
			tc.ball_info.valid=  say.get_ball(tc.ball_info.ball_pos,
					tc.ball_info.ball_vel,
					tc.ball_info.age_pos,
					tc.ball_info.age_vel);

			tc.ball_info.ball_pos *= WM::last_export_KONV;   //always convert 'user space' data to the 'world model' coordinates
			tc.ball_info.ball_vel *= WM::last_export_KONV;   //always convert 'user space' data to the 'world model' coordinates
			if ( tc.ball_info.valid )
				MAX_CAPACITY -= 2; //no more then one additional object is possible
		}
		else {
			//WARNING_OUT << ID << "too much to say (ball_info)";
		}
	}
#endif

	//TGdoa: begin
#if 1
	//direct opponent assignment
	if ( say.is_cmd_set() && say.direct_opponent_assignment_valid() )
	{
		MYLOG_COMM1(time.time,0,"Exporting DOA "<<tc.direct_opponent_assignment_info.assignment );
		tc.direct_opponent_assignment_info.valid
		= say.get_direct_opponent_assignment( tc.direct_opponent_assignment_info.assignment );
	}
#endif
	//TGdoa: end

	//TGpr: begin
#if 1
	//transfer pass request from say message to team comm object
	if ( say.is_cmd_set() && say.pass_request_valid() )
	{
		if(WSinfo::ws && WSinfo::ws->time>1) { LOG_POL(0,"Exporting PASS REQUEST!"); }
		MYLOG_COMM1(time.time,0,"Exporting PASS REQUEST!");
		if ( MAX_CAPACITY >= 1 )
		{
			tc.pass_request_info.valid = say.get_pass_request( tc.pass_request_info.pass_in_n_steps,
					tc.pass_request_info.pass_param );
			if ( tc.pass_request_info.valid )
			{
				MAX_CAPACITY -= 1; //pass request is not so large, two more objects are possible
				if(WSinfo::ws && WSinfo::ws->time>1) { LOG_POL(0,"Exporting PASS REQUEST - VALID: "<<tc.pass_request_info.pass_in_n_steps<<", "<<tc.pass_request_info.pass_param); }
				MYLOG_COMM1(time.time,0,"Exporting PASS REQUEST - VALID: "<<tc.pass_request_info.pass_in_n_steps<<", "<<tc.pass_request_info.pass_param);
			}
		}
		else
		{
			WARNING_OUT << ID << "too much to say (pass_request)";
		}
	}
#endif
	//TGpr: end

#if 1
	int num= say.get_players_num();
	if ( num > MAX_CAPACITY )
		num= MAX_CAPACITY;

	for (int i=0; i < num; i++ )  {
		Msg_teamcomm2::_player & p = tc.players[tc.players_num];
		say.get_player(i, p.pos, p.team, p.number);
		if ( p.number < 1 )
			continue;
		if ( p.team == MY_TEAM )
			p.team= my_TEAM;
		else if ( p.team == HIS_TEAM )
			p.team= his_TEAM;
		else
			continue;
		tc.players_num++;
		MAX_CAPACITY--;
		p.pos *= WM::last_export_KONV;   //always convert 'user space' data to the 'world model' coordinates
	}
#endif
	const _wm_player & me = my_team[WM::my_number];

	int candidates_identity[ obj_id_MAX_NUM ];
	int candidates_num= 0;
	int candidates_priority[ obj_id_MAX_NUM ];
	Vector normal_priority_center= ball.pos;
	double normal_priority_sqr_radius= 20.0*20.0;

	//#define DEBUG(XXX) XXX
	//DEBUG( MYLOG_WM(time.time,0, << _2D << C2D(normal_priority_center.x, normal_priority_center.y,20,"ff0000")));

	//collect objects information
	if ( time.time - ball.time_pos.time == 0
			&& SensorParser::pos_in_range13bit(ball.pos) ) {
		candidates_identity[candidates_num]= ball_id;
		candidates_priority[candidates_num]= LOW_PRIORITY;
		if ( me.pos.sqr_distance(ball.pos) < 30.0*30.0 ){
			if ( my_team[WM::my_number].pos.distance( ball.pos ) < 5 &&
					(time.time - teamcomm_times[ball_id]) > 1) //send every second cycle
				candidates_priority[candidates_num]= HIGH_PRIORITY;
			else
				candidates_priority[candidates_num]= NORMAL_PRIORITY;
		}
		candidates_num++;
	}

	for (int t= 0; t<2; t++)
		for (int i= 1; i< NUM_PLAYERS+1; i++) {
			const _wm_player * wm_p = const_ref_player(t,i);
			if ( ! wm_p->alive )
				continue;


			if ( ! SensorParser::pos_in_range13bit(wm_p->pos) )
				continue;  //player position cannot be encoded (out of range!)

			int time_diff= time.time - wm_p->time_pos.time;
			//DEBUG( MYLOG_WM(time.time,0, << "team= " << t << " num= " << i  << " time_diff= " << time_diff));
			if (time_diff > 0) //don't send too old information
				continue;

			candidates_identity[candidates_num]= get_obj_id(t,i);
			candidates_priority[candidates_num]= LOW_PRIORITY; //init value
			if ( normal_priority_center.sqr_distance( wm_p->pos ) < normal_priority_sqr_radius ) {
				if ( i == his_goalie_number && t != my_TEAM )
					candidates_priority[candidates_num]= HIGH_PRIORITY;
				else
					candidates_priority[candidates_num]= NORMAL_PRIORITY;
			}
			candidates_num++;
		}
	//all potential candidates are now collected

	if (candidates_num == 0)
		return tc;

	//DEBUG( MYLOG_WM(time.time,0, << "number candidates= " << candidates_num));

	//sort out best candidates
	for (int i=0; i< candidates_num; i++) { //put the best candidates to the begin of the array (using just priority)
		for (int j=i+1; j< candidates_num; j++)
			if ( candidates_priority[j] < candidates_priority[i] ) {
				int dum;
				dum= candidates_identity[i];
				candidates_identity[i]= candidates_identity[j];
				candidates_identity[j]= dum;
				dum= candidates_priority[i];
				candidates_priority[i]= candidates_priority[j];
				candidates_priority[j]= dum;
			}
		if ( i >= MAX_CAPACITY &&
				candidates_priority[i] != candidates_priority[i-1] ) { //don't sort the whole array, if you just need the best three (or more if they are equivalent by priority)
			candidates_num= i;
			break;
		}
	}


	if ( candidates_priority[0] == LOW_PRIORITY ) //don't send if all information has low quality
		return tc;

	//DEBUG( MYLOG_WM(time.time,0, << "number candidates2= " << candidates_num));
	//now further reduce the number of objects, take the priority and time of last sending into account
	if ( candidates_num > MAX_CAPACITY ) {
		int beg= 0;
		int end= 0;
		while ( beg < MAX_CAPACITY ) {
			while ( end < candidates_num && candidates_priority[beg] == candidates_priority[end] )
				end++;

			if ( end > MAX_CAPACITY ) {
				for (int i=beg; i < end; i++) //sort candidates of same priority with according to the time of last sending
					for (int j= i+1; j < end; j++) {
						int time_i= teamcomm_times[ candidates_identity[i] ];
						int time_j= teamcomm_times[ candidates_identity[j] ];
						if ( time_j < time_i ) {  //prefer objects which were not sent for a longer period
							int dum= candidates_identity[i];
							candidates_identity[i]= candidates_identity[j];
							candidates_identity[j]= dum;
							//no need to change the priority entries here, because they won't be used anymore!
						}
					}
				//maby some randomization could be done for objects with same time and priority
				// ...
			}
			beg= end;
		}
		candidates_num= MAX_CAPACITY;
	}

	if ( candidates_num < MAX_CAPACITY ) {
		//maybe I should break here, to not send data, if I'm not full with information
	}

	//
	for (int i=0; i < candidates_num; i++) {
		teamcomm_times[ candidates_identity[i] ]= time.time;

		if ( candidates_identity[i] == ball_id ) {
			tc.ball.valid= true;
			tc.ball.pos= ball.pos;
			MYLOG_WM(time.time,0, << _2D << VC2D(ball.pos,2,"00ffff"));
			continue;
		}
		const _wm_player * wm_p = const_ref_player( candidates_identity[i] );
		Msg_teamcomm2::_player & p = tc.players[tc.players_num];
		p.pos= wm_p->pos;
		p.number= get_obj_number( candidates_identity[i] );
		p.team=   get_obj_team( candidates_identity[i] );
		MYLOG_WM(time.time,0, << _2D << VC2D(p.pos,2,"00ffff"));
		tc.players_num++;
	}

	return tc;
}

void WMstate::export_msg_teamcomm(Msg_teamcomm & tc) const {
	const int max_age_of_object_to_be_sent= 20;
	//const int max_dist_to_object_to_be_sent= 50.0;
	tc.reset();

	tc.side= WM::my_side;
	tc.time= time.time;
	tc.time_cycle= time.cycle;

	tc.from= WM::my_number;

	if ( his_goalie_number != 0) {
		tc.his_goalie_number_upd= true;
		tc.his_goalie_number= his_goalie_number;
	}

	int time_diff= time.time - ball.time_pos.time;
	if ( time_diff <= max_age_of_object_to_be_sent ) {
		tc.ball_upd= true;
		tc.ball.how_old= time_diff;
		tc.ball.x= ball.pos.getX();
		tc.ball.y= ball.pos.getY();
		tc.ball.vel_x= ball.vel.getX();
		tc.ball.vel_y= ball.vel.getY();
	}

	for (int t= 0; t<2; t++)
		for (int i= 1; i< NUM_PLAYERS+1; i++) {
			const _wm_player * wm_p = const_ref_player(t,i);
			if ( !wm_p->alive )
				continue;

			time_diff= time.time - wm_p->time_pos.time;
			if (time_diff > max_age_of_object_to_be_sent) //don't send too old information
				continue;

			if ( tc.players_num > tc.players_MAX ) //to much players (should never happen)
				break;

			Msg_teamcomm::_tc_player & tc_p= tc.players[tc.players_num];
			tc.players_num++;

			tc_p.how_old= time_diff;
			tc_p.team= t;
			tc_p.number= i;
			tc_p.x= wm_p->pos.getX();
			tc_p.y= wm_p->pos.getY();
		}
}

//nur fuer debug zwecke (sehr kurzfristig)
//extern WMstate wmfull;

void WMstate::incorporate_cmd(const Cmd & cmd) {
	if ( cmd.cmd_view.is_cmd_set() )
		cmd.cmd_view.get_angle_and_quality( next_cycle_view_width,
				next_cycle_view_quality );
	//TGdoa: begin
	//manipulate the world model contents now!
	if ( cmd.cmd_say.is_cmd_set() )
		if ( cmd.cmd_say.direct_opponent_assignment_valid() )
			cmd.cmd_say.get_direct_opponent_assignment( current_direct_opponent_assignment );
	//TGdoa: end
}

void WMstate::incorporate_cmd_and_msg_sense_body(const Cmd & cmd, const Msg_sense_body & sb) {
	Vector tmpVec;
	double tmp, tmp2;
	ANGLE ang;
	bool collision = false; // JTS reset collision flag

	if ( ball.time_pos.after_reset() )
		ball.time_pos.reset2();

	if (time.time + 1 == sb.time) {
		time.time= sb.time;
		time.cycle= 0;
	}
	else if (time.time == sb.time)
		time.cycle++;
	else {
		ERROR_OUT << ID << "\n Last wmstate.time= " << time << " , sense_body.time= " << sb.time;
		time.time= sb.time;
		time.cycle= 0;
	}
	time.total_cycle++; //debug


	my_attentionto= sb.focus_target;

#if 0
	if (my_attentionto > 0) {
		my_attentionto_duration++;
		if (my_attentionto_duration > 100 && play_mode == PM_play_on)
			ERROR_OUT << ID << "\ntoo long my_attention_duration = " << my_attentionto_duration;
	}
	else
		my_attentionto_duration= 0;
#endif

	_wm_player & me = my_team[WM::my_number];


	//me_and_ball.my_move is initialized below
	me_and_ball.probable_collision= false;
	me_and_ball.my_old_pos= me.pos;

	if ( WMoptions::use_pfilter && time.time == 0 )
		pfilter.init_pset(me.pos);

	if ( cmd.cmd_body.is_cmd_set() ) {
		const Cmd_Body & cmd_main= cmd.cmd_body;
		Vector pos;
		ANGLE xxx;
		switch ( cmd_main.get_type()) {
		case Cmd_Body::TYPE_MOVETO   :
			MYLOG_WM(sb.time, 0, << " MOVETO TO " << me.pos << " -> " << pos);
			//if (move_count + 1 == sb.move_count) {
			double tmpX, tmpY;
			cmd_main.get_moveto(tmpX,tmpY);
			pos.setX(tmpX);
			pos.setY(tmpY);

			//if ( WM::my_side == right_SIDE ) //moveto is always team relative, so here it must be matched back!
			//  pos *= -1;
			pos *= WM::last_export_KONV;  //moveto is always team relative, so here it must be matched back!
			if ( pos.distance(me.pos) > 1 ) {
				ball.time_pos.reset(); // don't suppose you still know where the ball is
				ball.time_vel.reset();
				// maybe the same should be done for the players !?
			}
			me.pos= pos;
			me.vel= Vector(0);
			if ( WMoptions::use_pfilter )
				pfilter.init_pset(me.pos);
			//}
#if 0
			else
				MYLOG_WM_ERR(time.time,0,
						<< "Server missed a MOVE"
						<< " move_count= " << move_count << "[" << sb.move_count << "]"
						<< ", Player = " << WM::my_number << ", Time = [" << sb.time -1 << "," << sb.time << "]");
#endif
			break;
		case Cmd_Body::TYPE_TURN 	 :
			if (turn_count + 1 == sb.turn_count) {
				cmd_main.get_turn(tmp);
				ang= ANGLE(tmp);
				MYLOG_WM(time.time,0, << " INCORPORATING A TURN " << tmp << " my Angle before= " << my_angle.get_value() << ", my_vel= " << my_speed_value );
				xxx= my_angle;
				my_angle += ANGLE( ang.get_value_mPI_pPI()/(1.0 + WM::my_inertia_moment * my_speed_value ));
				xxx += ANGLE( ang.get_value_mPI_pPI()/(1.0 + WM::my_inertia_moment * sb.speed_value ));
				MYLOG_WM(time.time,0, << " INCORPORATING A TURN adding my Angle after= " << my_angle.get_value() << " xxx= " << xxx.get_value(); );
			}
			else {
				LOG_WM_ERR(time.time,0,
						<< "Server missed a TURN"
						<< ", Player = " << WM::my_number
						<< ", Time = [" << sb.time -1 << "," << sb.time << "]");
			}
			break;
#ifdef WMFULL
			//testing angle prediction
			if (wmfull.time == time) {
				double a= my_angle.diff(wmfull.my_angle);
				if ( a* 180.0 / M_PI > 3) {
					MYLOG_WM(time.time,0, " \n(ERRqOR) WRONG TURN PREDICTION " << a* 180.0 / M_PI);
					//cerr << "\nWRONG ANGLE PREDICTION " << a* 180.0 / M_PI;
				}
			}
#endif
		case Cmd_Body::TYPE_DASH 	 :
		{
			if (dash_count + 1 == sb.dash_count)
			{
				Angle dash_dir = 0.0;//for V13++
				cmd_main.get_dash(tmp, dash_dir);
				MYLOG_WM(time.time,0, << "INCORPORATING A DASH " << tmp);
				if (tmp > ServerOptions::maxpower) tmp= ServerOptions::maxpower;
				if (tmp < ServerOptions::minpower) tmp= ServerOptions::minpower;
				if (tmp> 0 && tmp> me.stamina) tmp= me.stamina;
				if (tmp< 0 && -2*tmp > me.stamina) tmp= -me.stamina;

				if (ClientOptions::server_version >= 13.0)
				{

					double dir_as_deg = RAD2DEG(dash_dir);
					dir_as_deg =   ServerOptions::dash_angle_step
							* rint(dir_as_deg / ServerOptions::dash_angle_step);  //discretization
					double dir_rate = ( fabs( dir_as_deg ) > 90.0
							? ServerOptions::back_dash_rate
									- ( ( ServerOptions::back_dash_rate - ServerOptions::side_dash_rate )
											* ( 1.0 - ( fabs( dir_as_deg ) - 90.0 ) / 90.0 ) )
											: ServerOptions::side_dash_rate + ( ( 1.0 - ServerOptions::side_dash_rate )
													* ( 1.0 - fabs( dir_as_deg ) / 90.0 ) )
					);
					ANGLE dir_rueck( DEG2RAD(dir_as_deg) );
					tmp *= dir_rate;

					tmp *= my_effort;
					tmp *= WM::my_dash_power_rate;
					tmpVec.init_polar(tmp, (my_angle+dir_rueck).get_value() );
					me.vel += tmpVec;
				}
				else
				{
					tmp *= my_effort;
					tmp *= WM::my_dash_power_rate;
					tmpVec.init_polar(tmp, my_angle.get_value() );
					me.vel += tmpVec;
				}
				//check for normalizing to dash max speed
				tmp= me.vel.norm();
				if (tmp > ServerOptions::player_speed_max) {
					MYLOG_WM(time.time,0, << " NORM IN DASH= " << tmp);
					me.vel.normalize(ServerOptions::player_speed_max);
					MYLOG_WM(time.time,0, << " AFTER CORRECTING " << me.vel.norm() );
				}
			}
			else {
				LOG_WM_ERR(time.time,0,
						<< "Server missed a DASH"
						<< ", Player = " << WM::my_number
						<< ", Time = [" << sb.time -1 << "," << sb.time << "]");
			}
			break;
		}//end of dash case

		case Cmd_Body::TYPE_KICK 	 :
			if (kick_count + 1 == sb.kick_count) {
				//cout << "\nKICK KICK KICK KICK KICK KICK";
				cmd_main.get_kick(tmp,tmp2);
				if (tmp > ServerOptions::maxpower) tmp= ServerOptions::maxpower;
				if (tmp < ServerOptions::minpower) tmp= ServerOptions::minpower;
				if ( me.pos.distance( ball.pos ) > WM::my_kick_radius )
					break;

				MYLOG_WM(time.time,0, << "INCORPORATING A KICK " << tmp << " , " << tmp2);

				//ang=  ANGLE( ball.pos.arg() ) - my_angle;
				pos= ball.pos- me.pos;
				ang= ANGLE( pos.arg() ) - my_angle;

				tmp *= ServerOptions::kick_power_rate *
						(
								1.0 - 0.25 * fabs( ang.get_value_mPI_pPI() )/PI
								- 0.25 * (me.pos.distance(ball.pos) - WM::my_radius - ServerOptions::ball_size)/ WM::my_kick_margin
						);

				ang= my_angle + ANGLE(tmp2);
				tmpVec.init_polar( tmp, ang.get_value() );
				ball.vel += tmpVec;
				tmp= ball.vel.norm();
				if (tmp > ServerOptions::ball_speed_max)
					ball.vel.normalize(ServerOptions::ball_speed_max);
			}
			else {
				LOG_WM_ERR(time.time,0,
						<< "Server missed a KICK"
						<< ", Player = " << WM::my_number
						<< ", Time = [" << sb.time -1 << "," << sb.time << "]");
			}
			break;
		case Cmd_Body::TYPE_CATCH    :
			if (catch_count + 1 == sb.catch_count) {
			}
			else {
				LOG_WM_ERR(time.time,0,
						<< "Server missed a CATCH"
						<< ", Player = " << WM::my_number
						<< ", Time = [" << sb.time -1 << "," << sb.time << "]");
			}
			break;
		case Cmd_Body::TYPE_TACKLE    :
			//add some prediction about the following player state //23.05.2002
			break;
		default:
			ERROR_OUT << ID << "\nwrong command";
		}
	}

#if 0
	if ( cmd.cmd_neck.is_cmd_set() ) {
		const Cmd_Neck & cmd_neck= cmd.cmd_neck;
		cmd_neck.get_turn(tmp);
		ang= tmp;

		if ( ang.get_value_mPI_pPI() > ServerOptions::maxneckmoment)
			ang= ServerOptions::maxneckmoment;
		else  if ( ang.get_value_mPI_pPI() < ServerOptions::minneckmoment)
			ang= ServerOptions::minneckmoment;

		my_neck_angle += ang;

		ang = my_neck_angle - my_angle;
		if ( ang.get_value_mPI_pPI() > ServerOptions::maxneckang)
			my_neck_angle= my_angle + ANGLE(ServerOptions::maxneckang);
		else
			if ( ang.get_value_mPI_pPI() < ServerOptions::minneckang)
				my_neck_angle= my_angle - ANGLE(ServerOptions::minneckang);
	}
#endif

	ball.pos += ball.vel;
	ball.vel *= ServerOptions::ball_decay;

	me_and_ball.approx_ball_rel_pos= ball.pos- (me.pos + me.vel);


	Vector me_old_pos= me.pos;

	me.pos += me.vel;
	me_and_ball.my_move= me.vel; //siehe unten fuer evtl. bessere version

	view_quality= sb.view_quality;
	view_width= sb.view_width;
	next_cycle_view_quality= view_quality; //can be changed afer a command was sent
	next_cycle_view_width= view_width;     //can be changed afer a command was sent

	my_stamina_capacity = sb.stamina_capacity;
	if ( play_mode == PM_play_on && view_quality != HIGH ) {
		WARNING_OUT << ID << "\nplay_mode= play_on but view_quality is LOW";
	}
	my_team[WM::my_number].stamina= sb.stamina;
	my_effort= sb.effort;

	my_speed_value= sb.speed_value;
	my_speed_angle= ANGLE(sb.speed_angle);
	my_neck_angle_rel= ANGLE( - DEG2RAD( sb.neck_angle) );
	my_neck_angle= my_angle + my_neck_angle_rel;
	//MYLOG_WM(time.time,0, << " neck log, sb.na= " << sb.neck_angle << " my_na_rel= " << my_neck_angle_rel.get_value() << " my_na_abs= " << my_neck_angle.get_value() << " ma_abs= " << my_angle.get_value() );

	//compute own velocity from speed_value and speed_angle
	//me.vel *= WM::my_decay; //old fashioned ;-)
	tmp= -DEG2RAD( sb.speed_angle ) + my_neck_angle.get_value();
	me.vel.init_polar(sb.speed_value, tmp );
#if 1
	me.pos = me_old_pos + 1.0/WM::my_decay*me.vel; // just a test, but may stay in the code
#endif

	Vector tmp_vec = 1.0/WM::my_decay * me.vel; //assumened my old velocity


	Vector tmp_vec2= -0.1 * me_and_ball.my_move; //computed my old velocity times collision factor


	if ( WMoptions::use_pfilter ) {
		//pfilter.update_with_sensed_movement(tmp_vec);
		pfilter.update_with_sensed_movement(me_and_ball.my_move);
		me.pos = pfilter.get_position();
		//MYLOG_WM(time.time, 0, <<_2D << C2D(me.pos.x, me.pos.y, 0.3, "green"));
		//MYLOG_WM(time.time, 0, << "update_with_sensed_movement called");
	}

	if(   ClientOptions::server_version >= 12
			&& WMoptions::use_server_based_collision_detection )
	{
		collision = sb.coll_ball;
		if (collision)
		{
			MYLOG_WM(time.time,0,<< "BALL: SERVER INFORMED COLLISION ball_dist= "
					<<  ball.pos.distance(me_old_pos)
					<< ", |tmp - tmp2|= |" << tmp_vec << " - " << tmp_vec2 <<"|= "
					<< tmp_vec.distance(tmp_vec2) );
			MYLOG_WM(time.time,0,<<"TG: Updated ball because of COLLISION.");
			timeOfCollision = time.time;
		}
	}
	else {
		//collision check must be done before my own position is updated (therefore me_old_pos)
		//if using protocol version < 12
		collision =  (((ball.pos.distance(me_old_pos) <= WM::my_radius + ServerOptions::ball_size) &&
				me_and_ball.my_move.sqr_norm() <= SQUARE(0.2))
				||
				(ball.pos.distance(me.pos) <= WM::my_kick_radius //>>>
						&& me_and_ball.my_move.sqr_norm() > SQUARE(0.2)
						&& tmp_vec.sqr_distance(tmp_vec2) < SQUARE(0.1)));  //<<< just valid in server version >= 7.10  (collisions always take all objects into account)
	}
	// now check if collision was set
	if ( collision )
	{
		me_and_ball.probable_collision= true;
		MYLOG_WM(time.time,0,<< "BALL: detected COLLISION ball_dist= " <<  ball.pos.distance(me_old_pos) << ", |tmp - tmp2|= |" << tmp_vec << " - " << tmp_vec2 <<"|= " << tmp_vec.distance(tmp_vec2) );
		//TG: LOG_POL(0,<< "BALL: detected COLLISION ball_dist= " <<  ball.pos.distance(me_old_pos) << ", |tmp - tmp2|= |" << tmp_vec << " - " << tmp_vec2 <<"|= " << tmp_vec.distance(tmp_vec2) );
		MYLOG_WM(time.time,1, << _2D << VC2D(ball.pos,0.3,"ff0000"));
		//change ball pos
		MYLOG_WM(time.time,0, << "BALL pos before collision= " << ball.pos << " vel= " << ball.vel);
		MYLOG_WM(time.time,0, << _2D << VC2D( me.pos, WM::my_radius, "00ff00"));
		//ball.pos= compute_object_pos_after_collision(ball.pos,ball.vel,ServerOptions::ball_size, me_old_pos, WM::my_radius);
		ball.pos= compute_object_pos_after_collision(ball.pos,ball.vel,ServerOptions::ball_size, me.pos, WM::my_radius); //if you don't consider the obstacle veolocity
		MYLOG_WM(time.time,0, << "BALL pos after  collision= " << ball.pos << " vel= " << ball.vel);
		//change ball vel!
		ball.vel= compute_object_vel_after_collision(ball.vel);
	}
	else
	{
		MYLOG_WM(time.time,0,<< "BALL: detected NO collision!");
	}
	if (collision)
	{
		MYLOG_WM(time.time,0,<<_2D<<VL2D(ball.pos,ball.pos+10.0*ball.vel,"aaffaa"));
	}

	if ( ( tmp_vec2.sqr_distance( tmp_vec ) < SQUARE(0.1) ) &&
			( me_and_ball.my_move.sqr_distance(tmp_vec) > SQUARE(tmp_vec2.distance(tmp_vec) + 0.2) ) ) { //no collision ?!?
		MYLOG_WM(sb.time,0, << " new method says $$$$ collision, my_move= " << me_and_ball.my_move << " me.vel= " << me.vel);
		//ERROR_OUT << " [t = " << time.time << "]  [p = " << WM::my_number << "] COLLISION DETECTED (NEW METHOD)";
		//me_and_ball.my_move *= 0.8; //assume that the player moved just 0.X so far
	}
	else {
#if LOGGING && WORLD_MODEL_LOGGING
		Vector old= me_and_ball.my_move;
#endif
		me_and_ball.my_move= 1.0/WM::my_decay * me.vel;
		//MYLOG_WM(sb.time,0, << " $$$$ NO collision, my_move= " << me_and_ball.my_move << "(old " << old << ") real= " << me.vel);
		MYLOG_WM(sb.time,0, << " $$$$ NO collision, my_move= " << me_and_ball.my_move << "(old " << old << ") real= " << tmp_vec);
	}

	/*
  if ( me_and_ball.my_move.distance( 1.0/WM::my_decay * me.vel) < 0.4 )  { //no collision ?!?
    Vector old= me_and_ball.my_move;
    me_and_ball.my_move= 1.0/WM::my_decay * me.vel;
    MYLOG_WM(sb.time,0, << " $$$$ NO collision, my_move= " << me_and_ball.my_move << "(old " << old << ") real= " << me.vel);
  }
  else {
    MYLOG_WM(sb.time,0, << " $$$$ collision, my_move= " << me_and_ball.my_move << " me.vel= " << me.vel);
    //me_and_ball.my_move *= 0.8; //assume that the player moved just 0.X so far
    }*/

	//JTS10 : card handling --> NOTE: do we want to rely on foul_cycles to set me.vel ?
	if ( ClientOptions::server_version >= 14 ) {
		me.foul_cycles = sb.foul_cycles;
		me.fouled = sb.fouled;
		if ( sb.card == YELLOW_CARD )
			me.yellow_card = true;
		else if ( sb.card == RED_CARD )
			me.red_card = true;
		else if ( sb.card == NO_CARD ) {
			// prune out card flags
			me.yellow_card = false;
			me.red_card = false;
		}
	}
	MYLOG_WM(sb.time,0, << " $$$$  vel from sense_body= " << me.vel << " norm= " << me.vel.norm() << " arg= " << me.vel.arg() << " speed_value= " << sb.speed_value << " speed_angle= " << sb.speed_angle);

	kick_count= sb.kick_count;
	dash_count= sb.dash_count;
	turn_count= sb.turn_count;
	if (say_count <  sb.say_count) {
		say_count= sb.say_count;
		teamcomm_statistics.send_time_of_last_teamcomm.time= sb.time -1;
		teamcomm_statistics.send_time_of_last_teamcomm.total_cycle= time.total_cycle - 1; //just an approximation
	}
	turn_neck_count= sb.turn_neck_count;
	catch_count= sb.catch_count;
	move_count= sb.move_count;
	change_view_count= sb.change_view_count;

	check_statistics();
}

int get_first_markers_by_distance(int num_of_first, int num_of_all, const Msg_see::_see_marker * from,  int * idx_of_taken) {
	int idx[num_of_all];
	int num_usefull_markers=0;
	for (int i=0 ; i < num_of_all; i++)
		if ( from[i].see_position && from[i].how_many >= 2 ) {
			idx[num_usefull_markers]= i;
			num_usefull_markers++;
		}

	if (num_of_first > num_usefull_markers)
		num_of_first= num_usefull_markers;

	for (int i=0; i< num_of_first; i++)
		for (int j=i+1; j< num_usefull_markers; j++)
			if ( from[idx[j]].dist < from[idx[i]].dist ) {
				int int_tmp=  idx[i];
				idx[i]= idx[j];
				idx[j]= int_tmp;
			}
	for (int i=0; i< num_of_first; i++)
		idx_of_taken[i]= idx[i];

	return num_of_first;
}

void WMstate::update_me_from_msg_see(const Msg_see & see)  {
	MYLOG_WM(time.time, 0, << " POSITION FROM MARKERS (ARTUR)");
	Vector tmpVec;
	_wm_player & me = my_team[ WM::my_number ];

	me.alive= true;
	me.time_pos= time;
	me.time_vel= time;
	me.time_angle= time;

	int best_five[5];
	int best_num= get_first_markers_by_distance(5, see.markers_num, see.markers, best_five);

#if 0
	MYLOG_WM(time.time,0, << " #Markers all= " << see.markers_num << " best= " << best_num );

	for (int i=0; i< best_num; i++)
		MYLOG_WM(time.time,0, << " Best (" << best_five[i] << ") " << see.markers[ best_five[i] ].x << " " << see.markers[best_five[i]].y
				<< " " << see.markers[best_five[i]].dist << " " << see.markers[best_five[i]].dir );
#endif

	//intersect the circles around 2 nearest markers
	if (best_num > 1) {
		MYLOG_WM(time.time,0,"COMPUTE POS: best_num= " << best_num);
		int num_solutions_1st_choice= 0;
		int num_solutions_2nd_choice= 0;
		Vector solutions_1st_choice[25];
		Vector solutions_2nd_choice[25];

		for (int i= 0; i< best_num; i++)
			for (int j= i+1; j< best_num; j++) {
				const Msg_see::_see_marker & m1 = see.markers[ best_five[i] ];
				const Msg_see::_see_marker & m2 = see.markers[ best_five[j] ];

				if ( !m1.see_position || !m2.see_position)
					continue;

				Circle2d c1( Vector(m1.x, m1.y), m1.dist);
				Circle2d c2( Vector(m2.x, m2.y), m2.dist);

				Vector p1,p2;

				//MYLOG_WM(time.time, 1 , << _2D << C2D(m1.x,m1.y,m1.dist, "gray" ) << C2D(m2.x,m2.y,m2.dist, "gray" ));


				int res= Geometry2d::intersect_circle_with_circle(p1,p2,c1,c2);
				if (res == 1) {
					double d1= me.pos.distance(p1);
					double d2= me.pos.distance(p2);

					if ( d1 < 2.0 && (d1 + 7.0  < d2) ) {
						solutions_1st_choice[num_solutions_1st_choice]= p1;
						num_solutions_1st_choice++;
					}
					else if ( d2 < 2.0 && (d2 + 7.0  < d1) ) {
						solutions_1st_choice[num_solutions_1st_choice]= p2;
						num_solutions_1st_choice++;
					}
					else if (d1 < d2) {
						solutions_2nd_choice[num_solutions_2nd_choice]= p1;
						num_solutions_2nd_choice++;
					}
					else {
						solutions_2nd_choice[num_solutions_2nd_choice]= p2;
						num_solutions_2nd_choice++;
					}
				}
				else {
					//no intersection, maybe take the neares point between the 2 circles
				}
			}

		MYLOG_WM(time.time,0,"COMPUTE POS: num_solutions 1st= " << num_solutions_1st_choice << " 2nd= " << num_solutions_2nd_choice);
		if (num_solutions_1st_choice <= 2) { //if there are to few 1st chois solutions, take also the 2nd choice solutions into account
			for (int i=0; i < num_solutions_2nd_choice; i++) {
				solutions_1st_choice[ num_solutions_1st_choice ]= solutions_2nd_choice[ i ];
				num_solutions_1st_choice++;
			}
		}
		if (num_solutions_1st_choice == 1 ) { //only adjust the neck_angle (keep old position)
			const Msg_see::_see_marker & m1 = see.markers[ best_five[best_num-1] ];
			ANGLE dir =  ANGLE(  DEG2RAD(m1.dir) );

			tmpVec = Vector(m1.x,m1.y) - me.pos;
			my_neck_angle= ANGLE( tmpVec.arg() ) + dir;
		}
		if (num_solutions_1st_choice > 1) { //there should be min 2 solutions
			MYLOG_WM(time.time,0,"COMPUTE POS: num_solutions_1st_choice= " << num_solutions_1st_choice);
			Vector weighted_sum= Vector(0,0);
			for (int i= 0; i< num_solutions_1st_choice; i++) {
				//MYLOG_WM(time.time,0, << _2D << C2D(solutions_1st_choice[i].x,solutions_1st_choice[i].y, 0.5, "green"));
				weighted_sum+= solutions_1st_choice[i];
			}
			weighted_sum *= 1/double(num_solutions_1st_choice);
			MYLOG_WM(time.time,1, << _2D << VC2D(weighted_sum, 0.5, "blue"));
			me.pos= weighted_sum;
			MYLOG_WM(time.time,1, << "me.pos " << me.pos << " weighted_sum " << weighted_sum);

			if ( WMoptions::use_pfilter ) {
				//pfilter.update_with_sensed_position(me.pos);
				Vector *v_arr = new Vector [see.markers_num];
				double d_arr[see.markers_num];
				for (int k = 0; k < see.markers_num; k++) {
					v_arr[k] = Vector(see.markers[k].x, see.markers[k].y);
					d_arr[k] = see.markers[k].dist;
				}

				int ret = pfilter.update_with_sensed_markers(see.markers_num, v_arr, d_arr, me.pos);
                delete [] v_arr;
                v_arr = NULL;

				MYLOG_WM(time.time, 0, << "NUMBER OF MARKERS: " << see.markers_num);

				if (ret == 3) { MYLOG_WM(time.time, 0, << "current measurement added as particle!!!!!!!!!!!!!!!!!!"); }

				MYLOG_WM(time.time,1, << "me.pos " << me.pos << " weighted_sum " << weighted_sum);


				Vector dan_pos = pfilter.get_position();

				MYLOG_WM(time.time, 0, << " My position in WMstate is " << me.pos);
				MYLOG_WM(time.time, 0, << " My position from ParticleSet is " << dan_pos);

				MYLOG_WM(time.time, 0, <<_2D << VC2D(me.pos, 0.4, "grey"));
				MYLOG_WM(time.time, 0, <<_2D << VC2D(dan_pos, 0.4, "red"));

				//actually use pfilter!!!!!
				me.pos = dan_pos;
			}

			/* compute now my_neck_angle:
	 take the furthest marker from the set of considered best markers!
	 Because of the fluctuation in the computed own position, it is better
	 if the marker which determines my_neck_angle is as far as possible!
	 (e.g. if the dist is smaller then approx 3 m, then the computed value 
	 will be quite bad!)
			 */
			const Msg_see::_see_marker & m1 = see.markers[ best_five[best_num-1] ];
			ANGLE dir =  ANGLE(  DEG2RAD(m1.dir) );

			tmpVec = Vector(m1.x,m1.y) - me.pos;
			my_neck_angle= ANGLE( tmpVec.arg() ) + dir;
#ifdef WMFULL
			//test predicted angle
			if (wmfull.time == time) {
				double ang_diff= my_neck_angle.diff(wmfull.my_neck_angle);
				if (ang_diff * 180.0/M_PI > 3) {
					MYLOG_WM(time.time,0,<< " MY_NECK_DIFF (DEG) = " << ang_diff * 180.0/M_PI
							<< " my_neck_ang= " << my_neck_angle.get_value() * 180.0/M_PI
							<< " full.my_neck_ang= " << wmfull.my_neck_angle.get_value() * 180.0/M_PI );
					MYLOG_WM(time.time,0,<< " MY_NECK_DIFF CONTINUE how_many " << m1.how_many << " x=" << m1.x << " y=" << m1.y << " dir=" << m1.dir << " dist=" << m1.dist);
				}
			}
#endif
		}
	}
	else {
		MYLOG_WM(time.time,0, << " I see just " << see.markers_num << " markers" );

		if ( ( WMoptions::use_pfilter ) &&
				( see.markers_num >= 1 ) ) {
			//pfilter.update_with_sensed_position(me.pos);
			Vector *v_arr = new Vector [see.markers_num];
			double d_arr[see.markers_num];
			for (int k = 0; k < see.markers_num; k++) {
				v_arr[k] = Vector(see.markers[k].x, see.markers[k].y);
				d_arr[k] = see.markers[k].dist;
			}

			int ret = pfilter.update_with_sensed_markers(see.markers_num, v_arr, d_arr, me.pos);
            delete [] v_arr;
            v_arr = NULL;

			MYLOG_WM(time.time, 0, << "NUMBER OF MARKERS: " << see.markers_num);

			if (ret == 3) { MYLOG_WM(time.time, 0, << "current measurement added as particle!!!!!!!!!!!!!!!!!!"); }

			Vector dan_pos = pfilter.get_position();

			MYLOG_WM(time.time, 0, << " My position in WMstate is " << me.pos);
			MYLOG_WM(time.time, 0, << " My position from ParticleSet is " << dan_pos);

			MYLOG_WM(time.time, 0, <<_2D << VC2D(me.pos, 0.4, "grey"));
			MYLOG_WM(time.time, 0, <<_2D << VC2D(dan_pos, 0.4, "red"));

			//actually use pfilter!!!!!
			me.pos = dan_pos;
		}


	}
}

void WMstate::update_ball_from_msg_see(const Vector & my_pos_before_update, const Msg_see & see) {
	_wm_player & me = my_team[ WM::my_number ];

	if (!see.ball_upd || see.ball.how_many < 2) { //change ball position relative to players position
		Vector my_translation= me.pos - my_pos_before_update;
		if (my_translation.sqr_norm() > SQUARE(1.5*WM::my_kick_radius)) //if my position changed more then usually possible, don't change the position of the ball pos
			return;

		ball.pos+= my_translation;
		MYLOG_WM(time.time,0, << "NO BALL INFO  moved ball to " << ball.pos << " from " << ball.pos - my_translation);
		if (!see.ball_upd && me.pos.sqr_distance(ball.pos) < SQUARE(0.8 * ServerOptions::visible_distance) ) {
			//if you don't see the ball, but it should be in your feel range, then invalidate ball info.
			ball.time_pos.reset();
			ball.time_vel.reset();
			MYLOG_WM(time.time,0,"INVALIDATING BALL because I should feel it, but I do not!!!!");
		}
		return;
	}

	//compute new ball pos and velocity
	ball.time_pos= time;
#ifdef WMFULL
	MYLOG_WM(time.time,0, << "BALL PARAMETER dist= " << see.ball.dist <<  " actual dist " << wmfull.ball.pos.distance(wmfull.my_team[WM::my_number].pos));
#endif
	ball.pos= compute_object_pos(me.pos, my_neck_angle, see.ball.dist, see.ball.dir);

	if ( see.ball.how_many >= 4) {
		//compute new ball velocity
		ball.vel= compute_object_vel(me.pos,me.vel, ball.pos, see.ball.dist, see.ball.dist_change, see.ball.dir_change);
		ball.time_vel= time;
		MYLOG_WM(time.time,0,<<"TG: Updated ball VELOCITY from SEE message.");
	}
#if 1
	else {
		if ( me_and_ball.time.time + 1 == ball.time_pos.time ) {
			bool checkForCollision = true; 							// JTS assume we want to check for collision
			//try to approximate new ball velocity from old ball pos and current ball pos
			Vector ball_rel_pos= ball.pos - me.pos;
			//me_and_ball.my_move= Vector(0.0,0.0); //test don't consider own movement
			Vector approx_ball_vel= ball_rel_pos + me_and_ball.my_move - me_and_ball.old_ball_rel_pos;
			MYLOG_WM(time.time,0, << "APPR_VEL: ball_rel_pos= " <<  ball_rel_pos << ", old_ball_rel_pos= " << me_and_ball.old_ball_rel_pos << ", my_move " << me_and_ball.my_move << " -> " << approx_ball_vel);
			approx_ball_vel *= ServerOptions::ball_decay;
			if (    ClientOptions::server_version >= 12 				// JTS COLLISION FIX !?
					&& WMoptions::use_server_based_collision_detection   // JTS COLLISION FIX !?
			)
			{ // if server_based_collision is set only check for collisions
				// if there actually was one in this timesteps sense_body msg
				if ( timeOfCollision != time.time )
					checkForCollision = false;
			}
			if (   checkForCollision && (see.ball.dist <= 0.3 || (see.ball.dist <= 0.4 && (me_and_ball.probable_collision || me_and_ball.approx_ball_rel_pos.sqr_distance(ball_rel_pos) > SQUARE(0.3) ))) )
			{
				//probably a collision!
				approx_ball_vel= compute_object_vel_after_collision(approx_ball_vel);
			}
			if (   see.ball.dist <= 3.0 || (see.ball.dist <= 6.0 && (approx_ball_vel.sqr_distance(ball.vel) > SQUARE(1.0))) )
			{
#ifdef WMFULL
				//////// test
				double tmp= -1.0;
				if (wmfull.time.time == time.time)
					tmp= wmfull.ball.vel.distance( approx_ball_vel );
				////////
				MYLOG_WM(time.time,0, << "APPR_VEL, acc=" << me_and_ball.my_move << " dist= " << see.ball.dist << " v_diff= " << tmp << " old_v_diff= " << wmfull.ball.vel.distance( ball.vel ) << " pc= " << me_and_ball.probable_collision);
#endif
				ball.vel= approx_ball_vel;
				MYLOG_WM(time.time,0,<<"TG: Updated ball VELOCITY from SEE message (from approx).");
			}
		}
	}
#endif

	//update the me_and_ball entries
	me_and_ball.time= time;
	me_and_ball.old_ball_rel_pos= ball.pos- me.pos;
	//me_and_ball.my_vel= me.vel;

	//print some results
	MYLOG_WM(time.time,0,<< "RESULT REL BALL (num of parameters=" << see.ball.how_many << ") POS= " << ball.pos - me.pos << " VEL= " << ball.vel);
#ifdef WMFULL
	Vector rel_ball_pos= ball.pos - me.pos;
	Vector full_rel_ball_pos= wmfull.ball.pos - wmfull.my_team[WM::my_number].pos;
	if (wmfull.time.time == time.time) {
		MYLOG_WM(time.time,0,<< "ACTUAL REL BALL                         POS= " << full_rel_ball_pos << " VEL= " << wmfull.ball.vel);
		if ( rel_ball_pos.sqr_distance( full_rel_ball_pos) > SQUARE(0.1) )
			MYLOG_WM(time.time,0, << "     BALL BIG POS DIFF= " << rel_ball_pos.distance( full_rel_ball_pos));
		if ( ball.vel.distance( wmfull.ball.vel) > 0.1 )
			MYLOG_WM(time.time,0, << "     BALL BIG VEL DIFF= " << ball.vel.distance( wmfull.ball.vel));
	}
#endif
}

void WMstate::update_players_from_msg_see(const Msg_see & see) {
	const int max_age_to_be_considered= 8;
	const int max_age_to_be_considered_lev2= 15;
	const double max_distance_to_be_matched= 5.0; //seems to be empirically adequate
	const double max_distance_to_be_matched_lev2= 8.0; //seems to be empirically adequate
	const _wm_player & me = my_team[ WM::my_number ];
	PlayerFlagSet players; //keeps all players which can be tracked by matching
	players.set_all();

	players.unset(my_TEAM,WM::my_number); //don't consider myself in later matching

	// remember all players which can be completly identified
	for (int i=0; i< see.players_num; i++) {
		const Msg_see::_see_player & p= see.players[i]; //shortcut
		if ( (p.team == my_TEAM || p.team == his_TEAM) && p.number > 0 )
			players.unset(p.team,p.number);
	}

	unknown_players.num= 0;
	unknown_players.time= time;
	for (int i=0; i< see.players_num; i++) {
		Vector approx_pos;
		Vector approx_vel;

		const Msg_see::_see_player & p= see.players[i]; //shortcut

		if (p.goalie == true && p.number > 0) { //update goalie numbers
			if (p.team == my_TEAM) {
				if (my_goalie_number > 0 && my_goalie_number != p.number)
					ERROR_OUT << ID << "\nmy_goalie_number changed" << std::flush;
				my_goalie_number= p.number;
			}
			else if (p.team == his_TEAM) {
				if (his_goalie_number > 0 && his_goalie_number != p.number)
					ERROR_OUT << ID << "\nhis_goalie_number changed" << std::flush;
				his_goalie_number= p.number;
				//INFO_OUT << ID << "\nsee his goalie number= " << his_goalie_number;
			}
			else {
				//don't know what to do without the team information (probably this case never occurs!)
			}
		}

		if (p.how_many < 2)
			continue; //skip players which don't provide any information about their position

		approx_pos= compute_object_pos(me.pos, my_neck_angle, p.dist, p.dir);

		if ( p.how_many >= 4 ){
			approx_vel= compute_object_vel(me.pos, me.vel, approx_pos, p.dist, p.dist_change, p.dir_change);
			// CHANGED by cijat, TG: top player speed is player_speed_max!
			if(approx_vel.norm()>ServerOptions::player_speed_max)
				approx_vel.normalize(ServerOptions::player_speed_max);
		}
		else
			approx_vel= Vector(0);

		if ( p.team == my_TEAM && p.number == WM::my_number ) {
			ERROR_OUT << ID << "\nseeing me self, this should never happen";
			continue;
		}

		if ( (p.team == my_TEAM || p.team == his_TEAM) && p.number > 0) {
			_wm_player * wm_p= ref_player(p.team,p.number);

			int age_pos= time.time - wm_p->time_pos.time;
			wm_p->alive= true;
			wm_p->time_pos= time;
			wm_p->tackle_flag= p.tackle_flag;
                        if (p.tackle_flag && ((time.time - wm_p->tackle_time) > ServerOptions::tackle_cycles)) wm_p->tackle_time= time.time - 1;
                        if (p.tackle_flag == false) wm_p->action_time= time.time - 1;
			wm_p->kick_flag= p.kick_flag;
			wm_p->pointto_flag= p.pointto_flag;
			//if ( p.pointto_flag ) // JTS 10 --> handle this later
			//wm_p->pointto_dir = ANGLE(DEG2RAD( - p.pointto_dir));

#if 0
			Vector move_dir= approx_pos - wm_p->pos;
#else
			//use here the relative movement of the player
			Vector move_dir= me_and_ball.my_move + ( approx_pos - me.pos ) - (wm_p->pos - me_and_ball.my_old_pos);
#endif

			wm_p->pos= approx_pos;
			wm_p->vel= approx_vel;
			wm_p->unsure_number= false;

			wm_p->pointto_flag= p.pointto_flag;
			if ( p.pointto_flag )
			{
				wm_p->pointto_dir = ANGLE( DEG2RAD( (-p.pointto_dir) ) );
				LOG_WM(time.time, 0, "WMSTATE pointto " << p.number << " "  << (-p.pointto_dir) << " " <<  RAD2DEG(my_neck_angle.get_value_mPI_pPI()));
				wm_p->pointto_time = time.time;
			}
			else
			{
				wm_p->pointto_flag = false;
			}
			if ( p.how_many >= 4 )
				wm_p->time_vel= time;
			else if ( age_pos == 1 ) {
				wm_p->time_vel= time;
				wm_p->vel= move_dir;
			}

			if ( p.how_many == 5 )
				ERROR_OUT << ID << "\ngot 5 parameters";

			if ( p.how_many >= 6 ) { //get also the angle information
				wm_p->time_angle= time;
				wm_p->body_angle= WMTools::conv_server_angle_to_angle_0_2Pi(p.body_dir);
				wm_p->body_angle+= my_neck_angle;
				wm_p->neck_angle= WMTools::conv_server_angle_to_angle_0_2Pi(p.head_dir);
				wm_p->neck_angle+= my_neck_angle;
			}
			else if ( age_pos == 1 && move_dir.sqr_norm() > SQUARE(0.2)) {
				wm_p->time_angle= time;
				wm_p->body_angle= move_dir.ARG(); //just set the angle accorting to the observed movement
			}
		}
		else if (p.team == unknown_TEAM || p.number == 0) {
			//suppose it is an opponent
			double sqr_dist= me.pos.sqr_distance(approx_pos);
			if ( sqr_dist < SQUARE(3.5) && p.team != my_TEAM &&
					(his_team[0].time_pos <= time || me.pos.sqr_distance(his_team[0].pos) > sqr_dist) ) {
				_wm_player * unknown_p= &his_team[0];

				{ // try to find the corresponding opponent or teammate
					int his_res_team, my_res_team;
					int his_res_number, my_res_number;
					double sqr_res_distance_to_his_team;
					double sqr_res_distance_to_my_team;
					bool res_to_my_team = get_nearest_player_to_pos(my_TEAM, approx_pos, players, 5,
							my_res_team, my_res_number, sqr_res_distance_to_my_team);


					bool res_to_his_team= get_nearest_player_to_pos(his_TEAM, approx_pos, players, 5,
							his_res_team, his_res_number, sqr_res_distance_to_his_team);

					if ( res_to_my_team & res_to_his_team ) {
						if ( sqr_res_distance_to_my_team < sqr_res_distance_to_his_team )
							res_to_his_team= false;
						else
							res_to_my_team= false;
					}

					if ( res_to_my_team && sqr_res_distance_to_my_team <= SQUARE(3.0) ) {
						unknown_p= ref_player(my_res_team,my_res_number);
						players.unset(my_res_team,my_res_number); // don't match players twice
						MYLOG_WM(time.time,0,"--> matched NEAREST player: team= " << my_res_team << " num= " << my_res_number << " dist= " << sqrt(sqr_res_distance_to_my_team));
					}

					if ( res_to_his_team && sqr_res_distance_to_his_team <= SQUARE(3.0) ) {
						unknown_p= ref_player(his_res_team,his_res_number);
						players.unset(his_res_team,his_res_number); // don't match players twice
						MYLOG_WM(time.time,0,"--> matched NEAREST player: team= " << his_res_team << " num= " << his_res_number << " dist= " << sqrt(sqr_res_distance_to_his_team));
					}
				}

				unknown_p->alive= true;

				int age_pos= time.time - unknown_p->time_pos.time;
				unknown_p->time_pos= time;
				unknown_p->tackle_flag= p.tackle_flag;
                                if (p.tackle_flag && ((time.time - unknown_p->tackle_time) > ServerOptions::tackle_cycles)) unknown_p->tackle_time= time.time - 1; 
                                if (p.tackle_flag == false) unknown_p->action_time= time.time - 1;
				unknown_p->kick_flag= p.kick_flag;
				unknown_p->pointto_flag= p.pointto_flag;
				if ( p.pointto_flag )
				{
					unknown_p->pointto_dir = ANGLE( DEG2RAD( - p.pointto_dir ) );
					unknown_p->pointto_time = time.time;
				}

#if 0
				Vector move_dir= approx_pos - unknown_p->pos;
#else
				Vector move_dir= me_and_ball.my_move + ( approx_pos - me.pos ) - (unknown_p->pos - me_and_ball.my_old_pos);
#endif

				unknown_p->pos= approx_pos;
				unknown_p->vel= approx_vel;
				if ( age_pos == 1 ) {
					unknown_p->time_vel= time;
					unknown_p->vel= move_dir;
					if (move_dir.sqr_norm() > SQUARE(0.2) ) {
						unknown_p->time_angle= time;
						unknown_p->body_angle= move_dir.ARG();
					}
				}
				MYLOG_WM(time.time,0,"setting HIS_TEAM[0] pos=" << approx_pos << " vel= " << approx_vel << " angle= " << 	unknown_p->body_angle);
				MYLOG_WM(time.time,0, _2D << VC2D(unknown_p->pos,1.5,"ff0000")
						<< VL2D(unknown_p->pos, unknown_p->pos+move_dir*30,"0000ff") << STRING2D(unknown_p->pos.getX()+1, unknown_p->pos.getY(),"UNKNOWN", "ff0000"));

#if 1   //following code or it's implications are not yet really used
				if ( unknown_players.num < unknown_players.max_num ) {
					unknown_players.pos[unknown_players.num]= approx_pos;
					unknown_players.num++;
				}
				else {
					if ( penalty_side == unknown_SIDE ) //during a penalty all players are crowded in the middle circle
						WARNING_OUT << ID << " see to much unknown players ";
				}
#endif
				continue;
			}
			//unknown players will be treated later
			if (p.team == unknown_TEAM)
				continue;

			//here p.team is knownt, but the number is 0!
			//try to track the correct player
			MYLOG_WM(time.time,0,"%%% PLAYER team= " << p.team << " at pos " << approx_pos);
			int res_team;
			int res_number;
			double sqr_res_distance;
			bool res= get_nearest_player_to_pos(p.team, approx_pos, players, max_age_to_be_considered,
					res_team, res_number, sqr_res_distance);
			if (!res)
				continue;

			if ( sqr_res_distance >= SQUARE(max_distance_to_be_matched) ) {
				bool res= get_nearest_player_to_pos(p.team, approx_pos, players, max_age_to_be_considered_lev2,
						res_team, res_number, sqr_res_distance);
				if (!res)
					continue;

				if ( sqr_res_distance >= SQUARE( max_distance_to_be_matched_lev2 ) ) {
					MYLOG_WM(time.time,0,"--> candidate was player: team= " << res_team << " num= " << res_number << " dist= " << sqrt(sqr_res_distance));
#if 0
					const _wm_player * dum= wmfull.const_ref_player(res_team,res_number);
					if ( time.time == wmfull.time.time && dum->pos.sqr_distance( approx_pos ) < SQUARE(3.0) ) {
						MYLOG_WM(time.time,0," CAND OK " << sqrt(sqr_res_distance) << " new dist= " << dum->pos.distance( approx_pos ));
					}
					else
						MYLOG_WM(time.time,0," CAND BAD " << sqrt(sqr_res_distance) << " new dist= " << dum->pos.distance( approx_pos ));
#endif      
					continue;
				}
				MYLOG_WM(time.time,0,<< "LEVEL 2 information");
			}
			_wm_player * wm_p= ref_player(res_team,res_number);
			wm_p->time_pos= time;
			wm_p->tackle_flag= p.tackle_flag;
                        if (p.tackle_flag && ((time.time - wm_p->tackle_time) > ServerOptions::tackle_cycles)) wm_p->tackle_time= time.time - 1;
                        if (p.tackle_flag == false) wm_p->action_time= time.time - 1;
			wm_p->kick_flag= p.kick_flag;
			wm_p->pointto_flag= p.pointto_flag;
			if ( p.pointto_flag )
				wm_p->pointto_dir = ANGLE( DEG2RAD( - p.pointto_dir) );
			wm_p->pos= approx_pos;
			wm_p->vel= approx_vel;
			wm_p->unsure_number= true;

			MYLOG_WM(time.time,0,"--> matched player: team= " << res_team << " num= " << res_number << " dist= " << sqrt(sqr_res_distance));
#if 0 //test how many mismatches happened
			const _wm_player * dum= wmfull.const_ref_player(res_team,res_number);
			if ( time.time == wmfull.time.time && dum->pos.sqr_distance( wm_p->pos ) > SQUARE(2.0) )
				MYLOG_WM(time.time,0," PLAYER WAS MISMATCHED " << sqrt(sqr_res_distance) << " new dist= " << dum->pos.distance( wm_p->pos ));
#endif      
			players.unset(res_team,res_number); // don't match players twice
		}
		else {
			//don't use the information at all, because to few is known about the player
		}
	}
}

void WMstate::handle_inconsistent_objects_in_view_area() {

	//TG08: reset ball and player if a ball holder info has been received
	if (   time.time == teamcomm_statistics.recent_ballholder_info_cycle
			&& teamcomm_statistics.recent_ballholder_number > 0
			&& teamcomm_statistics.recent_ballholder_number < 12)
	{
		MYLOG_COMM1(time.time,0,<<"TG08: Received ball holder information from teammate "<<teamcomm_statistics.recent_ballholder_number<<"this cycle.");
		_wm_player & ballHolder = my_team[ teamcomm_statistics.recent_ballholder_number ];
		if (    ballHolder.pos.sqr_distance( ball.pos ) > SQUARE(WM::my_kick_radius)
				&& ballHolder.pos.sqr_distance( ball.pos ) < SQUARE(WM::my_kick_radius + 1.0 ) )
		{
			MYLOG_COMM1(time.time,0,<<"TG08: I ("<<WM::my_number<<") shift ball (dist="<<ballHolder.pos.distance( ball.pos )<<") from "<<ball.pos<<" with vel="<<ball.vel<<" towards the ball holder!");
			ball.pos = ball.pos - ballHolder.pos;
			ball.pos.normalize(0.7);
			ball.pos += ballHolder.pos;
			ball.vel.setX(0.0);
			ball.vel.setY(0.0);
			MYLOG_COMM1(time.time,0,<<"TG08: New ball position is "<<ball.pos);
		}
	}
	//TG08: end

	if ( view_quality == LOW || next_cycle_view_quality == LOW ) //most object are not seen with this quality
		return;

	_wm_player & me = my_team[ WM::my_number ];

	int act_view_width= view_width;
	if ( act_view_width != next_cycle_view_width ) { //assume the smallest width
		switch (next_cycle_view_width) {
		case NARROW:
			act_view_width= NARROW;
			break;
		case NORMAL:
			if (act_view_width != NARROW)
				act_view_width= NORMAL;
			break;
		}
	}

	Vector my_shifted_pos;
	my_shifted_pos.init_polar(2.0, my_neck_angle.get_value());
	my_shifted_pos+= me.pos; //shifting towards neck_angle increases tolerance in in_view_area

	show_view_area(my_shifted_pos, my_neck_angle, act_view_width);
	MYLOG_WM(time.time,0, << "MY_AGE= " << me.time_pos << "BALL_POS_AGE= " << ball.time_pos);
	if ( ball.time_pos != time //the time MUST NOT be the same!!!
			&& (in_view_area(my_shifted_pos, my_neck_angle, act_view_width, ball.pos) || in_feel_area(me.pos,ball.pos,-0.2) )
			&& (time.time - ball.time_pos.time) > 1 //padua2003, not sure
	) {
		ball.time_pos.reset(); //invalidate ball pos
		ball.time_vel.reset(); //invalidate ball pos
		MYLOG_WM(time.time,0, << "DONT SEE THE BALL, BUT I SHOULD");
		//show_view_area(my_shifted_pos, my_neck_angle, act_view_width);
		MYLOG_WM(time.time,1, << _2D << VC2D(ball.pos,5, "ff0000"));
	}


	for (int t=0; t < 2; t++) {
		double dist;
		if (t == my_TEAM)
			dist= 35;
		else
			//dist= 25;
			dist= 35; //art03 for go2003

		for (int i=0; i < NUM_PLAYERS + 1; i++) {
			if (t== my_TEAM && i== WM::my_number)
				continue;

			_wm_player * p= ref_player(t,i);

			if ( ! p->alive )
				continue;

			if ( p->time_pos != me.time_pos
					&& me.pos.sqr_distance(p->pos) <= SQUARE(dist)
					&& in_view_area(my_shifted_pos, my_neck_angle, act_view_width, p->pos)
					&& (me.time_pos.time - p->time_pos.time) > 4 //padua2003
					//&& !in_feel_area(me.pos,p->pos,3.0) //if a player is to near to me, then never invalidate him (because the view area is to small to be reliably used)
			)
			{
				p->time_pos.reset(); //invalidate players pos
				MYLOG_WM(time.time,0, << "DONT SEE THE PLAYER " << i << "(team " << t << "), BUT I SHOULD");
				//show_view_area(my_shifted_pos, my_neck_angle, act_view_width);
				MYLOG_WM(time.time,1, << _2D << VC2D(p->pos,5, "ff0000"));
			}
		}
	}
}

void WMstate::stop_ball_in_immediate_vicinity_of_other_players() { //heuristics, shuould be done by the agent himself (in the future)
	if (ball.time_pos != time) //just treat situations seen in current cycle
		return;

	if ( ball.vel.sqr_norm() > SQUARE(1.0) )
		return;

	_wm_player & me = my_team[ WM::my_number ];

	if (me.pos.sqr_distance(ball.pos) <= SQUARE( WM::my_kick_radius) ) //I have the ball, so don't change anything
		return;

	for (int t=0; t < 2; t++)
		for (int i=0; i < NUM_PLAYERS + 1; i++) {
			_wm_player * p= ref_player(t,i);
			if ( !p->alive )
				continue;

			if ( p->time_pos == ball.time_pos
					&& ball.pos.sqr_distance(p->pos) <= SQUARE( WM::my_kick_margin * 0.8) ) {
				MYLOG_WM(time.time,0, << "I have stopped the ball by vel= " << ball.vel << " player " << i << " team " << t);
				ball.vel= Vector(0.0,0.0); //stop the ball

				return;
			}
		}
}

double WMstate::time_diff_ball_to_mdpstate_probability( int tdiff ) const
{
    if( tdiff > 7 )
        return 0.0;
    return 1.0;
}

double WMstate::time_diff_player_to_mdpstate_probability( int team, int tdiff ) const
{
    if( team == my_TEAM )
    {
        if( tdiff > WMoptions::max_cycles_to_forget_my_player )
            return 0.0;
        return 1.0;
    }
    if( tdiff > WMoptions::max_cycles_to_forget_his_player )
        return 0.0;
    return 1.0;
}

void WMstate::incorporate_msg_see(const Msg_see & see, long ms_time, long ms_time_delay) {
	if (time.time != see.time)
		INFO_OUT << ID << "\n Current wmstate.time= " << time << " , see.time= " << see.time;

	time_of_last_msg_see= time;
	ms_time_of_last_msg_see = ms_time;
	ms_time_of_last_msg_see_after_sb= ms_time_delay;

	_wm_player & me = my_team[ WM::my_number ];
	Vector my_pos_before_update= me.pos;

	update_me_from_msg_see(see);

	//compute now my_angle
	my_angle= my_neck_angle - my_neck_angle_rel;

	update_ball_from_msg_see(my_pos_before_update,see);

	update_players_from_msg_see(see);

	handle_inconsistent_objects_in_view_area();

	//stop_ball_in_immediate_vicinity_of_other_players();
}

void WMstate::incorporate_msg_hear(const Msg_hear & hear) {
	MYLOG_WM(time.time, 0 , << "play_mode= " << play_mode << "hear.play_mode= " << hear.play_mode << " " << hear.play_mode_upd);

	if (hear.play_mode_upd) {
		bool dont_change_play_mode=
				(/**/play_mode == /**/PM_goalie_catch_ball_l && hear.play_mode == PM_free_kick_l) ||              //don't change to free kick mode, after a galie catch, becase it is a mode a it's own!!!
				(play_mode == PM_goalie_catch_ball_r && hear.play_mode == PM_free_kick_r) || //don't change to free kick mode, after a galie catch, becase it is a mode a it's own!!!
				hear.play_mode == PM_half_time ||
				hear.play_mode == PM_time_extended;

		MYLOG_WM(time.time,0,<<"dont_change_play_mode="<<dont_change_play_mode);

		if ( ! dont_change_play_mode )
			play_mode= hear.play_mode;

		switch ( play_mode ) {
		case PM_before_kick_off:
			//case PM_goal_l:
			//case PM_goal_r:
		case PM_kick_off_l:
		case PM_kick_off_r:
			//here we are sure about the ball position
			ball.pos= Vector(0,0);
			ball.vel= Vector(0,0);
			ball.time_pos.time= hear.time;
			ball.time_vel.time= hear.time;
			break;
		case PM_penalty_onfield_l:
			penalty_side= left_SIDE;
			penalty_count= 0;
			break;
		case PM_penalty_onfield_r:
			penalty_side= right_SIDE;
			penalty_count= 0;
			break;
		case PM_penalty_setup_l:
			if ( WM::my_side == left_SIDE )
				penalty_count++;
			break;
		case PM_penalty_setup_r:
			if ( WM::my_side == right_SIDE )
				penalty_count++;
			break;
		default:
			break;//nothing to do, just to avoid compiler warnings
		}
	}

	if (hear.my_score_upd) {
		my_score= hear.my_score;
	}

	if (hear.his_score_upd) {
		his_score= hear.his_score;
	}

	// JTS 10: insert card handling
	if ( hear.card_update ) {
		incorporate_msg_card(hear.time, hear.card);
	}

	MYLOG_WM(time.time,0, << "teamcomm_upd= " << hear.teamcomm_upd << " WMoptions::recv_teamcomm= " << WMoptions::recv_teamcomm << " message= " <<  hear);
	if ( hear.teamcomm_partial_upd ) {
		teamcomm_statistics.teamcomm_partial_count++;
	}

	if ( hear.teamcomm_upd && WMoptions::recv_teamcomm ) {
		incorporate_msg_teamcomm(hear.time, hear.teamcomm);
	}

	if (hear.my_online_coachcomm_upd )
	{
		MYLOG_WM(time.time,0, << "Got our special online coach message");
		//INFO_OUT << ID << "TGCOACH" << hear.my_online_coachcomm;
		if (hear.my_online_coachcomm.his_player_types_upd)
			for (int i=0; i<NUM_PLAYERS; i++)
			{
				int pt= hear.my_online_coachcomm.his_player_types[i];
				if (pt>=0)
					his_team[i+1].type = pt;
			}
		if (hear.my_online_coachcomm.his_goalie_number_upd)
		{
			if ( his_goalie_number != hear.my_online_coachcomm.his_goalie_number ){
				if ( his_goalie_number > 0)
					ERROR_OUT << ID << "\ngot new inconsistent his_goalie_number=" << hear.my_online_coachcomm.his_goalie_number << "(old= " << his_goalie_number << ") from coach" << std::flush;
				else
					INFO_OUT << ID << "\ngot new his_goalie_number=" << hear.my_online_coachcomm.his_goalie_number << "(old= " << his_goalie_number << ") from coach" << std::flush;
			}
			his_goalie_number= hear.my_online_coachcomm.his_goalie_number;
		}
		if (hear.my_online_coachcomm.direct_opponent_assignment_upd)
		{
			//printf oder cout: INFO_OUT << ID << "\ngot new direct_opponent_assignment from coach" << std::flush;
			MYLOG_WM(time.time,0,  << "got new direct_opponent_assignment from coach" << std::flush);
			for (int i=0; i<NUM_PLAYERS; i++)
			{
				int don = hear.my_online_coachcomm.direct_opponent_assignment[i];
				int docn = hear.my_online_coachcomm.direct_opponent_conflict[i];
				my_team[i+1].direct_opponent_number = don;
				my_team[i+1].direct_opponent_conflict_number = docn;
			}
			number_of_direct_opponent_updates_from_coach ++ ;
		}
		if (hear.my_online_coachcomm.stamin_capacity_info_upd)
		{
		  MYLOG_WM(time.time,0,  << "got new stamina_capacity_information from coach" << std::flush);
          for (int i=0; i<NUM_PLAYERS; i++)
          {
            int stamina_cap = hear.my_online_coachcomm.stamin_capacity_info[i];
            my_team[i+1].stamina_capacity_bound = stamina_cap;
            //OUTDATED: Former non-CL-based communication provided stamina cap as from 'A'-'Z'
            //my_team[i+1].stamina_capacity_bound = Pointto10::get_stamina_capacity_from_base_encoding(stamina_cap);
          }
		}
	}
#if 0
	if (hear.rel_pos_of_communicating_opponent_upd) {
		//$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
		//not yet implemented, try to match the opponent and update his position!
	}
#endif
}

void WMstate::incorporate_msg_card(int hear_time, const Msg_card & mc) {
	_wm_player * wm_p = 0;
	if ( hear_time != mc.time )
		return; // invalid time for card info << SHOULD NEVER HAPPEN JTS 10
	if ( mc.side == WM::my_side ) {
		wm_p= ref_player(my_TEAM, mc.card_player);
	}
	else {
		wm_p= ref_player(his_TEAM, mc.card_player);
	}
	if ( !wm_p )
		return; // i do not know that player --> skip
	if ( wm_p->alive ) { // only update player card info if he is alive
		if ( mc.type == YELLOW_CARD )
			wm_p->yellow_card = true;
		if ( mc.type == RED_CARD )
			wm_p->red_card = true;
	}
}

void WMstate::incorporate_msg_teamcomm(int hear_time, const Msg_teamcomm2 & tc) {
	/* VERY VERY IMPORTANT
     ALL received data is in world model coordinates (this wasn't the case before 20030628). 
     pass_info must be converted during export, as are the other coordinates data
	 */
	if ( tc.from < 1 ) {
		ERROR_OUT << ID << "\nwrong message sender= " << tc.from;
		return;
	}
#if 1
	MYLOG_COMM1(hear_time,0,"+++++++++++++++++++++++++++++++++ WMstate::incorporate_msg_teamcomm"
			<<" at time.time: "<<time.time);
#endif
	teamcomm_statistics.teamcomm_count++;
	teamcomm_statistics.recv_time_of_last_teamcomm= time;
	teamcomm_statistics.sender_of_last_teamcomm= tc.from;

	MYLOG_COMM1(hear_time,0, "INCORPORATE TEAMCOMM from:  " << tc.from << "\n " << tc);
	if ( tc.msg.valid ) {
		msg= tc.msg;
		msg.time= hear_time;
		MYLOG_COMM(hear_time,0, "TEAMCOMM about free msg " << tc.msg);
	}

	if ( tc.ball.valid ) { //this info comes from the WMstate
		MYLOG_COMM1(hear_time,0, _2D << VC2D(tc.ball.pos,1.7,"ffffff"));
		//	   << STRING2D(tc.ball_pos.x+1.7,tc.ball_pos.y, "ball","ffffff") );
		int time_diff= hear_time - ball.time_pos.time;
		//if ( time_diff > 8 || time_diff > 3 && my_team[WM::my_number].pos.sqr_distance(ball.pos) > SQUARE(3.0) ) { //old
		if ( time_diff > 3  ) {
			ball.pos= tc.ball.pos; //don't update the time of the ball (to enforce looking to the ball after a while)a
			MYLOG_COMM1(hear_time,0, "TEAMCOMM about ball " << tc.ball.pos);
		}
	}

	if ( tc.ball_info.valid ) { //this info comes from the WSinfo, and MUST be converted
		MYLOG_COMM(hear_time,0, << _2D << C2D(tc.ball_info.ball_pos.x,tc.ball_info.ball_pos.y,1.8,"ffffff"));
		//	   << STRING2D(tc.ball_pos.x+1.7,tc.ball_pos.y, "ball","ffffff") );
		int time_diff= hear_time - ball.time_pos.time;
		//if ( time_diff > 8 || time_diff > 3 && my_team[WM::my_number].pos.sqr_distance(ball.pos) > SQUARE(3.0) ) { //old
		int ageThreshold = 1;
		if (tc.from == 1) ageThreshold = 2;
		if (tc.from == 1 && tc.ball_info.ball_pos.sqr_distance(MY_GOAL_CENTER) > SQUARE(20.0)) ageThreshold = 4;
		if ( time_diff > tc.ball_info.age_pos + ageThreshold )
		{
			ball.pos= tc.ball_info.ball_pos + tc.ball_info.ball_vel;
			ball.vel= tc.ball_info.ball_vel;
			ball.time_pos.time= hear_time - (tc.ball_info.age_pos + 1); //one cycle is needed to send the message
			ball.time_vel.time= hear_time - (tc.ball_info.age_vel + 1); //one cycle is needed to send the message
			MYLOG_COMM1(hear_time,0, "TEAMCOMM ball_info pos=" << tc.ball_info.ball_pos << " (age:" << tc.ball_info.age_pos
					<< ") vel= " << tc.ball_info.ball_vel << " (age:" << tc.ball_info.age_vel << ") || hear_time="<<hear_time<<", own ball age="<<time_diff);
		}
		else{
			MYLOG_COMM1(hear_time,0, " DON'T USE ball_info pos_age= " << tc.ball_info.age_pos << " own ball age= " << time_diff);
		}
	}
	else
	{
		MYLOG_COMM1(hear_time,0, "Incorporate Ball Info: DO NOT use ball information, because ball info is not valid.");
	}


	for (int i=0; i<tc.players_num; i++) {   //this info comes from the WMstate
		//consider updating the players
		const Msg_teamcomm2::_player & tc_p= tc.players[i];

		MYLOG_COMM1(hear_time,0, _2D << VC2D(tc_p.pos,1.7,"ffffff"));
		//	   << STRING2D(tc_p.pos.x+1.7,tc_p.pos.y, TEAM_STR[tc_p.team] << tc_p.number,"ffffff") );

		if (tc_p.team == my_TEAM && WM::my_number == tc_p.number ) //don't update me self from comm. information
			continue;

		_wm_player * wm_p= ref_player(tc_p.team, tc_p.number);

		if ( !wm_p )
			continue;
		MYLOG_COMM1(hear_time,0, "TEAMCOMM GOT INFO about player "
				<< TEAM_STR[tc_p.team] << tc_p.number << " pos= " << tc_p.pos
				<<" hear time: "<<hear_time<<" time_pos.time: "<<wm_p->time_pos.time);

		//int myDONumber;
		//if (Noball05::haveILostMyDirectOpponent(myDONumber))
		//{
		//  if (tc_p.team == his_TEAM && myDONumber == tc_p.number)
		//  {
		//    MYLOG_COMM1(hear_time,0, "I GOT MY DIRECT OPPONENT "<<myDONumber<<" VIA COMMUNICATION from temmate "<<tc.from<<"!");
		//  }
		//}
		if (    hear_time - wm_p->time_pos.time > 2     //don't update a player which was recently seen!
				|| (hear_time - wm_p->time_pos.time > 0 && tc_p.team ==my_TEAM && tc.from == tc_p.number) ) //if the sender sends its position, it is very trustworthy
		{
			//if ( hear_time - wm_p->time_pos.time > 1 ) { // ridi04: if hear time is more recent than see time, update!
			if (    tc.from != 1 //my goalie is eventually unreliable
					|| tc_p.pos.getX() < -20 )//goalie cannot see reliable so distant
			{
				wm_p->alive= true;
				if(tc_p.team ==my_TEAM && tc.from == tc_p.number)
					// if the sender sends its position, its the predicted position and therefore is valid
					wm_p->time_pos.time= hear_time;
				else
					wm_p->time_pos.time= hear_time -1; //the message was sent in the last cycle (in most cases)
				wm_p->pos = tc_p.pos;
				MYLOG_COMM1(hear_time,0, "TEAMCOMM INCORPORATING INFO about player "
						<< TEAM_STR[tc_p.team] << tc_p.number << " pos= " << tc_p.pos);
			}
		}
		else{
			MYLOG_COMM1(hear_time,0, "DO NOT incorportate, because recently seen ");
		}
	}

	if ( tc.pass_info.valid ) { //this info comes from the WSinfo directly into WSinfo, and MUST NOT be converted
		teamcomm_statistics.pass_info_count++;
		_wm_player & wm_p= my_team[tc.from];
		////////////// TG /////////////// [right before Osaka05]
		if ( hear_time - wm_p.time_pos.time > 1 ) //do not update player pos if he was seen recently
		{
			wm_p.alive = true;
			wm_p.pos   = tc.pass_info.ball_pos;
			wm_p.time_pos.time = hear_time - 1;
			MYLOG_COMM1(hear_time,0, "Incorporate Pass Info: TEAMCOMM INCORPORATING INFO about TEAMMATE "<<tc.from
					<< " pos= " << tc.pass_info.ball_pos);
		}
		else
		{
			MYLOG_COMM1(hear_time,0, "Incorporate Pass Info: DO NOT update player position, because he was recently seen.");
		}
		////////////// TG ///////////////
		////////////// TG /////////////// [right at Eindhoven06]
		if ( hear_time - ball.time_pos.time > 1  )  //do not update ball pos if it was seen recently
		{
			ball.vel = tc.pass_info.ball_vel;
			if (    tc.pass_info.time == 0 ) //immediate pass
			{
				ball.pos = tc.pass_info.ball_pos + tc.pass_info.ball_vel;
				ball.time_pos.time = hear_time - 1;
				MYLOG_COMM1(hear_time,0, "Incorporate Pass Info: TEAMCOMM INCORPORATING INFO about BALL at pos "
						<< ball.pos<<" and vel="<<ball.vel<<" (immediate pass -> added ballpos + ballvel).");
			}
			else
			{
				ball.pos = tc.pass_info.ball_pos;
				ball.time_pos.time = hear_time - 1;
				MYLOG_COMM1(hear_time,0, "Incorporate Pass Info: TEAMCOMM INCORPORATING INFO about BALL at pos "
						<< ball.pos<<" and vel="<<ball.vel<<" (non-immediate pass).");
			}
		}
		else
		{
			MYLOG_COMM1(hear_time,0, "Incorporate Pass Info: DO NOT update ball position, because it was recently seen (t ago "
					<<(hear_time - ball.time_pos.time)<<").");
		}
		////////////// TG ///////////////
		wm_p.pass_info.valid= true;
		wm_p.pass_info.recv_time= hear_time;
		wm_p.pass_info.ball_pos= tc.pass_info.ball_pos;
		wm_p.pass_info.ball_vel= tc.pass_info.ball_vel;
		wm_p.pass_info.time= hear_time + tc.pass_info.time; //tc stores just the time difference
	}

	if ( tc.ball_holder_info.valid ) {  //this info comes from the WSinfo, and MUST be converted
		_wm_player & wm_p= my_team[tc.from];
		wm_p.alive= true;
		wm_p.pos= tc.ball_holder_info.pos;
		wm_p.time_pos.time= hear_time -1; //the message was sent in the last cycle (in most cases)
		ball.pos= wm_p.pos;
		ball.vel= Vector(0,0);
		ball.time_pos= wm_p.time_pos;
		MYLOG_COMM1(hear_time,0, "Incorporate Ball Holder Info: TEAMCOMM INCORPORATING INFO about ball holder at "<<wm_p.pos);
		teamcomm_statistics.recent_ballholder_info_cycle = hear_time;//TG08
		teamcomm_statistics.recent_ballholder_number = tc.from;//TG08
	}
	else
	{
		MYLOG_COMM1(hear_time,0, "Incorporate Ball Holder Info: DO NOT incorporate, because info is not valid.");
	}

	//TGdoa: begin
	if ( tc.direct_opponent_assignment_info.valid )
	{
		MYLOG_COMM1(hear_time,0, "Incorporate Direct Opponent Assignment Info: Got DOA info; assignment = "
				<<tc.direct_opponent_assignment_info.assignment<<" (current assignment = "<<current_direct_opponent_assignment<<")");
		if (current_direct_opponent_assignment == -1)
			current_direct_opponent_assignment = tc.direct_opponent_assignment_info.assignment; //first assignment
		else
		{
			int diff = tc.direct_opponent_assignment_info.assignment - current_direct_opponent_assignment;
			if (diff == 1 || diff == -2) //due to modulo calculation //TG08: Changed from -3 to -2, since now we have one id less.
			{
				current_direct_opponent_assignment = tc.direct_opponent_assignment_info.assignment;
				MYLOG_COMM1(hear_time,0, "Incorporate Direct Opponent Assignment Info: DOA assignment (assignment="
						<<current_direct_opponent_assignment<<") has been incorporated into WMstate.");
			}
		}
	}
	//TGdoa: end

	//TGpr: begin
	if ( tc.pass_request_info.valid )
	{
		MYLOG_COMM1(hear_time,0, "Incorporate Pass Request: Pass Request received from player "
				<<tc.from<<", pass_in_n_steps="<<tc.pass_request_info.pass_in_n_steps
				<<", pass_param="<<tc.pass_request_info.pass_param);
		if(hear_time>1) { LOG_POL(0,"Incorporate Pass Request: Pass Request received from player "
				<<tc.from<<", pass_in_n_steps="<<tc.pass_request_info.pass_in_n_steps
				<<", pass_param="<<tc.pass_request_info.pass_param); }
		_wm_player & wm_p= my_team[tc.from];
		wm_p.alive= true;
		wm_p.pass_request.valid           = true;
		wm_p.pass_request.pass_in_n_steps = tc.pass_request_info.pass_in_n_steps;
		wm_p.pass_request.pass_param      = tc.pass_request_info.pass_param;
		wm_p.pass_request.received_at     = hear_time;
	}
	else
	{
		MYLOG_COMM1(hear_time,0, "Handle Pass Requests: NO MORE pass request received from player "
				<<tc.from
				<<", thus I invalidate his pass request.");
		if(hear_time>1) { LOG_POL(0,"Handle Pass Requests: NO MORE pass request received from player "
				<<tc.from
				<<", thus I invalidate his pass request."); }
		_wm_player & wm_p= my_team[tc.from];
		wm_p.pass_request.valid = false;
	}
	//TGpr: end

	MYLOG_COMM(hear_time,0,"---------------------------------WMstate::incorporate_msg_teamcomm");
}


void WMstate::incorporate_msg_teamcomm(const Msg_teamcomm & tc) {
#if 0
	MYLOG_COMM(time.time,0,"+++++++++++++++++++++++++++++++++ WMstate::incorporate_msg_teamcomm");
#endif
	if ( tc.side != WM::my_side ) {
		ERROR_OUT << ID << "\ntest: received message from other team";
		return;
	}

	teamcomm_statistics.recv_time_of_last_teamcomm= time;
	teamcomm_statistics.sender_of_last_teamcomm= tc.from;

	if ( tc.his_goalie_number_upd ) { //consider updating the his_goalie_number
		MYLOG_COMM(time.time,0,"&&&& RECEIVED GOALIE NUMBER old= " << his_goalie_number << " new= " << tc.his_goalie_number );
		if ( his_goalie_number == 0 )
			his_goalie_number= tc.his_goalie_number;
		else
			if ( his_goalie_number != tc.his_goalie_number )
				ERROR_OUT << ID << "TEAMCOMM: (" << WM::my_number
				<< ") his_goalie_number = " << his_goalie_number
				<< " != " << tc.his_goalie_number << "his_goalie_number (" << tc.from << ")";
	}

	if (tc.ball_upd) { //consider updating the ball
		int time_diff= tc.time - tc.ball.how_old - ball.time_pos.time;
		if ( time_diff > 0 ) {
			Vector from_pos= my_team[tc.from].pos;
			Vector my_pos= my_team[WM::my_number].pos;
			Vector tmp( tc.ball.x, tc.ball.y );
			if (! ( my_pos.sqr_distance(tmp) < SQUARE(from_pos.distance(tmp)) && time_diff <= 2) ) {
				/* the above check is done to avoid situations in
	   which the sender of the ball information was 
	   further from the ball then the player himself 
	   and his info is not considerable newer */
				MYLOG_COMM(time.time,0, "!! (" << tc.from << ") UPDATING ball   old_pos= "
						<< ball.pos << " new_pos= " << tc.ball.x << "," << tc.ball.y);

				ball.time_pos.time= tc.time - tc.ball.how_old;
				ball.time_vel.time= tc.time - tc.ball.how_old;
				ball.time_pos.cycle= 0;
				ball.time_vel.cycle= 0;
				ball.time_pos.total_cycle -= tc.ball.how_old; //just an approximation
				ball.time_vel.total_cycle -= tc.ball.how_old; //just an approximation

				ball.pos.setX(tc.ball.x);
				ball.pos.setY(tc.ball.y);
				ball.vel.setX(tc.ball.vel_x);
				ball.vel.setY(tc.ball.vel_y);
#if 0
				int time_diff= time.time - tc.time;
				//must be tested before using
				for (int i=0; i< time_diff; i++) { //simulate the missing time steps
					ball.pos+= ball.vel;
					ball.vel*= ServerOptions::ball_decay;
				}
#endif
			}
		}
	}

	for (int i=0; i<tc.players_num; i++) { //consider updating the players
		const Msg_teamcomm::_tc_player & tc_p= tc.players[i];
		if (tc_p.number > 0 && tc_p.number <= NUM_PLAYERS ) {
			if (tc_p.team == my_TEAM && WM::my_number == tc_p.number ) //don't update me self from comm. information
				continue;

			_wm_player * wm_p= ref_player(tc_p.team, tc_p.number);

			if ( !wm_p )
				continue;

			if ( tc.time- tc_p.how_old > wm_p->time_pos.time) {
#if 0
				MYLOG_COMM(time.time,0, "(" << tc.from << ") UPDATING player= " << tc_p.number << " team= " << tc_p.team << "  old_pos= " << wm_p->pos << " new_pos= " << tc_p.x << "," << tc_p.y << " how_old= " << tc_p.how_old);
#endif
				wm_p->alive= true;
				wm_p->time_pos.time= tc.time - tc_p.how_old;
				wm_p->time_pos.cycle= 0;
				wm_p->time_pos.total_cycle -= tc_p.how_old;
				wm_p->pos.setX(tc_p.x);
				wm_p->pos.setY(tc_p.y);
			}
		}
	}

	MYLOG_COMM(time.time,0,"---------------------------------WMstate::incorporate_msg_teamcomm");
}

void WMstate::incorporate_msg_change_player_type(const Msg_change_player_type & cpt) {
	if ( cpt.type <= 0 ) {
		//opponent player with number cpt.number has changed his type,
		//nothing yet implemented to handle it (the coach will handle it)
		return;
	}
	INFO_OUT << ID << "\nmy_teem[" << cpt.number << "].type changed: "
			<< my_team[cpt.number].type << " -> " << cpt.type;
	my_team[cpt.number].type = cpt.type;
	if ( cpt.number == WM::my_number )
		WM::set_my_type( ServerParam::get_player_type( cpt.type ) );
}

void WMstate::reduce_dash_power_if_possible(Cmd_Body & cmd) const {
	//////////////LOG_DEB(0," inside reduce dash power");
	if ( cmd.get_type() != cmd.TYPE_DASH )
		return;

	double dash_power, dash_angle;
	cmd.get_dash(dash_power, dash_angle);
	if ( dash_power > 100.1 ) {
		ERROR_OUT << ID << " dash_power too large " << dash_power;
		dash_power= 100.0;
	}
	else if ( dash_power < -100.1 ) {
		ERROR_OUT << ID << " abs. value of dash_power too large " << dash_power;
		dash_power= -100.0;
	}
	// no dash power reduction for side dashed //TG16
    if ( fabs(dash_angle) > 0.001 )
        return;

	const _wm_player & me= my_team[WM::my_number];
	Msg_player_type const* player_type= ServerParam::get_player_type( me.type );
	if ( !player_type) {
		ERROR_OUT << ID << " wrong player type pointer";
		return;
	}
	double player_speed= me.vel.norm();
	double dash_to_keep_max_speed= ( player_type->player_speed_max - player_speed ) / (player_type->dash_power_rate * my_effort);
	LOG_DEB(0, " dash_to_keep_max_speed= " << dash_to_keep_max_speed << " dash_power= " << dash_power
			<< " speed= " << player_speed << " speed_max= " << player_type->player_speed_max << " dprate= " << player_type->dash_power_rate << " dprate*effort= " << player_type->dash_power_rate * my_effort
			<< " stamina_per_meter= " << player_type->stamina_demand_per_meter);
	if ( dash_to_keep_max_speed < fabs(dash_power) - 0.5 ) {
		LOG_DEB(0, " reducing dash power from " << dash_power << " to +/- " << dash_to_keep_max_speed + 0.5);
		cmd.unset_lock();
		if ( dash_power > 0 ) {
			dash_power= dash_to_keep_max_speed + 0.5;
			if ( dash_power > 100.0 )
				dash_power= 100.0;
		}
		else {
			dash_power= -dash_to_keep_max_speed - 0.5;
			if ( dash_power < -100.0 )
				dash_power= -100.0;
		}
		cmd.set_dash( dash_power );
	}
}

int WMstate::server_playmode_to_player_playmode(PlayMode server_playmode) const {
	LOG_WM(time.time, 0 , << " exporting play mode= " << server_playmode);
	//std::cout << "\n exporting play mode= " << show_play_mode(server_playmode) << " penalty_count= " << penalty_count;

	bool keep;
	if (WM::my_side == left_SIDE)
		keep= true;
	else
		keep= false;

	switch ( server_playmode) {
	case PM_before_kick_off:      return PM_my_BeforeKickOff;
	case PM_time_over:            return PM_TimeOver;
	case PM_play_on:              return PM_PlayOn;
	case PM_kick_off_l:           return keep ? PM_my_KickOff : PM_his_KickOff;
	case PM_kick_off_r:           return keep ? PM_his_KickOff : PM_my_KickOff;
	case PM_kick_in_l:            return keep ? PM_my_KickIn : PM_his_KickIn;
	case PM_kick_in_r:            return keep ? PM_his_KickIn : PM_my_KickIn;
	case PM_indirect_free_kick_l:
	case PM_free_kick_l:          return keep ? PM_my_FreeKick : PM_his_FreeKick;
	case PM_indirect_free_kick_r:
	case PM_free_kick_r:          return keep ? PM_his_FreeKick : PM_my_FreeKick;
	case PM_corner_kick_l:        return keep ? PM_my_CornerKick : PM_his_CornerKick;
	case PM_corner_kick_r:        return keep ? PM_his_CornerKick : PM_my_CornerKick;
	case PM_goal_kick_l:          return keep ? PM_my_GoalKick : PM_his_GoalKick;
	case PM_goal_kick_r:          return keep ? PM_his_GoalKick : PM_my_GoalKick;
	case PM_goal_l:               return keep ? PM_my_AfterGoal : PM_his_AfterGoal;
	case PM_goal_r:               return keep ? PM_his_AfterGoal : PM_my_AfterGoal;
	case PM_drop_ball:            return PM_Drop_Ball;
	case PM_offside_l:            return keep ? PM_his_OffSideKick : PM_my_OffSideKick;
	case PM_offside_r:            return keep ? PM_my_OffSideKick : PM_his_OffSideKick;
	case PM_goalie_catch_ball_l:  return keep ? PM_my_GoalieFreeKick : PM_his_GoalieFreeKick;
	case PM_goalie_catch_ball_r:  return keep ? PM_his_GoalieFreeKick : PM_my_GoalieFreeKick;

	case PM_free_kick_fault_l:    return keep ? PM_his_FreeKick :  PM_my_FreeKick;
	case PM_free_kick_fault_r:    return keep ? PM_my_FreeKick :   PM_his_FreeKick;
	case PM_catch_fault_l:        return keep ? PM_his_FreeKick :  PM_my_FreeKick;
	case PM_catch_fault_r:        return keep ? PM_my_FreeKick :   PM_his_FreeKick;
	case PM_back_pass_l:          return keep ? PM_his_FreeKick :  PM_my_FreeKick;
	case PM_back_pass_r:          return keep ? PM_my_FreeKick :   PM_his_FreeKick;

	case PM_penalty_miss_l:       return keep ? PM_my_PenaltyKick : PM_his_PenaltyKick;
	case PM_penalty_miss_r:       return keep ? PM_his_PenaltyKick : PM_my_PenaltyKick;
	case PM_penalty_foul_l:       return keep ? PM_my_PenaltyKick : PM_his_PenaltyKick;
	case PM_penalty_foul_r:       return keep ? PM_his_PenaltyKick : PM_my_PenaltyKick;
	case PM_penalty_score_l:      return keep ? PM_my_PenaltyKick : PM_his_PenaltyKick;
	case PM_penalty_score_r:      return keep ? PM_his_PenaltyKick : PM_my_PenaltyKick;
	case PM_penalty_setup_l:      return keep ? PM_my_BeforePenaltyKick : PM_his_BeforePenaltyKick;
	case PM_penalty_setup_r:      return keep ? PM_his_BeforePenaltyKick : PM_my_BeforePenaltyKick;
	case PM_penalty_ready_l:      //return keep ? PM_my_BeforePenaltyKick : PM_his_BeforePenaltyKick;
	case PM_penalty_taken_l:
	case PM_penalty_kick_l:       return keep ? PM_my_PenaltyKick : PM_his_PenaltyKick;
	case PM_penalty_ready_r:      //return keep ? PM_his_BeforePenaltyKick : PM_my_BeforePenaltyKick;
	case PM_penalty_taken_r:
	case PM_penalty_kick_r:       return keep ? PM_his_PenaltyKick : PM_my_PenaltyKick;

	case PM_penalty_winner_l:
	case PM_penalty_winner_r:
	case PM_penalty_draw:         return PM_TimeOver;

	default: break;
	}

	LOG_WM(time.time, 0,<< " uknown play mode -> " << server_playmode);
	ERROR_OUT << ID << " uknown play mode -> " << 0;
	//return PM_PlayOn; //debug
	return PM_Unknown; //neve reaches this point
}

int WMstate::get_obj_id(int team, int number) const {
    if ( team== my_TEAM )
        return number;
    return number+NUM_PLAYERS;
}
int WMstate::get_obj_team(int obj_id) const {
    if (obj_id <= NUM_PLAYERS)
        return my_TEAM;
    return his_TEAM;
}
int WMstate::get_obj_number(int obj_id) const {
    if (obj_id <= NUM_PLAYERS)
        return obj_id;
    return obj_id -11;
}

Vector WMstate::compute_object_pos(const Vector & observer_pos, ANGLE observer_neck_angle_abs,
		double dist, double dir) const {
	ANGLE tmp=  ANGLE( DEG2RAD( - dir ) );
	tmp += observer_neck_angle_abs;
	Vector tmpVec;
	tmpVec.init_polar( dist, tmp.get_value() );
	return observer_pos + tmpVec;
}

Vector WMstate::compute_object_vel(const Vector & observer_pos, 
		const Vector & observer_vel,
		const Vector & object_pos,
		double dist, double dist_change, double dir_change) const {

	Vector e_rel= object_pos - observer_pos;

	e_rel.setY(-e_rel.getY()); //convert to server coordinates (VERY IMPORTANT)

	/* here use server formulas to compute object velocity

     dist_change=  v_rel.x * e_rel.x + v_rel.y * e_rel.y
     dir_change =  v_rel.x * (- e_rel.y) + v_rel.y * e_rel.x * 180.0/ ( dist * Pi)

     this can be written as a matrix

      /                      \     /         \     /                                \ 
     |	e_rel.x     e_rel.y   |   |  v_rel.x  |	  |  dist_change		     |
     |			      | * | 	      | = | 				     |
     |	- e_rel.y   e_rel.x   |	  |  v_rel.y  |	  |  dir_change * dist * Pi / 180.0  |
      \			     / 	   \	     / 	   \				    / 

      if e_rel.x > 0 or e_rel.y > 0 the left martix is invertible, and we get


      /         \     /                    \      /                                \ 
     |  v_rel.x	 |   |  e_rel.x   -e_rel.y  |	 |  dist_change                     |
     | 		 | = | 			    | *  |                                  |
     |  v_rel.y	 |   |  e_rel.y    e_rel.x  |	 |  dir_change * dist * Pi / 180.0  |
      \		/     \			   / 	  \                                / 

	 */
	if (dist > 0) //can sometimes happen to be zoro at the begin of a game (if one player moves to (0,0)
		e_rel *= 1.0/dist;
	else
		e_rel= Vector(0);

	double dum= dir_change * dist * M_PI / 180.0;
	Vector v_rel;

	v_rel.setX(e_rel.getX() * dist_change - e_rel.getY() * dum);
	v_rel.setY(e_rel.getY() * dist_change + e_rel.getX() * dum);

	v_rel.setY(-v_rel.getY()); //convert back to our coordinates (VERY IMPORTANT)

	/* here we have v_rel in our coordinates, it remains to add observer.vel to it */

	return observer_vel + v_rel;
}

Vector WMstate::compute_object_pos_after_collision(const Vector & object_pos, const Vector & object_vel, double object_radius, const Vector obstacle_pos, double obstacle_radius) const {
#if 0 //cut and pasted from the server, was never understood and does produce nan's
	//convert to server coordinates, probably not necessary, but you never know ;-)
	const Vector Server_object_pos= Vector( object_pos.x, -object_pos.y);
	const Vector Server_object_vel= Vector( object_vel.x, -object_vel.y);
	const Vector Server_obstacle_pos= Vector( obstacle_pos.x, -obstacle_pos.y);

	//see void MPObject::collide(MPObject& obj) in the server
	double r = object_radius + obstacle_radius ;
	Vector dif = (Server_object_pos - Server_obstacle_pos) ;
	double d = Server_object_pos.distance(Server_obstacle_pos);

	double dif_angle_vel= dif.angle(Server_object_vel); //art
	if (dif_angle_vel > PI)                      //art
		dif_angle_vel -= 2*PI;                     //art

	//Angle th = fabs(dif.angle(vel)) ;
	double th= fabs(dif_angle_vel);              //art

	double l1 = d * cos(th) ;
	double h = d * sin(th) ;
	double cosp = h / r ;
	//double sinp = sqrt(1.0 - square(cosp)) ;
	MYLOG_WM(time.time,0, << " cosp*cosp= " << cosp*cosp << " d= "<< d << " h= " << h << " r= " << r << " dif_angle_vel= " << dif_angle_vel);
	double sinp = sqrt(1.0 - cosp*cosp) ;        //art
	double l2 = r * sinp ;

	Vector dv = Server_object_vel ;

	MYLOG_WM(time.time,0, "         dv= " << dv << " l1= " << l1 << " l2= " << l2);
	dv.normalize(-(l1 + l2)) ;

	//convert to our coordinates
	dv.y= -dv.y;
	return object_pos + dv ;
#else  
	/* new version (9.06.2002), doesn't produce nan's and is very plausible
     further more it's cheaper in computation, because just one sqrt computation
     is needed!

     we have to solve the following identity

     distance( object_pos + K * object_vel,
               obstacle_pos + K * obstacle_vel )= 
     object_rad + obstacle_rad

     The result are 2 K's, take the smallest one which is >= 0, this is the right solution
	 */
	/* assume obstacle_vel to be zero, this seems also the solution taken
     in the server. Our solution works for arbitrary velocities, maybe
     the real obstacle_vel should be used in the future !!!
	 */
	Vector obstacle_vel= Vector(0.0,0.0); //@@@@@@@@@@@@@@@@@@@@@@@@

#if 0
	MYLOG_WM(time.time,0, << _2D
			<< C2D(obstacle_pos.x,obstacle_pos.y, obstacle_radius,"0000ff")
			<< C2D(object_pos.x, object_pos.y, object_radius,"0000ff")
			<< L2D(object_pos.x, object_pos.y, object_pos.x+object_vel.x, object_pos.y+object_vel.y,"00ff00")
	);
#endif

	Vector vec= object_vel - obstacle_vel;
	Vector tmp= object_pos - obstacle_pos;
	double d= object_radius + obstacle_radius;
	d *= d;
	/* now solve
     | K*vec + tmp |^2 = d
	 */

	double dum= vec.sqr_norm();
	if ( dum < 0.00000001 ) {
		/* there will be no (numarically stable) solution
       also here almost object_vel = obstacle_vel, so we cannot expect any solution
		 */
		MYLOG_WM(time.time,0, << "COLLISION no solution for collision (1)");
		//WARNING_OUT << ID << "\n" << time.time << " no solution for collision (1)";
		return object_pos;
	}

	d -= tmp.sqr_norm();
	d /= dum;

	dum= (vec.getX()*tmp.getX() + vec.getY() * tmp.getY())/dum;
	d += dum*dum;


	if ( d < 0.0 ) {
		MYLOG_WM(time.time,0, << "COLLISION no solution for collision (2)");
		//WARNING_OUT << ID << "\n" << time.time << " no solution for collision (2)";
		return object_pos;
	}

	d= sqrt(d);

	double K1= -dum - d;
	double K2= -dum + d;
	double K;
	if (fabs(K1) < fabs(K2))
		K=K1;
	else
		K=K2;

#if 0
	Vector tmp1= object_pos + K * object_vel - obstacle_pos;
	MYLOG_WM(time.time,0, << " diff= " << tmp1.norm() << " K= " << K << "(" << -dum - d << "," << -dum + d << ")";);
	//cout << "\n" << time.time << " diff= " << tmp1.norm() << " K= " << K << "(" << -dum - d << "," << -dum + d << ")";

	draw collision result
	MYLOG_WM(time.time,0, << _2D
			<< C2D(object_pos.x+ K*object_vel.x, object_pos.y+ K*object_vel.y, object_radius,"000000")
	);


#endif
	return object_pos + K * object_vel;

#endif
}

WMstate::_wm_player * WMstate::ref_player(int team, int number) {
    if (number < 0 || number > NUM_PLAYERS )
        return 0;
    if (team == my_TEAM)
        return my_team+number;
    if (team == his_TEAM)
        return his_team+number;
    return 0;
}

const WMstate::_wm_player * WMstate::const_ref_player(int team, int number) const {
    if (number < 0 || number > NUM_PLAYERS )
        return 0;
    if (team == my_TEAM)
        return my_team+number;
    if (team == his_TEAM)
        return his_team+number;
    return 0;
}

const WMstate::_wm_player * WMstate::const_ref_player(int obj_id) const {
    if (obj_id <= NUM_PLAYERS)
        return const_ref_player(my_TEAM,obj_id);
    return const_ref_player(his_TEAM,obj_id-NUM_PLAYERS);
}

/**
   the resulting values:
   res_team, res_number, sqr_res_distance
   are only reasonable if the return value is true
 */
bool WMstate::get_nearest_player_to_pos(int team, const Vector & pos, const PlayerFlagSet & players, int max_age, int & res_team, int & res_number, double & sqr_res_distance) const {
	bool min_exists= false;

	for (int t= 0; t<2; t++)
		if ( team == t || team == unknown_TEAM )
			for (int i= 1; i < NUM_PLAYERS+1; i++) {
				if ( players.get(t,i) == false )
					continue;

				const _wm_player * wm_p= const_ref_player(t,i);

				if ( !wm_p->alive )
					continue;

				if ( t == my_TEAM && i == WM::my_number ) //I'm never a candidate
					continue;

				if ( time.time - wm_p->time_pos.time > max_age ) //skip too old players
					continue;

				double sqr_distance= pos.sqr_distance( wm_p->pos );
				if ( !min_exists || sqr_distance < sqr_res_distance ) {
					min_exists= true;
					sqr_res_distance= sqr_distance;
					res_team= t;
					res_number= i;
				}
			}

	return min_exists;
}

bool WMstate::in_feel_area(const Vector & pos, const Vector & object_pos, double tolerance) const {
	return pos.distance(object_pos) <= ServerOptions::visible_distance + tolerance;
}


/** this method can be used to determine if an object is in the
  view area of a player. This area is defined by two rays. The angle
  between these rays is specified by the parameter view_width.

          / 
         / 
        /
       /
      /                               c
     /
    p
     \
      \      a
       \ 
        \
     b   \
          \

   in the above exaple 'a' and 'c' are in the view area of 'p', but 'b' is not.
   By now the distance between objects doesn't matter, it's enough to be between 
   the 2 specified rays.

   \param neck_angle absolute angle of players neck
   \param view_width is one of WIDE= 180 degree, NORMAL= 90 degree, NARROW= 45 degree
 */
bool WMstate::in_view_area(const Vector & pos, const ANGLE & neck_angle, int v_width, const Vector & object_pos) const {
	Vector tmp,tmp2;
	tmp.init_polar(1.0,neck_angle.get_value());
	tmp2= object_pos- pos;
	tmp2.normalize(1.0);

	double lowest_cosinus= 2.0;
	switch (v_width) {
	case WIDE:
		lowest_cosinus= 0; // = cos(Pi/2)
		break;
	case NORMAL:
		lowest_cosinus= 1.0/sqrt(2.0); // = cos(Pi/4)
		break;
	case NARROW:
		lowest_cosinus= cos(M_PI/8.0);
		break;
	default:
		ERROR_OUT << ID << "\nWMstate::in_view_area(...): wrong view_width mode: " << v_width << " not in {" << NARROW << "," << NORMAL << "," << WIDE << "}";
	}

	double cosinus= tmp.getX()*tmp2.getX() + tmp.getY()*tmp2.getY();
	return cosinus >= lowest_cosinus;
}

void WMstate::show_view_area(const Vector & pos, const ANGLE & neck_angle, int v_width) const {
	Vector tmp,tmp2;
	tmp.init_polar(40,neck_angle.get_value());
	tmp2= tmp;

	switch (v_width) {
	case WIDE:
		tmp.rotate(M_PI/2.0);
		tmp2.rotate(-M_PI/2.0);
		break;
	case NORMAL:
		tmp.rotate(M_PI/4.0);
		tmp2.rotate(-M_PI/4.0);
		break;
	case NARROW:
		tmp.rotate(M_PI/8.0);
		tmp2.rotate(-M_PI/8.0);
		break;
	default:
		ERROR_OUT << ID << "\nWMstate::show_view_area(...): wrong view_width mode: " << v_width << " not in {" << NARROW << "," << NORMAL << "," << WIDE << "}";
	}

	tmp+= pos;
	tmp2+= pos;

	MYLOG_WM(time.time,1, << _2D << VC2D(pos,3,"000000") << VL2D(pos,tmp,"000000") << VL2D(pos,tmp2,"000000") );
}
