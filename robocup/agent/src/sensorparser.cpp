/*
 * Author:   Artur Merke 
 */
#include "sensorparser.h"
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

#include "basics/wmoptions.h"
#include "basics/WMTools.h"
#include "str2val.h"
#include "server_options.h"
#include "serverparam.h"
#include "macro_msg.h"
#include "bit_fifo.h"
#include "ws_info.h"
#include "options.h"

#include "log_macros.h" //debug

const double SensorParser::ENCODE_SCALE= 150.0; // TG17: moved outside off class def for c++11 compatibility

bool SensorParser::get_message_type_and_time(const char * str, int & type, int & time) {
	const char * dum;

	if (strncmp(str,"(sense_body ", 12) == 0) {
		str+= 12;
		type= MESSAGE_SENSE_BODY;
		if (! str2val(str,time,dum) )
			return false;
		return true;
	}

	if (strncmp(str,"(see ", 4) == 0) {
		str+= 4;
		type= MESSAGE_SEE;
		if (! str2val(str,time,dum) )
			return false;
		return true;
	}

	if (strncmp(str,"(fullstate ", 11) == 0) {
		str+= 11;
		type= MESSAGE_FULLSTATE;
		if (! str2val(str,time,dum) )
			return false;
		return true;
	}

	if (strncmp(str,"(think)", 7) == 0) {
		str+= 11;
		type= MESSAGE_THINK;
		time= -1; //no time is provided with the think message
		return true;
	}

	if (strncmp(str,"(hear ", 6) == 0) {
		str+= 6;
		type= MESSAGE_HEAR;
		if (! str2val(str,time,dum) )
			return false;
		return true;
	}

	if (strncmp(str,"(change_player_type ", 20) == 0) {
		type= MESSAGE_CHANGE_PLAYER_TYPE;
		time= -1;
		return true;
	}

	if (strncmp(str,"(init ", 6) == 0) {
		type= MESSAGE_INIT;
		time= -1;
		return true;
	}

	if (strncmp(str,"(server_param ", 14) == 0) {
		type= MESSAGE_SERVER_PARAM;
		time= -1;
		return true;
	}

	if (strncmp(str,"(player_param ", 14) == 0) {
		type= MESSAGE_PLAYER_PARAM;
		time= -1;
		return true;
	}

	if (strncmp(str,"(player_type ", 13) == 0) {
		type= MESSAGE_PLAYER_TYPE;
		time= -1;
		return true;
	}

	if (strncmp(str,"(ok ", 3) == 0) {
		type= MESSAGE_OK;
		time= -1;
		return true;
	}

	if (strncmp(str,"(error ", 7) == 0) {
		type= MESSAGE_ERROR;
		time= -1;
		return true;
	}

	type= MESSAGE_UNKNOWN;
	return false;
}

//////////////////////////////////////////////////////////////////////
bool SensorParser::manual_parse_fullstate(const char* str, Msg_fullstate & fullstate) {
	fullstate.reset();

	if ( strncmp("(fullstate",str,10)!= 0 )
		return false;

	char* next;
	str += 10;
	fullstate.time= strtol(str,&next,10);
	str= next;

	fullstate.players_num= 0;
	while(true) {
		while(str[0]==' ')
			str++;
		if ( str[0] == ')' )
			return true;
		if ( str[0] != '(')
			return false;
		str ++;

		if ( (str[0]== 'l' || str[0]== 'r') && str[1]== '_' ) {
			Msg_fullstate::_fs_player & player= fullstate.players[fullstate.players_num];
			fullstate.players_num++;
			if ((str[0]== 'l' && WM::my_side== left_SIDE) || (str[0]== 'r' && WM::my_side== right_SIDE)) {
				player.team= my_TEAM;
			}
			else
				player.team= his_TEAM;

			str += 2;
			player.number= strtol(str,&next,10);
			str = next;
			player.x= strtod(str,&next);
			str = next;
			player.y= strtod(str,&next);
			str = next;
			player.vel_x= strtod(str,&next);
			str = next;
			player.vel_y = strtod(str,&next);
			str = next;
			player.angle = strtod(str,&next);
			str = next;
			player.neck_angle = strtod(str,&next);
			str = next;
			player.stamina = strtod(str,&next);
			str = next;
			player.effort = strtod(str,&next);
			str = next;
			player.recovery = strtod(str,&next);
			str = next;
			if (str[0]== ')')
				str++;
			else
				return false;
		}
		else if ( strncmp("ball ",str,5) == 0 ) {
			str += 5;
			fullstate.ball.x = strtod(str,&next);
			str = next;
			fullstate.ball.y = strtod(str,&next);
			str = next;
			fullstate.ball.vel_x = strtod(str,&next);
			str = next;
			fullstate.ball.vel_y = strtod(str,&next);
			str = next;
			if (str[0]== ')')
				str++;
			else
				return false;
		}
		else if ( strncmp("score ",str,6) == 0 ) {
			str += 6;
			fullstate.my_score= strtol(str,&next,10);
			str = next;
			fullstate.his_score= strtol(str,&next,10);
			str = next;
			if (WM::my_side == right_SIDE) {
				int tmp = fullstate.my_score;
				fullstate.my_score=  fullstate.his_score;
				fullstate.his_score= tmp;
			}

			if (str[0]== ')')
				str++;
			else
				return false;
		}
		else if ( strncmp("vmode ",str,6) == 0 ) {
			str += 6;
			const char * dum;
			if ( !manual_parse_view_mode(str,fullstate.view_quality, fullstate.view_width, dum) ) {
				ERROR_OUT << ID << "\nparse error " << str;
				return false;
			}
			str= dum;

			while ( str[0] == ' ' )
				str++;

			if (str[0]== ')')
				str++;
			else
				return false;
		}
		else if ( strncmp("pmode ",str,6) == 0 ) {
			str += 6;      /* 1234567890123456789 */
			PlayMode pm;
			const char * dum;
			int dum_int;
			if ( !manual_parse_play_mode(str,pm,dum_int,dum) ) {
				ERROR_OUT << ID << "\nparse error " << str;
				return false;
			}
			str= dum;
			fullstate.play_mode= pm;

			while ( str[0] == ' ' )
				str++;

			if (str[0]== ')')
				str++;
			else
				return false;
		}
	}
	// never reaches this point
	return false;
}

//////////////////////////////////////////////////////////////////////
bool SensorParser::manual_parse_fullstate(const char* str, Msg_fullstate_v8 & fullstate) {
	fullstate.reset();

	bool res= true;
	if ( strncmp("(fullstate",str,10)!= 0 )
		return false;

	str += 10;
	const char * dum= str;
	const char * dum2;
	int dum_int;

	res = res &&
			str2val(dum,fullstate.time,dum) &&
			//recognizing play mode
			strskip(dum,"(pmode",dum) &&
			manual_parse_play_mode(dum,fullstate.play_mode,dum_int,dum) &&
			strskip(dum,')',dum) &&
			//recognizing visual mode
			strskip(dum,"(vmode",dum) &&
			manual_parse_view_mode(dum,fullstate.view_quality, fullstate.view_width, dum) &&
			strskip(dum,')',dum) &&
#if 0
			//skipping stamina
			strskip(dum,"(stamina",dum) &&
			str2val(dum,dum_double,dum) &&
			str2val(dum,dum_double,dum) &&
			str2val(dum,dum_double,dum) &&
			strskip(dum,')',dum) &&
#endif
			//recognizing count
			strskip(dum,"(count",dum) &&
			str2val(dum,fullstate.count_kick,dum) &&
			str2val(dum,fullstate.count_dash,dum) &&
			str2val(dum,fullstate.count_turn,dum) &&
			str2val(dum,fullstate.count_catch,dum) &&
			str2val(dum,fullstate.count_kick,dum) &&
			str2val(dum,fullstate.count_turn_neck,dum) &&
			str2val(dum,fullstate.count_change_view,dum) &&
			str2val(dum,fullstate.count_say,dum) &&
			strskip(dum,')',dum);

	//some experimental arm stuff, will be skipped here "(arm (movable 0) (expires 0) (target 0 0) (count 0))"
	if ( res && strskip(dum,"(arm",dum2) ) {
		dum= dum2;
		res= strfind(dum,')',dum) && strfind(dum+1,')',dum) && strfind(dum+1,')',dum) && strfind(dum+1,')',dum) && strfind(dum+1,')',dum);
		dum++;
	}

	res= res &&
			//recognizing the score
			strskip(dum,"(score",dum) &&
			str2val(dum,fullstate.my_score,dum) &&
			str2val(dum,fullstate.his_score,dum);

	if (WM::my_side == right_SIDE) {
		int tmp = fullstate.my_score;
		fullstate.my_score=  fullstate.his_score;
		fullstate.his_score= tmp;
	}

	res = res && strskip(dum,')',dum) &&
			//recognizing the ball
			strskip(dum,"((b)",dum) &&
			str2val(dum,fullstate.ball.x,dum) &&
			str2val(dum,fullstate.ball.y,dum) &&
			str2val(dum,fullstate.ball.vel_x,dum) &&
			str2val(dum,fullstate.ball.vel_y,dum) &&
			strskip(dum,')',dum);

	//recognizing the players
	bool recognized_myself= false;

	fullstate.players_num= 0;
	while(res) {
		if ( ! strskip(dum,"((p",dum) )
			break;

		//recognizing player's header
		Msg_fullstate_v8::_fs_player & player= fullstate.players[fullstate.players_num];
		fullstate.players_num++;

		const char * dum2= dum;
		if ( strskip(dum,'l',dum2) ) {
			if (WM::my_side== left_SIDE)
				player.team= my_TEAM;
			else
				player.team= his_TEAM;
		}
		else if ( strskip(dum,'r',dum2) ) {
			if (WM::my_side== right_SIDE)
				player.team= my_TEAM;
			else
				player.team= his_TEAM;
		}
		else
			res= false;

		dum= dum2;

		res = res && str2val(dum,player.number,dum);

		if ( strskip(dum,'g',dum2 ) ){
			player.goalie= true;
			dum = dum2;
			if ( ClientOptions::server_version < 14 ) // JTS10 heterogenous goalie
				player.type= 0;
			else
				res = res && str2val(dum,player.type,dum2);
		}
		else {
			player.goalie= false;
			res = res && str2val(dum,player.type,dum2);
		}
		dum= dum2;

		res = res && strskip(dum,')',dum) &&
				//recognizing player's values
				str2val(dum,player.x,dum) &&
				str2val(dum,player.y,dum) &&
				str2val(dum,player.vel_x,dum) &&
				str2val(dum,player.vel_y,dum) &&
				str2val(dum,player.angle,dum) &&
				str2val(dum,player.neck_angle,dum);
		//recognizing player's arm pointing (which can be omitted)
		if (   str2val(dum,player.point_dist,dum2)
				&& str2val(dum2,player.point_dir,dum2) )
			dum = dum2;
		else
		{
			player.point_dist = -1.0; // invalidation
		}
		//recognizing stamina values
		res = res &&
				strskip(dum,"(stamina",dum) &&
				str2val(dum,player.stamina,dum) &&
				str2val(dum,player.effort,dum) &&
				str2val(dum,player.recovery,dum) &&
				//TG09
				(
						ClientOptions::server_version < 13
						|| str2val(dum,player.stamina_capacity,dum)
				) &&
				strskip(dum,')',dum);

		player.card = NO_CARD;
		player.kick_flag = false;
		player.tackle_flag = false;
		while (
				strskip(dum,'f',dum2)
				|| strskip(dum,'y',dum2)
				|| strskip(dum,'r',dum2)
				|| strskip(dum,'t',dum2)
				|| strskip(dum,'k',dum2)
		)
		{
			if (  ClientOptions::server_version >= 14 )
			{
				//JTS10 player card handling
				if ( strskip(dum,'f',dum2 ) ){
					player.fouled = true;
					dum = dum2;
				}
				if ( strskip(dum,'y',dum2 ) ){
					player.card = YELLOW_CARD;
					dum = dum2;
				}
				else
					if ( strskip(dum,'r',dum2 ) ){
						player.card = RED_CARD;
						dum = dum2;
					}
			}
			if (  ClientOptions::server_version >= 13 )
			{
				if ( strskip(dum,'k',dum2 ) ){
					player.kick_flag = true;
					dum = dum2;
				}
				if ( strskip(dum,'t',dum2 ) ){
					player.tackle_flag = true;
					dum = dum2;
				}
			}
		}
		//end of player's tuple
		res &= strskip(dum,')',dum);

		if (player.team == my_TEAM && player.number == WM::my_number)
			recognized_myself= true;
	}

	res = res && strskip(dum,')',dum);

	if (!res) {
		ERROR_OUT << ID << "\nparse error:\n";
		ERROR_STREAM << "\noriginal message: " << str;
		ERROR_STREAM << "\n\n>>>";
		show_parser_error_point(ERROR_STREAM,str,dum);
		return false;
	}
	else if ( ! recognized_myself  ) {
		ERROR_OUT << ID << "\ndidn't find myself in fullstate= " << str;
		return false;
	}

	return true;
}

bool SensorParser::manual_parse_see(const char* str, Msg_see & see, char const* & next) {
	const char * dum;
	//cout << "\nparsing=[" << str << "]";
	see.reset();
	next= str;

	if ( !strskip(next,"(see ",next) )
		return false;
	if ( !str2val(next, see.time, next) )
		return false;

	while (true) {
		if ( strskip(next,')',next) ) //ending braces -> end of message, otherwise eat up white space
			return true;

		if ( ! strskip(next,'(',next) )
			return false;

		ParseObject pobj;
		dum= next;
		if ( !manual_parse_see_object(next,pobj,next) ) {
			ERROR_OUT << ID << "\nunknown object" << next;
			return false;
		}

		if (pobj.res == pobj.UNKNOWN) {
			// *** IMPORTANT ***
			// "(F)" will be treated as pobj.UNKNOWN, bu maybe we should consider such information useful in the future
			// "(G)" will be treated as pobj.UNKNOWN, bu maybe we should consider such information useful in the future

			if (dum[1] != 'F' && dum[1] != 'G' && dum[2] != ')') { //are there some other unknown objects?
				INFO_OUT << ID << "\nunknown object: ";
				for (int i=0; dum[i] != '\0' && i <20; i++)
					INFO_STREAM << dum[i];
			}
		}
		double ppp[7];
		bool tackle_flag= false;
		bool kick_flag= false;
		int actual = str2val(next,7,ppp,next);  //read up to 7 values
		if (actual <= 0)
			return false;

		if ( strskip(next,'t',next) )
			tackle_flag= true;
		else if ( strskip(next,'k',next) )
			kick_flag = true;

		if ( ! strskip(next,')',next) )
			return false;

		//incorporate parsed data

		if (pobj.res == pobj.MARKER) {
			if ( see.markers_num < Msg_see::markers_MAX) {
				Msg_see::_see_marker & m= see.markers[see.markers_num];
				m.see_position= true;
				m.how_many= actual;
				m.x= pobj.x;
				m.y= pobj.y;
				m.dist = ppp[0];
				m.dir = ppp[1];
				m.dist_change = ppp[2];
				m.dir_change = ppp[3];
				see.markers_num++;
			}
			else
				ERROR_OUT << ID << "\nParser: to much markers";
		}
		else if (pobj.res >= pobj.PLAYER_OBJECT ) {
			//cout << "\nplayer object: " << pobj.res;
			if ( see.players_num < Msg_see::players_MAX) {
				Msg_see::_see_player & p= see.players[see.players_num];
				p.goalie= false;
				p.tackle_flag= tackle_flag;
				p.kick_flag= kick_flag;
				p.pointto_flag= false;
				switch (pobj.res) {
				case ParseObject::PLAYER_MY_TEAM_GOALIE :
					p.goalie= true;
				case ParseObject::PLAYER_MY_TEAM :
					p.team= my_TEAM;
					break;
				case ParseObject::PLAYER_HIS_TEAM_GOALIE :
					p.goalie= true;
				case ParseObject::PLAYER_HIS_TEAM :
					p.team= his_TEAM;
					break;
				case ParseObject::PLAYER_UNKNOWN :
					p.team= unknown_TEAM;
				}
				if (pobj.number > 0 && pobj.number < 12)
					p.number= pobj.number;
				else
					p.number= 0;

				p.how_many= actual;
				switch ( actual ) {
				case 7: p.how_many= 6; p.pointto_flag= true; p.pointto_dir= ppp[6];  //no braek at case 7,6,4
				case 6: p.body_dir = ppp[4]; p.head_dir = ppp[5];
				case 4: p.dist_change = ppp[2]; p.dir_change = ppp[3];
				case 2: p.dist = ppp[0]; p.dir = ppp[1];
				see.players_num++;
				break;
				case 3: p.how_many= 2; p.dist = ppp[0]; p.dir = ppp[1]; p.pointto_flag= true; p.pointto_dir= ppp[2];
				see.players_num++;
				break;
				case 1:
					//ignore players with just one parameter (players_num is not increased here!)
					if (p.number != 0) {
						// probably the view quality is low -> problems with chang view strategy, if this message comes to often!!!
						//WARNING_OUT << ID << "\nnumber of player parameters= " << actual << "\n --> the view quality is probably low (check change_view!!!)";
					}
					break;
				default:
					ERROR_OUT << ID << "\nwrong number of player parameters: " << actual
					<< "\np.number= " << p.number << " p.team= " << p.team << "\nmsg=" << str;
				}
			}
			else
				ERROR_OUT << ID << "\nParser: to much players";
		}
		else if (pobj.res == pobj.BALL_OBJECT) {
			if (!see.ball_upd) {
				see.ball.how_many= actual;
				see.ball.dist = ppp[0];
				see.ball.dir = ppp[1];
				see.ball.dist_change = ppp[2];
				see.ball.dir_change = ppp[3];
				see.ball_upd= true;
			}
			else
				ERROR_OUT << ID << "\nParser: more then one ball";
		}
		else if (pobj.res == pobj.MARKER_LINE) {
			if (!see.line_upd) {
				see.line.see_position= true;
				see.line.x= pobj.x;
				see.line.y= pobj.y;
				see.line.how_many= actual;
				see.line.dist = ppp[0];
				see.line.dir = ppp[1];
				see.line.dist_change = ppp[2];
				see.line.dir_change = ppp[3];
				see.line_upd= true;
			}
			else {
				;//ERROR_OUT << ID << "\nParser: more then one line";
			}
		}
		else if ( pobj.res == pobj.UNKNOWN )
			;
		else
			ERROR_OUT << ID << "\n-- UNKNOWN RESULT TYPE";
	}
	//never gets here
	return false;
}

bool SensorParser::manual_parse_init(const char* str, Msg_init & init) {
	init.reset();

	const char * dum;
	const char * origin= str;
	bool res= true;
	res= res && strskip(str,"(init ",str);

	dum= str;
	if ( strskip(str,'l',dum) )
		init.side= left_SIDE;
	else if ( strskip(str,'r',dum) )
		init.side= right_SIDE;
	else
		return res= false;
	str= dum;

	int dum_int;
	res= res && str2val(str,init.number,str) &&
			manual_parse_play_mode(str,init.play_mode,dum_int,str) &&
			strskip(str,')',str);

	if (!res) {
		ERROR_OUT << ID << "\nparse error:\n";
		show_parser_error_point(ERROR_STREAM,origin,str);
	}
	return res;
}

bool SensorParser::manual_parse_sense_body(const char * str, Msg_sense_body & sb, char const* & next) {
	sb.reset();
	const char * dum;
	next= str;
	//(sense_body 0 (view_mode high normal) (stamina 4000 1) (speed 0 0) (head_angle 0) (kick 0) (dash 0) (turn 0) (say 0) (turn_neck 0) (catch 0) (move 0) (change_view 0) (arm (movable 0) (expires 0) (target 0 0) (count 0)) (focus (target none) (count 0)) (tackle (expires 0) (count 0)))
	bool res= strskip(next,"(sense_body ",next) &&
			str2val(next,sb.time,next) &&
			strskip(next,"(view_mode ",next) &&
			manual_parse_view_mode(next, sb.view_quality, sb.view_width, next) &&
			strskip(next,')',next);
	if(ClientOptions::server_version >= 13)
		res &= strskip(next,"(stamina ",next) && str2val(next,sb.stamina,next) && str2val(next,sb.effort,next) && str2val(next,sb.stamina_capacity,next) && strskip(next,')',next);
	else
		res &= strskip(next,"(stamina ",next) && str2val(next,sb.stamina,next) && str2val(next,sb.effort,next) && strskip(next,')',next);
	res &=
			strskip(next,"(speed ",next) && str2val(next,sb.speed_value,next) && str2val(next,sb.speed_angle,next) && strskip(next,')',next) &&
			strskip(next,"(head_angle ",next) && str2val(next,sb.neck_angle,next) && strskip(next,')',next) &&
			strskip(next,"(kick ",next) && str2val(next,sb.kick_count,next) && strskip(next,')',next) &&
			strskip(next,"(dash ",next) && str2val(next,sb.dash_count,next) && strskip(next,')',next) &&
			strskip(next,"(turn ",next) && str2val(next,sb.turn_count,next) && strskip(next,')',next) &&
			strskip(next,"(say ",next) && str2val(next,sb.say_count,next) && strskip(next,')',next) &&
			strskip(next,"(turn_neck ",next) && str2val(next,sb.turn_neck_count,next) && strskip(next,')',next) &&
			strskip(next,"(catch ",next) && str2val(next,sb.catch_count,next) && strskip(next,')',next) &&
			strskip(next,"(move ",next) && str2val(next,sb.move_count,next) && strskip(next,')',next) &&
			strskip(next,"(change_view ",next) && str2val(next,sb.change_view_count,next) && strskip(next,')',next);

	//LOG_WM(sb.time,0,<<"MY STAMINA= " << sb.stamina);
#if 1 //extension for server version 8.04
	if ( strskip(next,')',dum ) )  //this is to support aserver version 7 (or version 7 servers in general)
		return true;

	/*(arm (movable 0) (expires 0) (target 0 0) (count 0))
    (focus (target none) (count 0)) 
    (tackle (expires 0) (count 0)))
	 */
	res= res && strskip(next,"(arm (movable",next) && str2val(next,sb.arm_movable,next) &&
			strskip(next,") (expires",next) && str2val(next,sb.arm_expires,next) &&
			strskip(next,") (target",next) && str2val(next,sb.arm_target_x,next) && str2val(next,sb.arm_target_y,next) &&
			strskip(next,") (count",next) && str2val(next,sb.arm_count,next) && strskip(next,')',next) &&
			strskip(next,") (focus (target",next);

	if ( ! res)
		return false;

	if ( strskip(next,"none",dum) ) {
		sb.focus_target=  -1;
		next= dum;
	}
	else {
		strspace(next,next);
		if ((next[0] == 'l' && WM::my_side == left_SIDE) || (next[0] == 'r' && WM::my_side == right_SIDE)) {
			next++;
			res= str2val(next,sb.focus_target,next);
		}
		else {
			next++;
			res= str2val(next,sb.focus_target,next);
			ERROR_OUT << ID << "\nI have attention to player " << sb.focus_target << " of opponent team";
			sb.focus_target= 0;
		}
	}
	res= res && strskip(next,") (count",next) && str2val(next,sb.focus_count,next) && strskip(next,')',next) &&
			strskip(next,") (tackle (expires",next) && str2val(next,sb.tackle_expires,next) &&
			strskip(next,") (count",next) && str2val(next,sb.tackle_count,next) && strskip(next,')',next) &&
			strskip(next,')',next);

	// initialize collision flags to be 0
	sb.coll_ball= 0;
	sb.coll_player= 0;
	sb.coll_post= 0;
	// if we use protocol version 12 we may parse collision info
	if(ClientOptions::server_version >= 12)
	{
		res = res && strskip(next,"(collision ",next);
		if(strskip(next,"none)",dum))
		{
			next= dum;
		}
		else
		{
			if(strskip(next,"(ball)", dum))
			{
				sb.coll_ball= 1;
				next= dum;
			}
			if(strskip(next,"(player)", dum))
			{
				sb.coll_player= 1;
				next= dum;
			}
			if(strskip(next,"(post)", dum))
			{
				sb.coll_post= 1;
				next= dum;
			}
			// ok now skip till end of collision
			res = res && strskip(next,")",next);
		}
	}
	//JTS10: parse foul messages from sense body
	if(ClientOptions::server_version >= 14)
	{
		res &= strskip(next,"(foul ",next) && strskip(next,"(charged ",next) && str2val(next, sb.foul_cycles, next) &&
				strskip(next,") (card ",next);
		if ( strskip(next,"none)",dum) )
			sb.card = NO_CARD;
		else if ( strskip(next,"yellow)",dum) )
			sb.card = YELLOW_CARD;
		else if ( strskip(next,"red)",dum) )
			sb.card = RED_CARD;
		else
			res = false; // none | yellow | red  should be exhaustive cases
		next = dum;
		res &= strskip(next,")",next); // skip foul closing brace
		if ( sb.foul_cycles > 0 )
			sb.fouled = true; // set flag if in idle cycle
	}
	// finally skip last closing braces
	res = res && strskip(next,")",next);
#endif  
	return res;
}


bool SensorParser::manual_parse_hear(const char * str, Msg_hear & hear, char const* & next, bool & reset_int) {
	const char * dum;
	hear.reset();
	LOG_WM(hear.time,0, << "\nHearParsing0= " << str );
	if ( !strskip(str,"(hear ",next) )
		return false;

	//cout << "\nCOMM= " << next;

	int number;

	if ( !str2val(next,hear.time,next) )
		return false;


	if ( strskip(next,"referee",dum) ) {
		next= dum;
		int score_in_pm;
		if ( ! manual_parse_play_mode(next,hear.play_mode,score_in_pm, next) ) {
			// JTS 10: parse new yellow_card -- red_card messages
			if ( manual_parse_card(next,hear.card,next) ) {
				hear.card_update = true;
				return true;
			} 
			else {
				return false;
			} 
		}
		hear.play_mode_upd= true;
		if ( score_in_pm > 0 ) {
			if (hear.play_mode == PM_goal_l){
				if ( WM::my_side == left_SIDE) {
					hear.my_score_upd= true;
					hear.my_score= score_in_pm;
				}
				else {
					hear.his_score_upd= true;
					hear.his_score= score_in_pm;
				}
			}
			else{
				if (hear.play_mode == PM_goal_r){
					if ( WM::my_side == right_SIDE) {
						hear.my_score_upd= true;
						hear.my_score= score_in_pm;
					}
					else {
						hear.his_score_upd= true;
						hear.his_score= score_in_pm;
					}
				}
			}
		}
		if ( !strskip(next,")",next ) )
			return false;

		return true;
	}
	else if ( strskip(next,"self",dum) ) { //communication from myself
		next= dum;
		return true;
	}
	else if ( str2val(next,number,dum) ) { //communication from my teammate or from an opponent
		next= dum;
		LOG_WM(hear.time,0, << "\nHearParsing1= " << str << " now at " << next );

		//cout << "\ncomm= " << next;
		if ( strskip(next,"opp",dum) ) {
			//cout << "\n got opp message " << str;
			WARNING_OUT << ID << " I still get messages from opponents";
			return true;  //ignore opponents messages
		}

		if ( ! strskip(next,"our",next) )
			return false;

		if ( ! str2val(next,number,next) )  //this is the number of the communicating player
			return false;

		if ( ! strskip(next,'"',next) )
			return false;
		if ( ! manual_parse_teamcomm(next,hear.teamcomm,next) ) {
			WARNING_OUT << ID << "\nproblems with teamcomm";
			return false;
		}
		hear.teamcomm.from= number;

		if ( ! strskip(next,'"',next) )
			return false;

		//cout << "hear.teamcomm_upd= true;";
		hear.teamcomm_upd= true;

		//cout << "\nteamcomm ok " << hear.teamcomm.from << " at time " << hear.time << " = " << str;
		if ( ! strskip(next,')',next) )
			return false;

		LOG_WM(hear.time,0, << "done with HearParsing");
		return true;
	}
	else if ( strskip(next,"online_coach_",dum) ) {
		next= dum;
		if ( WM::my_side == left_SIDE ) {
			if ( strskip(next,"right",dum) )
				return true; //don't read online coach messages of the opponent
			if ( ! strskip(next,"left",next) )
				return false; //wrong message type
		}
		else {
			if ( strskip(next,"left",dum) )
				return true; //don't read online coach messages of the opponent
			if ( ! strskip(next,"right",next) )
				return false; //wrong message type
		}
		// first, try parsing the coach message as a message based on the official coach language
		bool res = manual_parse_my_online_coachcomm_clbased(next,hear.my_online_coachcomm,next);
		// second, if not successful, try also parsing the coach message the usual way (freeform)
		// (note: this should be kept for downward compatibilty)
		if (!res)
		  res = manual_parse_my_online_coachcomm(next,hear.my_online_coachcomm,next);
		if (!res)
			return false;
		hear.my_online_coachcomm_upd= true;
		hear.my_online_coachcomm.time= hear.time;

		//INFO_OUT << ID << "\nsuccessfully parsed coach message:" << str << "\nresult->" << hear;
		LOG_WM( hear.time, 0, << "Successfully parsed coach message:" << str << " || result -> " << hear );

		return true;
	}
	else if ( strskip(next,"coach",dum) ) {
		next= dum;
		bool res= manual_parse_my_trainercomm(next);
		if (!res)
			return false;
		//INFO_OUT << ID << "\nsuccessfully parsed coach message:" << str << "\nresult->" << hear;
		reset_int=true;
		return true;
	}
	else if ( strskip(next,"our",dum) ) {
		hear.teamcomm_partial_upd= true;
		return true;
	}

	return false;
}

bool parse_score(const char * str, int & score, const char *& next) {
	if ( *str != '_' )
		return true;

	str++;
	if ( !str2val(str,score,next) )
		return false;
	return true;
}

bool SensorParser::manual_parse_card(const char * str, Msg_card &mc, const char *& next) {
	// JTS 10: insert to parse yellow and red cards from referee message
	mc.reset();
	const char *dum;

	while (str[0] == ' ')
		str++;

	if ( !str2val(str, mc.time, dum ) ) 
		return false; // no timestamp, thus not a card message
	str = dum;
	if ( strskip(str,"yellow_card",dum) ) {
		mc.type = YELLOW_CARD;
	}
	else if ( strskip(str,"red_card",dum) ) {
		mc.type = RED_CARD;
	}
	else {
		return false;
	}
	str = dum;
	if ( strskip(str,"_l_",dum) ) {
		mc.side = left_SIDE;
	}
	else if ( strskip(str,"_r_",dum) ) {
		mc.side = right_SIDE;
	}
	else {
		return false;
	}
	str = dum;
	if ( ! str2val(str, mc.card_player, str) ) 
		return false;
	// --> DONE
	next = str;
	return true;
}

bool SensorParser::manual_parse_play_mode(const char * str, PlayMode & pm, int & score, const char *& next) {
	score= -1;
	while (str[0] == ' ')
		str++;
	/* 1234567890123456789 */
	if      (strncmp("before_kick_off", str, 15) == 0) { pm= PM_before_kick_off; str+= 15; }
	else if (strncmp("time_over", str, 9) == 0)        { pm= PM_time_over; str+= 9; }
	else if (strncmp("play_on", str, 7) == 0)          { pm= PM_play_on; str+= 7; }
	else if (strncmp("kick_off_l", str, 10) == 0)      { pm= PM_kick_off_l; str+= 10; }
	else if (strncmp("kick_off_r", str, 10) == 0)      { pm= PM_kick_off_r; str+= 10; }
	else if (strncmp("kick_in_l", str, 9) == 0)        { pm= PM_kick_in_l; str+= 9; }
	else if (strncmp("foul_r", str, 6) == 0)           { pm= PM_kick_in_l; str+= 6; }  //match foul_r to kick_in_l
	else if (strncmp("kick_in_r", str, 9) == 0)        { pm= PM_kick_in_r; str+= 9; }
	else if (strncmp("foul_l", str, 6) == 0)           { pm= PM_kick_in_r; str+= 6; }  //match foul_l to kick_in_r
	else if (strncmp("free_kick_l", str, 11) == 0)     { pm= PM_free_kick_l; str+= 11; }
	else if (strncmp("foul_charge_r", str, 13) == 0)   { pm= PM_free_kick_l; str+= 13; }  //JTS10 match foul_charge_r to free_kick_l
	else if (strncmp("free_kick_r", str, 11) == 0)     { pm= PM_free_kick_r; str+= 11; }
	else if (strncmp("foul_charge_l", str, 13) == 0)           { pm= PM_free_kick_r; str+= 13; }  //JTS10 match foul_charge_l to free_kick_r
	else if (strncmp("indirect_free_kick_l", str, 20) == 0)  { pm= PM_free_kick_l; str+= 20; } //no dinstiction to free_kick_* at the moment, go2003
	else if (strncmp("indirect_free_kick_r", str, 20) == 0)  { pm= PM_free_kick_r; str+= 20; } //no dinstiction to free_kick_* at the moment, go2003
	else if (strncmp("corner_kick_l", str, 13) == 0)         { pm= PM_corner_kick_l; str+= 13; }
	else if (strncmp("corner_kick_r", str, 13) == 0)         { pm= PM_corner_kick_r; str+= 13; }
	else if (strncmp("goal_kick_l", str, 11) == 0)           { pm= PM_goal_kick_l; str+= 11; }
	else if (strncmp("goal_kick_r", str, 11) == 0)           { pm= PM_goal_kick_r; str+= 11; }
	else if (strncmp("goal_l", str, 6) == 0) {
		pm= PM_goal_l; str+= 6;
		if ( str[0] == '_' ) {
			str++;
			const char * dum;
			if ( !str2val(str,score,dum) )
				return false;
			str= dum;
		}
	}
	else if (strncmp("goal_r", str, 6) == 0) {
		pm= PM_goal_r; str+= 6;
		if ( str[0] == '_' ) {
			str++;
			const char * dum;
			if ( !str2val(str,score,dum) )
				return false;
			str= dum;
		}
	}
	else if (strncmp("drop_ball", str, 9) == 0)              { pm= PM_drop_ball; str+= 9; }
	else if (strncmp("offside_l", str, 9) == 0)              { pm= PM_offside_l; str+= 9; }
	else if (strncmp("offside_r", str, 9) == 0)              { pm= PM_offside_r; str+= 9; }
	else if (strncmp("goalie_catch_ball_l", str, 19) == 0)   { pm= PM_goalie_catch_ball_l; str+= 19; }
	else if (strncmp("goalie_catch_ball_r", str, 19) == 0)   { pm= PM_goalie_catch_ball_r; str+= 19; }
	else if (strncmp("free_kick_fault_l",str, 17) == 0)      { pm= PM_free_kick_fault_l; str+= 17; }
	else if (strncmp("free_kick_fault_r",str, 17) == 0)      { pm= PM_free_kick_fault_r; str+= 17; }
	else if (strncmp("back_pass_l",str, 11) == 0)            { pm= PM_back_pass_l; str+= 11; }
	else if (strncmp("back_pass_r",str, 11) == 0)            { pm= PM_back_pass_r; str+= 11; }
	/* 1234567890123456789 */
	else if (strncmp("penalty_kick_l",str,14) == 0)          { pm= PM_penalty_kick_l; str+= 14; }
	else if (strncmp("penalty_kick_r",str,14) == 0)          { pm= PM_penalty_kick_r; str+= 14; }
	else if (strncmp("catch_fault_l",str,13) == 0)           { pm= PM_catch_fault_l; str+= 13; }
	else if (strncmp("catch_fault_r",str,13) == 0)           { pm= PM_catch_fault_r; str+= 13; }
	else if (strncmp("indirect_free_kick_l",str,20) == 0)    { pm= PM_indirect_free_kick_l; str+= 20; }
	else if (strncmp("indirect_free_kick_r",str,20) == 0)    { pm= PM_indirect_free_kick_r; str+= 20; }
	else if (strncmp("penalty_setup_l",str,15) == 0)         { pm= PM_penalty_setup_l; str+= 15; }
	else if (strncmp("penalty_setup_r",str,15) == 0)         { pm= PM_penalty_setup_r; str+= 15; }
	else if (strncmp("penalty_ready_l",str,15) == 0)         { pm= PM_penalty_ready_l; str+= 15; }
	else if (strncmp("penalty_ready_r",str,15) == 0)         { pm= PM_penalty_ready_r; str+= 15; }
	else if (strncmp("penalty_taken_l",str,15) == 0)         { pm= PM_penalty_taken_l; str+= 15; }
	else if (strncmp("penalty_taken_r",str,15) == 0)         { pm= PM_penalty_taken_r; str+= 15; }
	else if (strncmp("penalty_miss_l",str,14) == 0)          { pm= PM_penalty_miss_l; str+= 14; }
	else if (strncmp("penalty_miss_r",str,14) == 0)          { pm= PM_penalty_miss_r; str+= 14; }
	else if (strncmp("penalty_score_l",str,15) == 0)         { pm= PM_penalty_score_l; str+= 15; }
	else if (strncmp("penalty_score_r",str,15) == 0)         { pm= PM_penalty_score_r; str+= 15; }

	else if (strncmp("penalty_onfield_l",str,17) == 0)       { pm= PM_penalty_onfield_l; str+= 17; }
	else if (strncmp("penalty_onfield_r",str,17) == 0)       { pm= PM_penalty_onfield_r; str+= 17; }
	else if (strncmp("penalty_foul_l",str,14) == 0)          { pm= PM_penalty_foul_l; str+= 14; }
	else if (strncmp("penalty_foul_r",str,14) == 0)          { pm= PM_penalty_foul_r; str+= 14; }
	else if (strncmp("penalty_winner_l",str,16) == 0)        { pm= PM_penalty_winner_l; str+= 16; }
	else if (strncmp("penalty_winner_r",str,16) == 0)        { pm= PM_penalty_winner_r; str+= 16; }
	else if (strncmp("penalty_draw",str,12) == 0)            { pm= PM_penalty_draw; str+= 16; }
	/* 1234567890123456789 */
	else if (strncmp("half_time",str, 9) == 0) { pm= PM_half_time; str+= 9; }
	else if (strncmp("time_up",str, 7) == 0) { pm= PM_time_up; str+= 7; }
	else if (strncmp("time_extended",str, 13) == 0) { pm= PM_time_extended; str+= 13; }
	/* 1234567890123456789 */
	else {
		pm= PM_Null;
		next= str;
		return false;
	}

	next= str;
	return true;
}

bool SensorParser::manual_parse_view_mode(const char * str, int & quality, int & width, const char *& next) {
	while (str[0] == ' ')
		str++;

	if (strncmp("high normal",str,11) == 0 ) {
		str+= 11;
		quality= HIGH;
		width= NORMAL;
	}
	else if (strncmp("high wide",str,9) == 0 ) {
		str+= 9;
		quality= HIGH;
		width= WIDE;
	}
	else if (strncmp("high narrow",str,11) == 0 ) {
		str+= 11;
		quality= HIGH;
		width= NARROW;
	}
	else if (strncmp("low normal",str,10) == 0 ) {
		str+= 10;
		quality= LOW;
		width= NORMAL;
	}
	else if (strncmp("low wide",str,8) == 0 ) {
		str+= 8;
		quality= LOW;
		width= WIDE;
	}
	else if (strncmp("low narrow",str,10) == 0 ) {
		str+= 10;
		quality= LOW;
		width= NARROW;
	}
	else
		return false;

	next= str;
	return true;
}

bool SensorParser::manual_parse_see_object(const char * str, ParseObject & pobj, const char *& next) {
#if 1
	const char * dum;
	pobj.res= pobj.UNKNOWN;
	pobj.number= -1;
	pobj.x= -1000.0;
	pobj.y= -1000.0;

	int num= 0;
	const char XXX = -10;
	const char _F_ = -1;
	const char _P_ = -2;
	const char _G_ = -3;
	const char _C_ = -4;
	const char _L_ = -5; //line and left
	const char _R_ = -6;
	const char _T_ = -7;
	const char _B_ = -8;

	int obj[5];

	while ( str[0] == ' ')
		str++;
	if (str[0] != '(')
    {
        std::cout << "\nparse_error";
        return false;
    }
	str++;
	while (num<4) {
		if (str[0] == 'f' || str[0] == 'F') {
			str++;
			obj[num]= _F_;
		}
		else if (str[0] == 'p' || str[0] == 'P') {
			str++;
			obj[num]= _P_;
		}
		else if (str[0] == 'g' || str[0] == 'G') {
			str++;
			obj[num]= _G_;
		}
		else if (str[0] == 'l' || str[0] == 'L') {
			str++;
			obj[num]= _L_;
		}
		else if (str[0] == 'r') {
			str++;
			obj[num]= _R_;
		}
		else if (str[0] == 't') {
			str++;
			obj[num]= _T_;
		}
		else if (str[0] == 'b' || str[0] == 'B') {
			str++;
			obj[num]= _B_;
		}
		else if (str[0] == 'c') {
			str++;
			obj[num]= _C_;
		}
		else if (str[0] == '0') {
			str+= 1;
			obj[num]= 0;
		}
		else if (str[0] == '1' && str[1] == '0') {
			str+= 2;
			obj[num]= 10;
		}
		else if (str[0] == '2' && str[1] == '0') {
			str+= 2;
			obj[num]= 20;
		}
		else if (str[0] == '3' && str[1] == '0') {
			str+= 2;
			obj[num]= 30;
		}
		else if (str[0] == '4' && str[1] == '0') {
			str+= 2;
			obj[num]= 40;
		}
		else if (str[0] == '5' && str[1] == '0') {
			str+= 2;
			obj[num]= 50;
		}
		else
        {
            std::cout << "\nparse_error";
            return false;
        };
		num++;

		if (str[0] == '\0')
        {
            std::cout << "\nparse_error";
            return false;
        };
		if (str[0] == ' ')
			str++;
		if (str[0] == ')' || str[0] == '"')
			break;
	}

	if (num<=0)
    {
        std::cout << "\nparse_error";
        return false;
    };

	obj[num]= XXX;
	num++;

	const double LEFT_LINE= -PITCH_LENGTH/2.0;
	const double LEFT_OUTSIDE=  LEFT_LINE - PITCH_MARGIN;
	const double LEFT_PENALTY=  LEFT_LINE + PENALTY_AREA_LENGTH;
	const double RIGHT_LINE= PITCH_LENGTH/2.0;
	const double RIGHT_OUTSIDE=  RIGHT_LINE + PITCH_MARGIN;
	const double RIGHT_PENALTY=  RIGHT_LINE - PENALTY_AREA_LENGTH;
	const double TOP_LINE = PITCH_WIDTH/2.0;
	const double TOP_OUTSIDE = TOP_LINE + PITCH_MARGIN;
	const double TOP_PENALTY = PENALTY_AREA_WIDTH/2.0;
	const double TOP_GOAL = ServerOptions::goal_width/2;
	const double BOTTOM_LINE = -PITCH_WIDTH/2.0;
	const double BOTTOM_OUTSIDE = BOTTOM_LINE - PITCH_MARGIN;
	const double BOTTOM_PENALTY = -PENALTY_AREA_WIDTH/2.0;
	const double BOTTOM_GOAL = -ServerOptions::goal_width/2;

	if (obj[0] == _P_) { //the object was a player, read his attributes, the date is read later on
		if (str[0] == ')') {
			str++;
			pobj.res= pobj.PLAYER_UNKNOWN;
			next= str;
			return true;
		}

		if (str[0] == '"') {
			str++;
			pobj.res= pobj.PLAYER_HIS_TEAM;
			if ( strncmp(str,WM::my_team_name,WM::my_team_name_len) == 0 ) {
				str+= WM::my_team_name_len;
				if (str[0] == '"')  //this check is necessary, because oppenent's name can have my_team_name as prefix
					pobj.res= pobj.PLAYER_MY_TEAM;
			}

			while ( str[0] != '\0' && str[0] != '"' )
				str++;

			if ( str[0] == '\0' )
				return false;

			str++;
			if (str[0] == ')') {
				str++;
				pobj.number= 0;
				next= str;
				return true;
			}

			if ( !str2val(str,pobj.number,dum) )
				return false;

			str= dum;
			if (str[0] == ')') {
				str++;
				next= str;
				return true;
			}
			if ( !strskip(str,"goalie",dum) )
				return false;
			str= dum;
			if (pobj.res == pobj.PLAYER_MY_TEAM)
				pobj.res= pobj.PLAYER_MY_TEAM_GOALIE;
			else if (pobj.res == pobj.PLAYER_HIS_TEAM)
				pobj.res= pobj.PLAYER_HIS_TEAM_GOALIE;
			if (str[0] != ')')
				return false;
			str++;
			next= str;
			return true;
		}
	}
	// 0 --------------------------------------------
	else if (obj[0] == _F_) {
		// 1 --------------------------------------------
		if (obj[1] == _P_) {
			// 2 --------------------------------------------
			if (obj[2] == _L_) {
				if (obj[3] == _T_) {
					pobj.res= pobj.MARKER;
					pobj.x = LEFT_PENALTY;
					pobj.y = TOP_PENALTY;
				}
				else if (obj[3] == _C_) {
					pobj.res= pobj.MARKER;
					pobj.x = LEFT_PENALTY;
					pobj.y = 0.0;
				}
				if (obj[3] == _B_) {
					pobj.res= pobj.MARKER;
					pobj.x = LEFT_PENALTY;
					pobj.y = BOTTOM_PENALTY;
				}
			}
			// 2 --------------------------------------------
			else if (obj[2] == _R_) {
				if (obj[3] == _T_) {
					pobj.res= pobj.MARKER;
					pobj.x = RIGHT_PENALTY;
					pobj.y = TOP_PENALTY;
				}
				else if (obj[3] == _C_) {
					pobj.res= pobj.MARKER;
					pobj.x = RIGHT_PENALTY;
					pobj.y = 0.0;
				}
				if (obj[3] == _B_) {
					pobj.res= pobj.MARKER;
					pobj.x = RIGHT_PENALTY;
					pobj.y = BOTTOM_PENALTY;
				}
			}
		}
		// 1 --------------------------------------------
		if (obj[1] == _G_) {
			// 2 --------------------------------------------
			if (obj[2] == _L_) {
				if (obj[3] == _T_) {
					pobj.res= pobj.MARKER;
					pobj.x = LEFT_LINE;
					pobj.y = TOP_GOAL;
				}
				else if (obj[3] == _B_) {
					pobj.res= pobj.MARKER;
					pobj.x = LEFT_LINE;
					pobj.y = BOTTOM_GOAL;
				}
			}
			// 2 --------------------------------------------
			else if (obj[2] == _R_) {
				if (obj[3] == _T_) {
					pobj.res= pobj.MARKER;
					pobj.x = RIGHT_LINE;
					pobj.y = TOP_GOAL;
				}
				else if (obj[3] == _B_) {
					pobj.res= pobj.MARKER;
					pobj.x = RIGHT_LINE;
					pobj.y = BOTTOM_GOAL;
				}
			}
		}
		// 1 --------------------------------------------
		else if (obj[1]== _L_) {
			// 2 --------------------------------------------
			if ( obj[2] == 0) {
				pobj.res= pobj.MARKER;
				pobj.x = LEFT_OUTSIDE;
				pobj.y = 0.0;
			}
			// 2 --------------------------------------------
			else if ( obj[2] == _T_) {
				if ( obj[3] == XXX) {
					pobj.res= pobj.MARKER;
					pobj.x = LEFT_LINE;
					pobj.y = TOP_LINE;
				}
				else if ( obj[3] > 0 && obj[3] <= 30) {
					pobj.res= pobj.MARKER;
					pobj.x= LEFT_OUTSIDE;
					pobj.y = double(obj[3]);
				}
			}
			// 2 --------------------------------------------
			else if ( obj[2] == _B_) {
				if ( obj[3] == XXX) {
					pobj.res= pobj.MARKER;
					pobj.x =  LEFT_LINE;
					pobj.y =  BOTTOM_LINE;
				}
				else if ( obj[3] > 0 && obj[3] <= 30) {
					pobj.res= pobj.MARKER;
					pobj.x = LEFT_OUTSIDE;
					pobj.y = double(-obj[3]);
				}
			}
		}
		// 1 --------------------------------------------
		else if (obj[1] == _R_) {
			// 2 --------------------------------------------
			if ( obj[2] == 0) {
				pobj.res= pobj.MARKER;
				pobj.x = RIGHT_OUTSIDE;
				pobj.y = 0.0;
			}
			// 2 --------------------------------------------
			else if ( obj[2] == _T_) {
				if ( obj[3] == XXX) {
					pobj.res= pobj.MARKER;
					pobj.x = RIGHT_LINE;
					pobj.y = TOP_LINE;
				}
				else if ( obj[3] > 0 && obj[3] <= 30 ) {
					pobj.res= pobj.MARKER;
					pobj.x= RIGHT_OUTSIDE;
					pobj.y = double(obj[3]);
				}
			}
			// 2 --------------------------------------------
			else if ( obj[2] == _B_) {
				if ( obj[3] == XXX) {
					pobj.res= pobj.MARKER;
					pobj.x =  RIGHT_LINE;
					pobj.y =  BOTTOM_LINE;
				}
				else if ( obj[3] > 0 && obj[3] <= 30) {
					pobj.res= pobj.MARKER;
					pobj.x = RIGHT_OUTSIDE;
					pobj.y = double(-obj[3]);
				}
			}
		}
		// 1 --------------------------------------------
		else if (obj[1]== _C_) {
			// 2 --------------------------------------------
			if ( obj[2] == XXX) {
				pobj.res= pobj.MARKER;
				pobj.x = 0.0;
				pobj.y = 0.0;
			}
			// 2 --------------------------------------------
			else if ( obj[2] == _T_) {
				pobj.res= pobj.MARKER;
				pobj.x = 0.0;
				pobj.y = TOP_LINE;
			}
			// 2 --------------------------------------------
			else if ( obj[2] == _B_) {
				pobj.res= pobj.MARKER;
				pobj.x = 0.0;
				pobj.y = BOTTOM_LINE;
			}
		}
		// 1 --------------------------------------------
		else if (obj[1]== _T_) {
			if (obj[2] == 0) {
				pobj.res= pobj.MARKER;
				pobj.x = 0.0;
				pobj.y= TOP_OUTSIDE;
			}
			else if (obj[2] == _L_ && obj[3] > 0 && obj[3] <= 50) {
				pobj.res= pobj.MARKER;
				pobj.x = double(-obj[3]);
				pobj.y= TOP_OUTSIDE;
			}
			else if (obj[2] == _R_ && obj[3] > 0 && obj[3] <= 50) {
				pobj.res= pobj.MARKER;
				pobj.x = double(obj[3]);
				pobj.y= TOP_OUTSIDE;
			}
		}
		// 1 --------------------------------------------
		else if (obj[1]== _B_) {
			if (obj[2] == 0) {
				pobj.res= pobj.MARKER;
				pobj.x = 0.0;
				pobj.y= BOTTOM_OUTSIDE;
			}
			else if (obj[2] == _L_ && obj[3] > 0 && obj[3] <= 50) {
				pobj.res= pobj.MARKER;
				pobj.x = double(-obj[3]);
				pobj.y= BOTTOM_OUTSIDE;
			}
			else if (obj[2] == _R_ && obj[3] > 0 && obj[3] <= 50) {
				pobj.res= pobj.MARKER;
				pobj.x = double(obj[3]);
				pobj.y= BOTTOM_OUTSIDE;
			}
		}
	}
	// 0 --------------------------------------------
	else if (obj[0] == _G_) {
		if (obj[1] == _L_) {
			pobj.res= pobj.MARKER;
			pobj.x = LEFT_LINE;
			pobj.y = 0.0;
		}
		else if (obj[1] == _R_) {
			pobj.res= pobj.MARKER;
			pobj.x = RIGHT_LINE;
			pobj.y = 0.0;
		}
	}
	// 0 --------------------------------------------
	else if (obj[0] == _B_) {
		pobj.res= pobj.BALL_OBJECT;
		if (str[0] != ')')
			return false;
		str++;
		next= str;
		return true;
	}
	// 0 --------------------------------------------
	else if (obj[0] == _L_) {
		if (obj[1] == _L_) {
			pobj.res= pobj.MARKER_LINE;
			pobj.x = LEFT_LINE;
			pobj.y = 0.0;
		}
		else if (obj[1] == _R_) {
			pobj.res= pobj.MARKER_LINE;
			pobj.x = RIGHT_LINE;
			pobj.y = 0.0;
		}
		if (obj[1] == _T_) {
			pobj.res= pobj.MARKER_LINE;
			pobj.x = 0.0;
			pobj.y = TOP_LINE;
		}
		else if (obj[1] == _B_) {
			pobj.res= pobj.MARKER_LINE;
			pobj.x = 0.0;
			pobj.y = BOTTOM_LINE;
		}
	}
	if (str[0] != ')')
		return false;

	str++;
	next= str;
	return true;
#endif
}

int str2val(const char * str, NOT_NEEDED & val, const char* & next) {
	return strfind(str,')',next);
}

bool SensorParser::manual_parse_player_type(const char * str, Msg_player_type & pt) {
	const char * origin= str;
//	std::cerr << origin << std::endl;
	bool res= strskip(str,"(player_type ",str) &&
			strskip(str,"(id",str) && str2val(str,pt.id,str) && strskip(str,')',str) &&
			strskip(str,"(player_speed_max",str) && str2val(str,pt.player_speed_max,str) && strskip(str,')',str) &&
			strskip(str,"(stamina_inc_max",str) && str2val(str,pt.stamina_inc_max,str) && strskip(str,')',str) &&
			strskip(str,"(player_decay",str) && str2val(str,pt.player_decay,str) && strskip(str,')',str) &&
			strskip(str,"(inertia_moment",str) && str2val(str,pt.inertia_moment,str) && strskip(str,')',str) &&
			strskip(str,"(dash_power_rate",str) && str2val(str,pt.dash_power_rate,str) && strskip(str,')',str) &&
			strskip(str,"(player_size",str) && str2val(str,pt.player_size,str) && strskip(str,')',str) &&
			strskip(str,"(kickable_margin",str) && str2val(str,pt.kickable_margin,str) && strskip(str,')',str) &&
			strskip(str,"(kick_rand",str) && str2val(str,pt.kick_rand,str) && strskip(str,')',str) &&
			strskip(str,"(extra_stamina",str) && str2val(str,pt.extra_stamina,str) && strskip(str,')',str) &&
			strskip(str,"(effort_max",str) && str2val(str,pt.effort_max,str) && strskip(str,')',str) &&
			strskip(str,"(effort_min",str) && str2val(str,pt.effort_min,str) && strskip(str,')',str);
	if ( ClientOptions::server_version >= 14.0 ) {
		res&=
				strskip(str,"(kick_power_rate",str) && str2val(str,pt.kick_power_rate,str) && strskip(str,')',str) &&
				strskip(str,"(foul_detect_probability",str) && str2val(str,pt.foul_detect_probability,str) && strskip(str,')',str) &&
				strskip(str,"(catchable_area_l_stretch",str) && str2val(str,pt.catchable_area_l_stretch,str) && strskip(str,')',str);
	}
	res &=
			strskip(str,')',str);

	if (!res) {
		ERROR_OUT << ID << "\nparse error:\n";
		show_parser_error_point(ERROR_STREAM,origin,str);
	}
	return res;
}

bool SensorParser::manual_parse_change_player_type(const char * str, Msg_change_player_type & cpt) {
	cpt.reset();
	bool res= strskip(str,"(change_player_type ",str) && str2val(str,cpt.number,str);
	if (!res)
		return false;

	if ( strskip(str,')',str) )
		return true;

	res= str2val(str,cpt.type,str) && strskip(str,')',str);
	return res;
}

/*
 * Method to parse a single coach language directive. These directives
 * are structured as follows:
 *   ( [do|dont] [our|opp] {[num]} (action) )
 * where action can be one of the following
 *   (hold) 
 *   (shoot)
 *   (home (rec (pt x y) (pt x y)) )
 *   (mark {[unumset]} )
 *   (htype [type])
 * This is just a subset of the set of actions allowed by coach language.
 * See, the Soccer Server Manual for all details.
 */
bool
SensorParser::manual_parse_coach_lang_directive( const char * str,
                                                 Msg_my_online_coachcomm & moc,
                                                 const char *& next)
{
  LOG_WM(WSinfo::ws->time, 3,<<"Parse a single CL directive from: "<<str);
  const char * dum;
  bool isDo = false, isDont = false, isOpp = false, isOur = false;
  int  actorPl, targetPl, hpType;
  double val;
  if ( !strskip(str,"(",next) )
    return false;
  if ( strncmp(next,"do ",3) == 0)
  {
    isDo = true;
    strskip(next,"do",next);
  }
  if ( strncmp(next,"dont ",5) == 0)
  {
    isDont = true;
    strskip(next,"dont",next);
  }
  if ( isDont == false && isDo == false ) return false;
  if ( strskip(next,"opp",dum) )
  {
    isOpp = true;
    strskip(next,"opp",next);
  }
  if ( strskip(next,"our",dum) )
  {
    isOur = true;
    strskip(next,"our",next);
  }
  if ( isOur == false && isOpp == false ) return false;
  if ( !strskip(next,"{",next) ) return false;
  if ( !str2val(next,actorPl,next) ) return false;
  if ( !strskip(next,"}",next) ) return false;
  if ( !strskip(next,"(",next) ) return false;

  if ( strskip(next,"hold",dum) )
  {
    //information about his goalie player number
    if ( !strskip(next,"hold)",next) ) return false;
    moc.his_goalie_number_upd = true;
    moc.his_goalie_number = actorPl;
    LOG_WM(WSinfo::ws->time, 1,<<"  PARSED: his goalie number: " << actorPl);
  }
  else if ( strskip(next,"shoot",dum) )
  {
    //information about the opponent team name
    if ( !strskip(next,"shoot)",next) ) return false;
    WSinfo::set_current_opponent_identifier( TEAM_IDENTIFIER_BASE_CODE + actorPl );
    LOG_WM(WSinfo::ws->time, 1,<< "  PARSED: opponent team code: " << actorPl << " -> "
                               << WSinfo::get_current_opponent_identifier() );
  }
  else if ( strskip(next,"home",dum) )
  {
    //information about teammate's stamina capacity
    strskip(next,"home",next);
    if ( !strskip(next,"(rec",next) ) return false;

    int stamCap = 0;

    if ( !strskip(next,"(pt",next) ) return false;
    if ( !str2val(next,val,next) ) return false;
    stamCap += 10000000.0 * val;
    if ( !str2val(next,val,next) ) return false;
    stamCap += 100000.0 * val;
    if ( !strskip(next,")",next) ) return false;

    if ( !strskip(next,"(pt",next) ) return false;
    if ( !str2val(next,val,next) ) return false;
    stamCap += 1000.0 * val;
    if ( !str2val(next,val,next) ) return false;
    stamCap += 10.0 * val;
    if ( !strskip(next,")",next) ) return false;

    if ( !strskip(next,")",next) ) return false;
    if ( !strskip(next,")",next) ) return false;

    //0-based array, 1-based CL player naming
    moc.stamin_capacity_info[actorPl-1] = stamCap;
    moc.stamin_capacity_info_upd = true;
    LOG_WM(WSinfo::ws->time, 1,<< "  PARSED: stamina capacity info: " << actorPl << " -> " << stamCap );
  }
  else if ( strskip(next,"mark",dum) )
  {
    strskip(next,"mark",next);
    if ( !strskip(next,"{",next) ) return false;
    if ( !str2val(next,targetPl,next) ) return false;
    if ( !strskip(next,"}",next) ) return false;
    if ( !strskip(next,")",next) ) return false;
    if (targetPl < 1 || targetPl > 11)
    {
      LOG_WM(WSinfo::ws->time, 0,<< "PARSING ERROR: mark target player " << targetPl << ": out of range (1-11)");
    }
    moc.direct_opponent_assignment[actorPl-1]= targetPl;
    moc.direct_opponent_assignment_upd = true;
    LOG_WM(WSinfo::ws->time, 1,<< "  PARSED: DOA assignment: " << actorPl << " -> " << targetPl );
  }
  else if ( strskip(next,"htype",dum) )
  {
    strskip(next,"htype",next);
    if ( !str2val(next,hpType,next) ) return false;
    if ( hpType < -1 || hpType > ServerParam::number_of_player_types() ) return false;
    strskip(next,")",next);
    moc.his_player_types[actorPl-1]=  hpType;
    moc.his_player_types_upd = true;
    LOG_WM(WSinfo::ws->time, 1,<< "  PARSED: heterogeneous player type: " << actorPl << " -> " << hpType );
  }
  else
    return false;
  if ( !strskip(next,")",next) ) return false;
  return true;
}

/*
 * This method assumes that a coach message is formatted according to the 
 * coach language and tries to parse it correspondingly.
 * 
 * In general, a CL-based coach message is made up of a (larger) number 
 * of coach language directives, which are handled one by one by the 
 * method manual_parse_coach_lang_directive().
 * 
 * Note: At its current stage, the full power of the coach language is still
 * not exploited. Instead, as far as define messages are considered, we just
 * employ definerule versions. This is just a subset of what can be done
 * with the coach language.
 */
bool SensorParser::manual_parse_my_online_coachcomm_clbased(const char * str,
		Msg_my_online_coachcomm & moc,
		const char *& next)
{
	//printf("MANUALPARSEHEAR_CL: %s\n",str);
	LOG_WM(WSinfo::ws->time, 0, <<"MANUALPARSEHEAR_CL: "<<str);
	const char * next1;
	moc.reset();
	next = str;
	bool isFreeform = false, isInfoAdvice = false, isDefine = false;
    if ( strskip(str,"(info (6000 (true)",next) )
      isInfoAdvice = true;
    else if ( strskip(str,"(advice (6000 (true)",next) )
      isInfoAdvice = true;
    else if ( strskip(str,"(define (definerule R direc ((true)",next) )
      isDefine = true;
	else
	if ( strskip(str,"(freeform",next) )
      isFreeform = true;
	else
      return false;

    if (isFreeform)
    {
      LOG_WM(WSinfo::ws->time, 0,<<"Freeform message received. Parsing not supported.");
      next = str;
      return false;
    }
    else
    if (isInfoAdvice || isDefine)
    {
      // parsing of define / info / advice is identical
      LOG_WM(WSinfo::ws->time, 1,<<"Start parsing CL message: "<<next);
      moc.his_goalie_number_upd = false;
      moc.direct_opponent_assignment_upd = false;
      moc.his_player_types_upd = false;
      moc.stamin_capacity_info_upd = false;
      int numberOfParsedDirectives = 0;
      while ( strskip(next,"(do",next1) )
      {
        if ( !manual_parse_coach_lang_directive(next,moc,next) )
          break;
        numberOfParsedDirectives ++ ;
      }
      if ( numberOfParsedDirectives == 0 )
      {
        next = str;
        return false;
      }
    }
    else
    {
      LOG_WM(WSinfo::ws->time, 0, << "Severe error: Unknown coach directive.");
      LOG_ERR( 0, << "Severe error: Unknown coach directive.");
    }
    return true;
}

bool SensorParser::manual_parse_my_online_coachcomm(const char * str,
		Msg_my_online_coachcomm & moc,
		const char *& next)
{
	//printf("MANUALPARSEHEAR_OLD: %s\n",str);
	const char * next1;
	moc.reset();
	int tmp;
	int intBuf[2];
	next= str;
	//if ( ! str2val(next,moc.time,next) )
	//  return false;
	bool isFreeform = false;
	str= next;
	if ( strskip(str,"(info",next) )
		;
	else if ( strskip(str,"(advice",next) )
		;
	else if ( strskip(str,"(freeform",next) )
		isFreeform = true;
	else
		return false;

	if (isFreeform == false)
	{
		if ( ! strskip(next,'(',next) )
			return false;
		if ( ! str2val(next,tmp,next) )
			return false;
		if ( ! strskip(next,"(true)",next) )
			return false;
	}
	if ( ! strskip(next,'"',next) )
		return false;
	if ( strskip(next,"pt",next1) )
	{
		next = next1;

		for (int i=0; i<NUM_PLAYERS; i++)
		{
			switch (*next) //TG. extended to the increased number of player types
			{
			case '0' : moc.his_player_types[i]= 0; break;
			case '1' : moc.his_player_types[i]= 1; break;
			case '2' : moc.his_player_types[i]= 2; break;
			case '3' : moc.his_player_types[i]= 3; break;
			case '4' : moc.his_player_types[i]= 4; break;
			case '5' : moc.his_player_types[i]= 5; break;
			case '6' : moc.his_player_types[i]= 6; break;
			//TG08: more potential player types
			case '7' : moc.his_player_types[i]= 7; break;
			case '8' : moc.his_player_types[i]= 8; break;
			case '9' : moc.his_player_types[i]= 9; break;
			case 'A' : moc.his_player_types[i]=10; break;
			case 'B' : moc.his_player_types[i]=11; break;
			case 'C' : moc.his_player_types[i]=12; break;
			case 'D' : moc.his_player_types[i]=13; break;
			case 'E' : moc.his_player_types[i]=14; break;
			case 'F' : moc.his_player_types[i]=15; break;
			case 'G' : moc.his_player_types[i]=16; break;
			case 'H' : moc.his_player_types[i]=17; break;
			case 'I' : moc.his_player_types[i]=18; break;
			case 'J' : moc.his_player_types[i]=19; break;
			case 'K' : moc.his_player_types[i]=20; break;
			case 'L' : moc.his_player_types[i]=21; break;
			case 'M' : moc.his_player_types[i]=22; break;
			case 'N' : moc.his_player_types[i]=23; break;
			case 'O' : moc.his_player_types[i]=24; break;
			//TG08: end
			case '_' : moc.his_player_types[i]= -1; break;
			default: return false;
			}
			next++;
		}
		if ( ! strskip(next,'g',next) )
			return false;

		if ( strskip(next,'_', str ) ) {
			next= str;
			moc.his_goalie_number_upd= false;
		}
		else
		{
			moc.his_goalie_number_upd= true;
			if ( ! str2val(next,moc.his_goalie_number, next) )
				return false;
			//TG08: begin
			//TODO: TG: His goalie is always of the default player type!
			//JTS10: re-enable heterogenous goalie detection as it was introduced
			//       by ss protocol 14
			//if ( moc.his_player_types[ moc.his_goalie_number ] != 0 )
			//  moc.his_player_types[ moc.his_goalie_number ] = 0;
			//TG08: end
		}
		moc.his_player_types_upd= true;
	}
	else
		if ( strskip(next, "doa", next1) )
		{
			next = next1;
			for (int i=0; i<NUM_PLAYERS; i++)
			{
				switch (*next)
				{
				case '_' : moc.direct_opponent_assignment[i]= -1; break;
				case '1' : moc.direct_opponent_assignment[i]= 1; break;
				case '2' : moc.direct_opponent_assignment[i]= 2; break;
				case '3' : moc.direct_opponent_assignment[i]= 3; break;
				case '4' : moc.direct_opponent_assignment[i]= 4; break;
				case '5' : moc.direct_opponent_assignment[i]= 5; break;
				case '6' : moc.direct_opponent_assignment[i]= 6; break;
				case '7' : moc.direct_opponent_assignment[i]= 7; break;
				case '8' : moc.direct_opponent_assignment[i]= 8; break;
				case '9' : moc.direct_opponent_assignment[i]= 9; break;
				case 'A' : moc.direct_opponent_assignment[i]= 10; break;
				case 'B' : moc.direct_opponent_assignment[i]= 11; break;
				case TEAM_IDENTIFIER_ATHUMBOLDT:
                                case TEAM_IDENTIFIER_HERMES:
				case TEAM_IDENTIFIER_WRIGHTEAGLE:
                                case TEAM_IDENTIFIER_OXSY:
				case TEAM_IDENTIFIER_HELIOS:
                                case TEAM_IDENTIFIER_GLIDERS:
                                case TEAM_IDENTIFIER_CYRUS:
				{
					moc.direct_opponent_assignment[i]= -1;
					WSinfo::set_current_opponent_identifier( *next );
					break;
				}
				default:  return false;
				}
				next++;
				// clear all conflict information
				moc.direct_opponent_conflict[i] = -1;
			}
			// next read all stamina_information
			for (int i=0; i<NUM_PLAYERS; i++)
			{
				moc.stamin_capacity_info[i] = *next;
				next++;
			}
			while ( strskip(next,'(',next) )
			{ // read all conflict information
				for (int j = 0; j < 2; j++)
				{
					switch (*next)
					{
					case '_' : intBuf[j] = -1; break;
					case '0' : intBuf[j] = 0 ; break;
					case '1' : intBuf[j] = 1 ; break;
					case '2' : intBuf[j] = 2 ; break;
					case '3' : intBuf[j] = 3 ; break;
					case '4' : intBuf[j] = 4 ; break;
					case '5' : intBuf[j] = 5 ; break;
					case '6' : intBuf[j] = 6 ; break;
					case '7' : intBuf[j] = 7 ; break;
					case '8' : intBuf[j] = 8 ; break;
					case '9' : intBuf[j] = 9 ; break;
					case 'A' : intBuf[j] = 10; break;
					default : return false;
					}
					next++;
				}
				if ( ! strskip(next,')',next) ) return false;
				// add conflict information to coach message
				moc.direct_opponent_conflict[intBuf[0]] = intBuf[1];
			}
			moc.his_player_types_upd= false;
			moc.his_goalie_number_upd= false;
			moc.direct_opponent_assignment_upd = true;
		}
		else return false;
	return true;
}


bool SensorParser::manual_parse_my_trainercomm(const char * str) {
	int tmp;
	const char * next;
	next= str;
	//if ( ! str2val(next,moc.time,next) )
	//  return false;
	//str= next;
	if ( ! strskip(next,'"',next) )
		return false;
	//if ( ! strskip(next,"r",next) ) hack
	if ( ! ( next[9] == 'r' ) )
		return false;
	return true;

	if ( ! str2val(next,tmp,next) )
		return false;
	if ( ! strskip(next,"(true)",next) )
		return false;
	if ( ! strskip(next,'"',next) )
		return false;
	if ( ! strskip(next,"pt",next) )
		return false;

}


bool SensorParser::manual_parse_teamcomm(const char * str, Msg_teamcomm & tc, const char *& next) {
	const char * beg= str;
	tc.reset();
	const double SCALE= ENCODE_SCALE; //shortcut
	int idum;

	if ( str[0] != '*' || str[1] != '*' || str[2] != '*')
		return false;
	str+= 3;
	if ( !WMTools::a64_to_uint6(str,tc.side) )
		return false;

	if ( tc.side != WM::my_side )
		return false;

	str++;
	if ( !WMTools::a3x64_to_int18(str,tc.time) )
		return false;
	str+= 3;
	if ( !WMTools::a3x64_to_int18(str,tc.time_cycle) )
		return false;
	str+= 3;
	if ( !WMTools::a64_to_uint6(str,tc.from) )
		return false;
	str++;

	//get his goalie number (if available)
	if ( !WMTools::a64_to_uint6(str,idum) )
		return false;
	str++;
	tc.his_goalie_number_upd= idum;
	if ( tc.his_goalie_number_upd ) {
		if ( !WMTools::a64_to_uint6(str,tc.his_goalie_number) ) {
			ERROR_OUT << ID << "SensorParser::manual_parse_teamcomm: wrong range for his_goalie_number (" << tc.his_goalie_number << ")";
			return false;
		}
		str++;
	}

	//get ball information (if available)
	if ( !WMTools::a64_to_uint6(str,idum) )
		return false;
	str++;
	tc.ball_upd= idum;

	if ( tc.ball_upd) {
		if ( !WMTools::a64_to_uint6(str,tc.ball.how_old) )
			return false;
		str++;
		if ( !WMTools::a3x64_to_int18(str, idum) )
			return false;
		str+= 3;
		tc.ball.x= double(idum)/SCALE;
		if ( !WMTools::a3x64_to_int18(str, idum) )
			return false;
		str+= 3;
		tc.ball.y= double(idum)/SCALE;
		if ( !WMTools::a3x64_to_int18(str, idum) )
			return false;
		str+= 3;
		tc.ball.vel_x= double(idum)/SCALE;
		if ( !WMTools::a3x64_to_int18(str, idum) )
			return false;
		str+= 3;
		tc.ball.vel_y= double(idum)/SCALE;
	}

	if ( !WMTools::a64_to_uint6(str,tc.players_num) )
		return false;
	str++;
	if ( tc.players_num < 0 || tc.players_num > tc.players_MAX) {
		ERROR_OUT << ID << "\nto many _tc_player entries";
		return false;
	}

	for (int i=0; i< tc.players_num; i++) {
		if ( !WMTools::a64_to_uint6(str,tc.players[i].how_old) )
			return false;
		str++;
		if ( !WMTools::a64_to_uint6(str,tc.players[i].team) )
			return false;
		str++;
		if ( !WMTools::a64_to_uint6(str,tc.players[i].number) )
			return false;
		str++;
		if ( !WMTools::a3x64_to_int18(str, idum) )
			return false;
		str+= 3;
		tc.players[i].x= double(idum)/SCALE;
		if ( !WMTools::a3x64_to_int18(str, idum) )
			return false;
		str+= 3;
		tc.players[i].y= double(idum)/SCALE;
	}

	int check_sum= 0;
	while ( beg < str) {
		check_sum += *beg;
		beg++;
	}
	check_sum= abs(check_sum) % 63;
	int saved_check_sum;
	if ( !WMTools::a64_to_uint6(str,saved_check_sum) )
		return false;
	str++;

	if ( saved_check_sum != check_sum ) {
		ERROR_OUT << ID << "\nwrong checksum";
		return false;
	}
	if ( str[0] != '*' ) {
		ERROR_OUT << ID << "\nwrong end character";
		return false;
	}
	str++;
	next= str;
	return true;
}

bool SensorParser::manual_parse_teamcomm(const char * str, Msg_teamcomm2 & tc, const char *& next) {
	next= str;
	tc.reset();
	BitFIFO bfifo;
	int dum;

	for (int i=0; i<10; i++) { //the message must be 10 bytes long, last char is the checksum!!!
		if ( ! WMTools::a64_to_uint6(next, dum) )
		{
			return false;
		}

		bfifo.put(6,dum);
		next++;
	}
	unsigned int tmp;
	//cout << "\nrecv= "; bfifo.show(cout);

	//here in bfifo we have the right information, and the checksum of the information was right

	while ( bfifo.get_size() >= 5 ) {
		unsigned int obj_id;

		//cout << "\ndecoding ";bfifo.show(cout);
		if ( ! bfifo.get(5,obj_id) )
		{
			return false;
		}
		//cout << "\nnew obj_id= " << obj_id;

		if (obj_id == invalid_id )
			return true;

		if (obj_id > max_id)
		{
			return false;
		}

		if ( obj_id == msg_id ) {
			if ( ! bfifo.get(16,tmp) )
			{
				return false;
			}
			tc.msg.param1= (short)tmp;

			if ( ! bfifo.get(16,tmp) )
			{
				return false;
			}
			tc.msg.param2= (short)tmp;

			if ( ! bfifo.get(8,tmp) ) //hauke
					{
				return false;           //hauke
					}
			tc.msg.type= (unsigned char)tmp;  //hauke

			tc.msg.valid= true;
			tc.msg.from= tc.from;
			continue;
		}

		//TGdoa: begin
		if ( obj_id >= direct_opponent_assignment_0_id && obj_id <= direct_opponent_assignment_2_id )
		{
			tc.direct_opponent_assignment_info.valid = true;
			switch (obj_id)
			{
			case direct_opponent_assignment_0_id:
			{ tc.direct_opponent_assignment_info.assignment = 0; break; }
			case direct_opponent_assignment_1_id:
			{ tc.direct_opponent_assignment_info.assignment = 1; break; }
			case direct_opponent_assignment_2_id:
			{ tc.direct_opponent_assignment_info.assignment = 2; break; }
			//case direct_opponent_assignment_3_id:  //TG08: Note: We had to reduce by one id.
			//{ tc.direct_opponent_assignment_info.assignment = 3; break; }
			}
			continue;
		}
		//TGdoa: end

		//TGpr: begin
		if ( obj_id == pass_request_id )
		{
			if ( ! bfifo.get(4,tmp) )
			{
				return false;
			}
			tc.pass_request_info.pass_in_n_steps = (unsigned char)tmp;
			if ( ! bfifo.get(4,tmp) )
			{
				return false;
			}
			tc.pass_request_info.pass_param = (unsigned char)tmp;
			tc.pass_request_info.valid = true;
			continue;
		}
		//TGpr: end

		//it must be an object, so read the positions of it
		if ( ! bfifo.get(13,tmp) )
		{
			return false;
		}

		Vector pos;
		if ( ! range13bit_to_pos(tmp, pos) )
		{
			return false;
		}

		if ( obj_id == ball_id ) {
			tc.ball.valid= true;
			tc.ball.pos= pos;
			continue;
		}

		if ( obj_id == pass_info_id ) {
			tc.pass_info.ball_pos= pos;

			if ( ! bfifo.get(12,tmp) )
			{
				return false;
			}

			if ( ! range12bit_to_vel(tmp, tc.pass_info.ball_vel) )
			{
				return false;
			}

			if ( ! bfifo.get(7,tmp) )
			{
				return false;
			}

			tc.pass_info.time= tmp;

			tc.pass_info.valid= true;
			continue;
		}

		if ( obj_id == ball_info_id ) {
			tc.ball_info.ball_pos= pos;

			if ( ! bfifo.get(12,tmp) )
			{
				return false;
			}

			if ( ! range12bit_to_vel(tmp, tc.ball_info.ball_vel) )
			{
				return false;
			}

			if ( ! bfifo.get(3,tmp) )
			{
				return false;
			}

			tc.ball_info.age_pos= tmp;

			if ( ! bfifo.get(3,tmp) )
			{
				return false;
			}

			tc.ball_info.age_vel= tmp;
			tc.ball_info.valid= true;
			continue;
		}

		if ( obj_id == ball_holder_info_id ) {
			tc.ball_holder_info.valid= true;
			tc.ball_holder_info.pos= pos;
			continue;
		}

		Msg_teamcomm2::_player & p= tc.players[tc.players_num];
		p.pos= pos;
		//p.number= (obj_id - 2) % 11 + 1; //TG08
		p.number= (obj_id - (1+players_to_their_numbers_offset) ) % 11 + 1;
		//if ( obj_id <= 12 ) //TG08
		if ( obj_id <= (11+players_to_their_numbers_offset) )
			p.team= my_TEAM;
		else
			p.team= his_TEAM;
		tc.players_num++;
	}
	return true;
}

bool SensorParser::manual_encode_teamcomm(char * str, const Msg_teamcomm & tc, char *& end) {
	const char * beg= str;
	const double SCALE= ENCODE_SCALE; //shortcut

	str[0]= '*';   str[1]= '*';   str[2]= '*';
	str+= 3;
	WMTools::uint6_to_a64(tc.side,str);
	str++;
	WMTools::int18_to_a3x64(tc.time,str);
	str+= 3;
	WMTools::int18_to_a3x64(tc.time_cycle,str);
	str+= 3;
	WMTools::uint6_to_a64(tc.from,str);
	str++;

	if ( tc.his_goalie_number_upd) {
		WMTools::uint6_to_a64(1,str);
		str++;
		WMTools::uint6_to_a64(tc.his_goalie_number,str);
		str++;
	}
	else {
		WMTools::uint6_to_a64(0,str);
		str++;
	}

	if ( tc.ball_upd) {
		WMTools::uint6_to_a64(1,str);
		str++;
		WMTools::uint6_to_a64(tc.ball.how_old,str);
		str++;
		WMTools::int18_to_a3x64( int( rint(SCALE * tc.ball.x) ),str);
		str+= 3;
		WMTools::int18_to_a3x64( int( rint(SCALE * tc.ball.y) ),str);
		str+= 3;
		WMTools::int18_to_a3x64( int( rint(SCALE * tc.ball.vel_x) ),str);
		str+= 3;
		WMTools::int18_to_a3x64( int( rint(SCALE * tc.ball.vel_y) ),str);
		str+= 3;
	}
	else {
		WMTools::uint6_to_a64(0,str);
		str++;
	}

	WMTools::uint6_to_a64(tc.players_num,str);
	str++;
	for (int i=0; i< tc.players_num; i++) {
		WMTools::uint6_to_a64(tc.players[i].how_old,str);
		str++;
		WMTools::uint6_to_a64(tc.players[i].team,str);
		str++;
		WMTools::uint6_to_a64(tc.players[i].number,str);
		str++;
		WMTools::int18_to_a3x64( int( rint(SCALE * tc.players[i].x) ),str);
		str+= 3;
		WMTools::int18_to_a3x64( int( rint(SCALE * tc.players[i].y) ),str);
		str+= 3;
	}

	int check_sum= 0;
	while ( beg < str) {
		check_sum += *beg;
		beg++;
	}
	check_sum= abs(check_sum) % 63;
	WMTools::uint6_to_a64(check_sum,str);
	str++;
	str[0]= '*';
	str++;
	end= str;
	return true;
}


bool SensorParser::manual_encode_teamcomm(char *str, const Msg_teamcomm2 &tc, char *&next)
{
	next = str;

	//we can encode 6 bits / character
	//in version 9 we have 10 characters ---> 60 bits of information !!!

	int bits_num  = max_bit_size;
	int bits_left = max_bit_size;

	//const double SCALE= ENCODE_SCALE; //shortcut
	unsigned int tmp;
	unsigned int tmp2;

	BitFIFO bitfifo;

	if ( tc.msg.valid )
	{
		bitfifo.put(  5, msg_id);

		tmp = tc.msg.param1;
		bitfifo.put( 16, tmp);

		tmp = tc.msg.param2;
		bitfifo.put( 16, tmp);

		tmp = tc.msg.type;     //hauke
		bitfifo.put(  8, tmp); //hauke

		bits_left -= 45;       //hauke -37
	}

	if( tc.pass_info.valid )
	{
		if( !pos_to_range13bit( tc.pass_info.ball_pos, tmp ) || !vel_to_range12bit( tc.pass_info.ball_vel, tmp2 ) )
			return false;

		if( bits_left < pass_info_bit_size )
		{
			WARNING_OUT << ID << "\nto much info for " << bits_num << " bits, skipping pass_info";
		}
		else
		{
			bitfifo.put(  5, pass_info_id );
			bitfifo.put( 13, tmp          );
			bitfifo.put( 12, tmp2         );

			if( tc.pass_info.time < 0 || tc.pass_info.time > 127 )
				return false;

			tmp2 = tc.pass_info.time;
			bitfifo.put(  7, tmp2 );

			bits_left -= pass_info_bit_size; //later on + 2 bits for the time
		}
	}

	bool ball_encoded = false;
	if( tc.ball_info.valid && bits_left >= ball_info_bit_size )
	{
		if( !pos_to_range13bit( tc.ball_info.ball_pos, tmp ) || !vel_to_range12bit( tc.ball_info.ball_vel, tmp2 ) )
			return false;

		if( bits_left < ball_info_bit_size )
		{
			WARNING_OUT << ID << "\nto much info for " << bits_num << " bits, skipping ball_info";
		}
		else
		{
			bitfifo.put(  5, ball_info_id );
			bitfifo.put( 13, tmp          );
			bitfifo.put( 12, tmp2         );

			if ( tc.ball_info.age_pos < 0 || tc.ball_info.age_pos > 7 )
				return false;

			tmp2 = tc.ball_info.age_pos;
			bitfifo.put(  3,  tmp2 );

			if( tc.ball_info.age_vel < 0 || tc.ball_info.age_vel > 7 )
				return false;

			tmp2 = tc.ball_info.age_vel;
			bitfifo.put(  3,  tmp2);

			bits_left -= ball_info_bit_size; //later on + 2 bits for the time
			ball_encoded = true;
		}
	}

	if ( !ball_encoded && tc.ball.valid )
	{
		if( !pos_to_range13bit( tc.ball.pos, tmp ) )
			return false;

		if( bits_left < object_bit_size )
		{
			WARNING_OUT << ID << "\nto much info for " << bits_num << " bits, skipping ball object";
		}
		else
		{
			bitfifo.put(  5, ball_id );
			bitfifo.put( 13, tmp     );

			bits_left -= object_bit_size;
            LOG_POL( 0, "___Msg_teamcomm2 send dump = !!!!!\"" << tc << "\"!!!!!" );
            if(tc.ball.valid){
            LOG_POL( 0, _2D << VL2D(WSinfo::me->pos, tc.ball.pos, "#FF8888" ) );
            LOG_POL( 0, _2D << VC2D( tc.ball.pos, 1.2, "#FF8888" ) );
            LOG_POL( 0, _2D << VC2D( tc.ball.pos, 2.2, "#FF8888" ) );
            LOG_POL( 0, _2D << VC2D( tc.ball.pos, 3.2, "#FF8888" ) );
            }
		}
	}

	if( tc.ball_holder_info.valid )
	{
		if( !pos_to_range13bit( tc.ball_holder_info.pos, tmp ) )
			return false;

		if( bits_left < ball_holder_info_bit_size )
		{
			WARNING_OUT << ID << "\nto much info for " << bits_num << " bits_left= " << bits_left;
		}
		else
		{
			bitfifo.put(  5, ball_holder_info_id );
			bitfifo.put( 13, tmp                 );

			bits_left -= ball_holder_info_bit_size;
		}
	}

	//TGpr: begin
	if( tc.pass_request_info.valid )
	{
		if( bits_left < pass_request_bit_size )
		{
			WARNING_OUT << ID << "\nto much info for direct opponent assignment (13bits), skipping.";
		}
		else
		{
			bool doEncodingOfPassRequest = true;
			if( tc.pass_request_info.pass_in_n_steps < 0 || tc.pass_request_info.pass_in_n_steps > 15 )
			{
				WARNING_OUT << ID << "\ncannot encode a pass request with pass_in_n_steps="
						<< tc.pass_request_info.pass_in_n_steps << " (value must be from 0..15) ";
				doEncodingOfPassRequest = false;
			}
			if( tc.pass_request_info.pass_param < 0 || tc.pass_request_info.pass_param > 15 )
			{
				WARNING_OUT << ID << "\ncannot encode a pass request with pass_param="
						<< tc.pass_request_info.pass_param << " (only 4bits are available)";
				doEncodingOfPassRequest = false;
			}

			//start encoding

			if( doEncodingOfPassRequest )
			{
				bitfifo.put( 5, pass_request_id                      );
				bitfifo.put( 4, tc.pass_request_info.pass_in_n_steps );
				bitfifo.put( 4, tc.pass_request_info.pass_param      );

				bits_left -= pass_request_bit_size;
			}
		}
	}
	//TGpr: end

	for( int i = 0; i < tc.players_num; i++ )
	{
		if( bits_left < object_bit_size )
		{
			WARNING_OUT << ID << "\nto much info for " << bits_num << " bits, skipping " << tc.players_num - i << " players";
			break;
		}

		const Msg_teamcomm2::_player &p = tc.players[ i ];

		if( !pos_to_range13bit( p.pos, tmp ) )
		{
			ERROR_OUT << ID << "\n ! pos_to_range13bit pos= " << p.pos;
			return false;
		}

		int number = p.number;
		if( p.team != my_TEAM )
			number += 11;

		//bitfifo.put(5, number+1); //TG08
		bitfifo.put(  5, number + players_to_their_numbers_offset );
		bitfifo.put( 13, tmp                                      );

		bits_left -= object_bit_size;
	}

	//TGdoa: begin
	if( tc.direct_opponent_assignment_info.valid )
	{
		if( bits_left < direct_opponent_assignment_bit_size )
		{
			WARNING_OUT << ID << "\nto much info for direct opponent assignment (5bits), skipping.";
		}
		else
		{
			bool doaEncoded = false;

			switch( tc.direct_opponent_assignment_info.assignment )
			{
			case 0: { bitfifo.put( 5, direct_opponent_assignment_0_id ); doaEncoded = true; break; }
			case 1: { bitfifo.put( 5, direct_opponent_assignment_1_id ); doaEncoded = true; break; }
			case 2: { bitfifo.put( 5, direct_opponent_assignment_2_id ); doaEncoded = true; break; }
			//case 3: { bitfifo.put( 5, direct_opponent_assignment_3_id ); break; }
			}

			if( doaEncoded )
			{
				bits_left -= direct_opponent_assignment_bit_size;
			}
		}
	}
	//TGdoa: end

	if( bits_left == bits_num ) //nothing to be encoded
	{
		//hauke:
		WARNING_OUT << ID << " no info in teamcomm" << tc;
		*next = '\0';
		return true;
	}

	if( bits_left > 0 )
	{
		//cout << "\nfilling up with " << bits_left << " bits";
		if( !bitfifo.fill_with_zeros( bits_left ) )
			return false;
	}

	//cout << "\ncode= ";bitfifo.show(cout);

	//now read the bit sequence in 6 bit chunks, each chunk is then encoded by one character
	for( int i = 0; i < 10; i++ )
	{
		bitfifo.get( 6, tmp );
		if( !WMTools::uint6_to_a64( tmp, next ) )
		{
			return false;
		}
		next++;
	}
	return true;
}

#include "default_server_param.h"


bool produce_server_param_parser(char const* param_str) {
	using std::cout;
	using std::endl;
	/* generates a parser for the server_param string */
	int max_size=200;
	int size= 0;
	char ** tab= new char*[max_size];
	char const* str= 0;

	if ( strskip(param_str,"server_param")) {
		str= DEFAULT_MESSAGE_SERVER_PARAM;
		strskip(str,"(server_param ",str);
	}
	else if ( strskip(param_str,"player_param")) {
		str= DEFAULT_MESSAGE_PLAYER_PARAM;
		strskip(str,"(player_param ",str);
	}
	else {

		cout << param_str << endl;
		ERROR_OUT << ID << " do not recognize " << param_str;
		return false;
	}
	char const* dum;
	char const* dum2;

	bool res= true;
	while (true) {
		res= strskip(str,'(',str)  && strfind(str,' ',dum) && strfind(dum,')',dum2);
		if ( ! res ) {
			res= strskip(str,')',str);
			break;
		}
		tab[size]= new char[dum-str+1];
		strncpy(tab[size],str,dum-str);
		tab[size][dum-str]= '\0';
#if 0
		cout << "\n" << size << "  [";
		while ( str < dum )
			cout << *str++;
		cout << "]";
		cout << " : " << tab[size];
#endif
		size++;
		str= dum2+1;
		//if ( size > 1) break;
	}
	if ( ! res ) {
		ERROR_OUT << ID << "something wrong with the " << param_str << " string";
		return false;
	}

	/* sort the server param decreasingly (very important, increasing order will
     not work because of strskip(...) semantics 
	 */
	for (int i=0; i<size; i++)
		for (int j=i+1; j<size; j++)
			if ( strcmp(tab[i],tab[j]) < 0 ) {
				char * dum= tab[i];
				tab[i]= tab[j];
				tab[j]= dum;
			}


	//for (int i=0; i<size; i++) { cout << "\n" << setw(3) << i << " : [" << tab[i] << "] , "; }

	//output
	const char SPACE0[]="\n";
	const char SPACE1[]="\n  ";
	const char SPACE2[]="\n    ";
	const char SPACE3[]="\n      ";

	cout << SPACE0 << "/* automatically generated from a server_param message using"
			<< SPACE0 << "   bool produce_server_param_parser()"
			<< SPACE0 << "   in sensorparser.c"
			<< SPACE0 << "                  DO NOT EDIT THIS FILE !!!"
			<< SPACE0 << "*/"
			<< SPACE0 << "#include \"sensorparser.h\""
			<< SPACE0 << "#include \"str2val.h\""
			<< SPACE0 << "#include \"macro_msg.h\""
			<< SPACE0
			<< SPACE0 << "bool SensorParser::manual_parse_" << param_str << "(const char * str, Msg_" << param_str << " & param) {"
			<< SPACE1 << "const char * origin= str;"
			<< SPACE1
			<< SPACE1 << "bool res;"
			<< SPACE1
			<< SPACE1 << "char const* dum;"
			<< SPACE1 << "res= strskip(str,\"(" << param_str << "\",str);";

	cout << SPACE1 << "while (res) {"
			<< SPACE2  << "res= strskip(str,'(',str);"
			<< SPACE2  << "if ( ! res ) {"
			<< SPACE2  << "  res= strskip(str,')',str);"
			<< SPACE2  << "  break;"
			<< SPACE2  << "}"
			<< SPACE2  << "bool unknown_option= false;"
			<< SPACE2  << "switch( *str ) {";

	//for (int i=0; i<size; i+= 20) {
	for (int i=0; i<size; i++) {
		bool new_block= i==0 || tab[i][0] != tab[i-1][0];
		if (new_block) {
			if ( i != 0 )
				cout << SPACE3 << "unknown_option= true;"
				<< SPACE3 << "break;";
			cout << SPACE3 << "// ----------------------------------------------------------------"
					<< SPACE2  << "case '" << tab[i][0] << "': ";
		}
		if (new_block)
			cout << SPACE3 << "if ";
		else
			cout << SPACE3 << "else if ";

		cout << "( strskip(str,\"" << tab[i] << "\",dum) )  {"
				<< SPACE3 << "  str= dum;"
				<< SPACE3 << "  res= str2val(str,param."<< tab[i] << ",str) && strskip(str,')',str);"
				<< SPACE3 << "  break;"
				<< SPACE3 << "}";
	}
	cout << SPACE2  << "default: "
			<< SPACE2  << "  unknown_option= true;"
			<< SPACE2  << "}"
			<< SPACE2  << "if ( unknown_option ) {"
			<< SPACE2  << "  WARNING_OUT << \"\\nunkown server option [\";"
			<< SPACE2  << "    while ( *str != '\\0' && *str != ')' )"
			<< SPACE2  << "      WARNING_STREAM << *str++;"
			<< SPACE2  << "    WARNING_STREAM << \"]\";"
			<< SPACE2  << "    if ( *str == ')' )"
			<< SPACE2  << "      str++;"
			<< SPACE2  << "    else"
			<< SPACE2  << "      res= false;"
			<< SPACE2  << "}"
			<< SPACE1  << "} //while";

	cout << SPACE1
			<< SPACE1 << "if (!res) {"
			<< SPACE2 << "ERROR_OUT << \"\\nparse error:\\n\";"
			<< SPACE2 << "show_parser_error_point(ERROR_STREAM,origin,str);"
			<< SPACE1 << "}"
			<< SPACE1 << "return res;"
			<< SPACE0 << "}"
			<< std::endl;

	return true;
}

/**
   parse_error_point is allowed to be == 0, in that case just the message 'origin' is shown
 */
void SensorParser::show_parser_error_point(std::ostream & out, const char * origin, const char * parse_error_point) {
	if (parse_error_point)
		for (  ; origin<parse_error_point && *origin!= '\0';origin++)
			out << *origin;
	else {
		out << origin
				<< "\n[no parse error point provided]";
	}

	if (origin != parse_error_point) {
		out << "\n[something wrong with parse error point]";
		return;
	}
	out << "   <***parse error***>   ";
	int i=0;
	while (i<25 && *origin != '\0') {
		out << *origin;
		origin++;
		i++;
	}
	if (*origin != '\0')
		out << " .......";
}
