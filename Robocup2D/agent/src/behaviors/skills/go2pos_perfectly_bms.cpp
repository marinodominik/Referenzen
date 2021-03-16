#include "go2pos_perfectly_bms.h"

bool Go2PosPerfectly::initialized = false;
const double Go2PosPerfectly::start_nearing_at = 2.;
const double Go2PosPerfectly::start_nearing_tolerance = 0.6;
const double Go2PosPerfectly::min_nearing_angle = 0.5;
const double Go2PosPerfectly::min_nearing_dist = 0.6;
const double Go2PosPerfectly::speed_dash = 15;

Go2PosPerfectly::Go2PosPerfectly()
{
	ivpNeuroGo2Pos = new NeuroGo2Pos();
	ivpFaceBall = new FaceBall();
	ivpSearchBall = new SearchBall();
	ivpBasicCmd = new BasicCmd();
	init_state = 0;
	done_pos = false;
}

bool 
Go2PosPerfectly::get_cmd(Cmd & cmd)
{
	Vector desired_pos = WSinfo::ball->pos;
	bool we_are_near = (WSinfo::me->pos - WSinfo::ball->pos).norm() < 
	    Go2PosPerfectly::start_nearing_at + Go2PosPerfectly::start_nearing_tolerance;
	bool angle_is_good = Tools::my_angle_to(WSinfo::ball->pos).get_value() < 
		Go2PosPerfectly::min_nearing_angle;
	bool we_are_there = ((WSinfo::me->pos +WSinfo::me->vel) - WSinfo::ball->pos).norm() <
			   Go2PosPerfectly::min_nearing_dist;
		
	if(!WSinfo::is_ball_pos_valid())
	{ // if we don't know where the ball is start searching for it
		LOG_POL(1, << "STARTING BALL SEARCH");
		if(!ivpSearchBall->is_searching())
			ivpSearchBall->start_search();
		return ivpSearchBall->get_cmd(cmd);
	}
	// check if done_pos is set 
	if(done_positioning())
		return false;
	if(!we_are_near)
	{ // if this holds we definitley should start from state 0
		done_pos = false;
		init_state = 0;
	}
	//std::cerr << "go2pos_perfectly: init_state " << init_state << " near? " << we_are_near << " time " << WSinfo::ws->time << std::endl;
	if(init_state == 0)
	{
			if(we_are_near)
			{ // we are near to the ball so we redefine our desired pos
			  // to be behind the ball
				LOG_POL(1, << "go2pos_perfectly: GOING TO NEARING POS");
				Vector steigung = WSinfo::ball->pos - HIS_GOAL_CENTER;
				steigung.normalize();
				desired_pos = WSinfo::ball->pos + (Go2PosPerfectly::start_nearing_at * steigung);
    		}
			if((WSinfo::me->pos - desired_pos).norm() > Go2PosPerfectly::start_nearing_tolerance)
			{ // if we still are not at our desired pos call NeuroGo2Pos
				//std::cerr << "calling NEURO " << (WSinfo::me->pos - desired_pos).norm() << std::endl;
				LOG_POL(1, << "go2pos_perfectly: CALLING NEUROGO2");
				ivpNeuroGo2Pos->set_target(desired_pos, Go2PosPerfectly::start_nearing_tolerance);
				return ivpNeuroGo2Pos->get_cmd(cmd);
			}
			else
			{
		     // if we arrive here we are at our nearing pos!
			  // we now need to turn to the ball and start nearing
			  init_state = 1;
			}
	}
	if(init_state == 1)
	{
			if(angle_is_good)
			{
				LOG_POL(1, << "go2pos_perfectly: DONE TURNING");
				init_state = 2;
			}
			else
			{
				
				LOG_POL(1, << "go2pos_perfectly: STARTING TURN TO BALL");
				ivpFaceBall->turn_to_ball(true);
				ivpFaceBall->get_cmd(cmd);
			}
	}
	if(init_state == 2)
	{
			if(we_are_there)
			{	
				LOG_POL(1, << "go2pos_perfectly: DONE GO2POSPERFECTLY");
				done_pos = true;
				return false;
			}
			if(!angle_is_good)
			{
				LOG_POL(1, << "go2pos_perfectly: WRONG IN CASE 2 RESETTING");
				init_state = 1;
				return false;
			}
			else
			{
				LOG_POL(1, << "go2pos_perfectly: STARTING DASHES");
 				ivpBasicCmd->set_dash(Go2PosPerfectly::speed_dash);
				return ivpBasicCmd->get_cmd(cmd);
			}
	}
	// we should never arrive here
	return false;
}

