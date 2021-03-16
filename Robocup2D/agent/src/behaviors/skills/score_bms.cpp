#include "score_bms.h"
#include "ws_info.h"
#include "mdp_info.h"
#include "oneortwo_step_kick_bms.h"


#if 1
#define BASELEVEL 0 // level for logging; should be 3 for quasi non-logging
//#define LOG_DAN(YYY,XXX) LOG_DEB(YYY,XXX)
#define LOG_DAN(YYY,XXX)
#else
#define BASELEVEL 3 // level for logging; should be 3 for quasi non-logging
#define LOG_DAN(YYY,XXX)
#endif


bool Score::initialized = false;

Score::Score() {
  last_time_look_to_goal = -1;
  goalshot_param1 = 0.3;
  goalshot_mode = 0;   //standard goalshot
  nr_of_looks_to_goal = -1;  

  ValueParser vp(CommandLineOptions::policy_conf,"score_bms");
  vp.get("goalshot_param1", goalshot_param1);
  vp.get("goalshot_mode", goalshot_mode);

  oneortwo = new OneOrTwoStepKick();
}

Score::~Score() {
  delete oneortwo;
}

bool Score::get_cmd(Cmd &cmd) {
  /*
  Intention intention;
  double speed;
  Vector target;

  if (test_shoot2goal(intention)) {
    speed = intention.kick_speed;
    target = intention.kick_target;
    mdpInfo::set_my_intention(DECISION_TYPE_SCORE, speed, 0, target.x, target.y,0);
    LOG_POL(BASELEVEL,<<"Score get_cmd: try to score w speed "<<speed<<" to target "<<target);
    neurokick->kick_to_pos_with_initial_vel(speed,target);
    neurokick->get_cmd(cmd);
    return true;
  } else {
    return false;
  }
  */
  return false;
}

bool Score::test_shoot2goal(Intention &intention) {
  int multi_step;
  double velocity, direction;
  Vector target;
  int best_index;

  int result = get_goalshot_chance(multi_step, velocity, direction, target, best_index);

  double target_dir = (mdpInfo::opponent_goalpos()-WSinfo::me->pos).arg();

  risky_goalshot_possible = 0;

  intention.immediatePass = false; //allow multi kicks via neurokick!

  if(goalshot_mode == 0){
    if (result == 4) result = 1;
  }
  else{
    if (result == 4) {
      LOG_ERR(0,"Goalshot risky, continue planning");
    }
  }
  
  if (result == 0) {
    LOG_DAN(BASELEVEL,"no goalshot found");
    //no shot found
    return false;

  } else if (result == 1) {

    //secure shot
    if (multi_step) {
      LOG_DAN(0, << "shoot with neuro_kick2! velocity = " << velocity 
	      << " target.x = " << target.x << " target.y = " << target.y);
      mdpInfo::set_my_neck_intention(NECK_INTENTION_LOOKINDIRECTION, target_dir);//direction);
      //set_aaction_score(aaction, target, velocity);
      intention.set_score(target, velocity, WSinfo::ws->time);
      return true;
      //return Move_Factory::get_1or2_Step_Kick(velocity, target);
    } else {
      LOG_DAN(0, << "shoot with one step kick! velocity = " << velocity 
	         << " dir = " << direction);
      intention.set_score(target, velocity, WSinfo::ws->time);
      return true;
      //      return Move_Factory::get_1Step_Kick(velocity, direction);
    }

  } else if (result == 2) {

    //risky because of goalie_age?
    if ((mdpInfo::opponent_goalpos() - WSinfo::me->pos).norm() > 30.0) {
      LOG_DAN(BASELEVEL,"result 2 and too far away");
      return false;
    }

#if 0 // ridi 22.6.03 : replaced by new access to goalie
    PlayerSet alive_opps = WSinfo::alive_opponents;
    PPlayer goalie = alive_opps.get_player_by_number(WSinfo::ws->his_goalie_number);
#endif
    PPlayer goalie;
    if(WSinfo::his_goalie)
      goalie = WSinfo::his_goalie;
    else
      return false;
    //if (goalie == NULL) return false;
    PlayerSet pset_tmp = WSinfo::valid_opponents;
    pset_tmp.keep_players_in_circle(WSinfo::me->pos, 3.0);
    if (pset_tmp.num > 0) {
      LOG_DAN(0, << "doing a risky shot because opponents are close!");
      if (multi_step) {
	LOG_DAN(0, << "shoot with neuro_kick2! velocity = " << velocity 
		<< " target.x = " << target.x << " target.y = " << target.y);
	mdpInfo::set_my_intention(DECISION_TYPE_SCORE, velocity, direction,
				  target.getX(), target.getY(), best_index);
	mdpInfo::set_my_neck_intention(NECK_INTENTION_LOOKINDIRECTION, target_dir);//direction);
	intention.set_score(target, velocity, WSinfo::ws->time);
	//set_aaction_score(aaction, target, velocity);
	return true;
	//return Move_Factory::get_Neuro_Kick2(5, velocity, target.x, target.y);
      } else {
	LOG_DAN(0, << "shoot with one step kick! velocity = " << velocity 
		<< " dir = " << direction);
	//	return Move_Factory::get_1Step_Kick(velocity, direction);
	intention.set_score(target, velocity, WSinfo::ws->time);
	//set_aaction_score(aaction, target, velocity);
	return true;
      }

    } else {

      if (goalie->age <= 4) {
	LOG_DAN(0, << "opponent goalie too young, don't look to him");
	return false;
      }

      LOG_DAN(0, << "opponent goalie too old, look to goal with hold ball");
      //double target_dir = (mdpInfo::opponent_goalpos()-WSinfo::me->pos).arg();
      
      if (last_time_look_to_goal != WSinfo::ws->time - 1) {
	nr_of_looks_to_goal = 1;
      } else {
	++nr_of_looks_to_goal;
      }
      last_time_look_to_goal = WSinfo::ws->time;

      /*
      if (nr_of_looks_to_goal == 1) {
	LOG_DAN(0, << "look to center of goal");
	target_dir = (mdpInfo::opponent_goalpos()-WSinfo::me->pos).arg();
      }
      else if (nr_of_looks_to_goal == 2) {
	LOG_DAN(0, << "look to left goalcorner");
	target_dir = (ServerOptions::their_left_goal_corner-WSinfo::me->pos).arg();
      }
      else if (nr_of_looks_to_goal == 3) {
	LOG_DAN(0, << "look to right goalcorner");
	target_dir = (ServerOptions::their_right_goal_corner-WSinfo::me->pos).arg();
      }
      else if (nr_of_looks_to_goal == 4) {*/
      if (nr_of_looks_to_goal >= 3) {
	if (multi_step) {
	  LOG_DAN(0, << "I looked to goal, didn't see goalie and I'm close to goal, shoot");
	  LOG_DAN(0, << "shoot with neuro_kick2! velocity = " << velocity 
		  << " target.x = " << target.getX() << " target.y = " << target.getY());
	  mdpInfo::set_my_neck_intention(NECK_INTENTION_LOOKINDIRECTION, direction);
	  mdpInfo::set_my_intention(DECISION_TYPE_SCORE, velocity, direction,
				    target.getX(), target.getY(), best_index);
	  intention.set_score(target, velocity, WSinfo::ws->time);
	  //set_aaction_score(aaction, target, velocity);
	  return true;

	} else {
	  LOG_DAN(0, << "shoot with one step kick! velocity = " << velocity 
		  << " dir = " << direction);
	  //return Move_Factory::get_1Step_Kick(velocity, direction);
	  intention.set_score(target, velocity, WSinfo::ws->time);
	  //set_aaction_score(aaction, target, velocity);
	  return true;
	}
      }
	
      mdpInfo::set_my_neck_intention(NECK_INTENTION_LOOKINDIRECTION, target_dir);
      if(mdpInfo::could_see_in_direction(target_dir) == true) {
	// I can see in direction by turning neck without turning body
	mdpInfo::set_my_neck_intention(NECK_INTENTION_LOOKINDIRECTION, target_dir);
	mdpInfo::set_my_intention(DECISION_TYPE_LOOKFORGOALIE, mdpInfo::opponent_goalpos().getX(),
				  mdpInfo::opponent_goalpos().getY());
	// ridi03:todo
	//	return Move_Factory::get_Hold_Ball(1);	
	
	//LOG_ERR(0,"WBALL03: should hold ball and look; not yet implemented");
	//LOG_DAN(BASELEVEL,"WBALL03: should hold ball and look; not yet implemented");
	intention.set_holdturn(ANGLE(target_dir), WSinfo::ws->time);
	return true;
      } else {
	// have to turn body.
	LOG_DAN(0, " have to turn body, but don't do it");
	last_time_look_to_goal = -1;
	return false;
	/*
	  mdpInfo::set_my_neck_intention(NECK_INTENTION_LOOKINDIRECTION, target_dir);     
	  return Move_Factory::get_Hold_Ball(target_dir,1); 
	*/
      }
    }

  } else if (result == 3) {
    //risky because opponent could intercept

  } else if (result == 4) {
    //risky because goal could be missed

    saved_velocity = velocity;
    saved_direction = direction;
    saved_target = target;
    saved_multi_step = multi_step;
    risky_goalshot_possible = 1;

  } else if ((result == 5) || (result == 6)) {
    //opponent goalie not alive or goalie number not known or other strange problem

    if ((mdpInfo::opponent_goalpos() - WSinfo::me->pos).norm() > 25.0) {
      LOG_DAN(0, " result 5 and too far");
      return false;
    }

    LOG_DAN(0, << "opponent goalie not alive, number not known or other strange problem, look to goal with hold ball");
    double target_dir = (mdpInfo::opponent_goalpos()-WSinfo::me->pos).arg();
    
    if (last_time_look_to_goal != WSinfo::ws->time - 1) {
      nr_of_looks_to_goal = 1;
    } else {
      ++nr_of_looks_to_goal;
    }
    last_time_look_to_goal = WSinfo::ws->time;

    target_dir = (mdpInfo::opponent_goalpos()-WSinfo::me->pos).arg();
    /*
    if (nr_of_looks_to_goal == 1) target_dir = (mdpInfo::opponent_goalpos()-WSinfo::me->pos).arg();
    else if (nr_of_looks_to_goal == 2) target_dir = (ServerOptions::their_left_goal_corner-WSinfo::me->pos).arg();
    else if (nr_of_looks_to_goal == 3) target_dir = (ServerOptions::their_right_goal_corner-WSinfo::me->pos).arg();
    else if (nr_of_looks_to_goal == 4) {*/
    if (nr_of_looks_to_goal >= 3) {
      if (result == 6) {
	Vector goal = Vector(ServerOptions::pitch_length/2.0, 0.0);
	ANGLE angle = (goal-WSinfo::ball->pos).ARG();
	double one_step_vel, multi_step_vel;
	//Move_1or2_Step_Kick::get_vel_in_dir(3.0, angle, one_step_vel, multi_step_vel);
	oneortwo->kick_in_dir_with_max_vel(ANGLE(angle));
	oneortwo->get_vel(one_step_vel, multi_step_vel);

	if (one_step_vel > ServerOptions::ball_speed_max)
	  one_step_vel = ServerOptions::ball_speed_max;

	if (multi_step_vel > ServerOptions::ball_speed_max)
	  multi_step_vel = ServerOptions::ball_speed_max;

	LOG_DAN(0, << "I looked to goal, didn't see goalie AND opponents and I'm close to goal, shoot");
	LOG_DAN(0, << "shoot with neuro_kick2! velocity = " << multi_step_vel 
		<< " target.x = " << goal.x << " target.y = " << goal.y);
	mdpInfo::set_my_neck_intention(NECK_INTENTION_LOOKINDIRECTION, 
				       (goal-WSinfo::me->pos).arg());
	//return Move_Factory::get_Neuro_Kick2(5, velocity, target.x, target.y);
	intention.set_score(goal, multi_step_vel, WSinfo::ws->time);
	//set_aaction_score(aaction, target, velocity);
	return true;	
      }
      if (multi_step) {
	LOG_DAN(0, << "I looked to goal, didn't see goalie and I'm close to goal, shoot");
	LOG_DAN(0, << "shoot with neuro_kick2! velocity = " << velocity 
		<< " target.x = " << target.x << " target.y = " << target.y);
	mdpInfo::set_my_neck_intention(NECK_INTENTION_LOOKINDIRECTION, 
				       (target-WSinfo::me->pos).arg());
	//return Move_Factory::get_Neuro_Kick2(5, velocity, target.x, target.y);
	intention.set_score(target, velocity, WSinfo::ws->time);
	//set_aaction_score(aaction, target, velocity);
	return true;

	//return Move_Factory::get_1or2_Step_Kick(velocity, target);
      } else {
	LOG_DAN(0, << "shoot with one step kick! velocity = " << velocity 
		<< " dir = " << direction);
	intention.set_score(target, velocity, WSinfo::ws->time);
	//set_aaction_score(aaction, target, velocity);
	return true;
	//return Move_Factory::get_1Step_Kick(velocity, direction);
      }
    }
    
    mdpInfo::set_my_neck_intention(NECK_INTENTION_LOOKINDIRECTION, target_dir);
    if(mdpInfo::could_see_in_direction(target_dir) == true) {
      // I can see in direction by turning neck without turning body
      mdpInfo::set_my_neck_intention(NECK_INTENTION_LOOKINDIRECTION, target_dir);
      mdpInfo::set_my_intention(DECISION_TYPE_LOOKFORGOALIE, mdpInfo::opponent_goalpos().getX(),
				mdpInfo::opponent_goalpos().getY());

      // ridi03:todo
      //	return Move_Factory::get_Hold_Ball(1);	
      LOG_ERR(0,"WBALL03: should hold ball and look; not yet implemented");
      LOG_DAN(0,"WBALL03: should hold ball and look; not yet implemented");
      intention.set_holdturn(ANGLE(target_dir), WSinfo::ws->time);
      return true;
    } else {
      // have to turn body.
      LOG_DAN(0, " have to turn body, but don't do it");
      last_time_look_to_goal = -1;
      return false;
    }
  }    
  LOG_DAN(0, " shoot 2 goal - unsuccessful result "<<result);    
  return false;
}


int Score::get_goalshot_chance(int &multi_step, double &velocity,
					       double &direction, Vector &target, int &best_index) {

  if (WSinfo::me->pos.getX() < 15.0) return 0;

  if ( (Tools::my_abs_angle_to(Vector(FIELD_BORDER_X, 0.0)).get_value() > DEG2RAD(83)) &&
       (Tools::my_abs_angle_to(Vector(FIELD_BORDER_X, 0.0)).get_value() < DEG2RAD(277)) &&
       fabs(WSinfo::me->pos.getY()) > 8.0) {
    //LOG_DEB(0, << "don't shoot, wrong angle");
    return 0;
  }
  
  Vector test_targets[9];
  double test_dirs[9];
  double test_vels_1step[9];
  double test_vels_multi[9];
  int one_step_kick_possible[9];
  int multi_kick_possible[9];
  int secure_1step_kick_possible[9];
  int secure_multi_kick_possible[9];
  int ball_steps_to_goal[9];
  int ret = 0;

  Vector ball_pos = WSinfo::ball->pos;

#if 0 // ridi 22.6.03 : replaced by new access to goalie
  PlayerSet alive_opps = WSinfo::alive_opponents;
  PPlayer goalie = alive_opps.get_player_by_number(WSinfo::ws->his_goalie_number);
#endif

  PPlayer goalie =WSinfo::his_goalie;

  if (goalie == NULL) {
    ret = 5;
    PlayerSet valid_opps = WSinfo::valid_opponents;
    valid_opps.keep_and_sort_closest_players_to_point(1, mdpInfo::opponent_goalpos());
    if (valid_opps.num <= 0) {
      return 6;
    }
    goalie = valid_opps[0];
  }

  Vector goalie_pos = goalie->pos;
  int goalie_age = goalie->age;
  Vector goalie_vel = goalie->vel;
  int goalie_vel_age = goalie->age_vel;
  
  double goalie_initial_size = goalshot_param1;//0.4


  if (((ball_pos-goalie_pos).norm() > 20.0) &&
      (Tools::get_angle_between_null_2PI(((ServerOptions::their_left_goal_corner-ball_pos).arg() - 
					  (goalie_pos-ball_pos).arg())) < M_PI) &&
      (Tools::get_angle_between_null_2PI(((goalie_pos-ball_pos).arg() - 
					  (ServerOptions::their_right_goal_corner-ball_pos).arg()))<M_PI)) {
    LOG_DAN(0, << "don't consider shooting, goalie is on place and I'm too far away!");
    return 0;
  }

  consider_special_cases(goalie_age, goalie_vel_age, goalie_vel, goalie_initial_size, goalie_pos, goalie);

  fill_target_arrays(test_targets, test_dirs, 9, ball_pos);

  //fill_velocity_arrays(test_vels_1step, test_vels_multi, test_dirs, 9);
  fill_velocity_arrays(test_vels_1step, test_vels_multi, test_targets, 9);

  Vector kick_vel;
  int goalie_intercepts = 0;
  int player_intercepts = 0;
  int best_kick = -1;
  int kick_found = 0;

  double goalie_size = goalie_initial_size;

  //test for multi-step-kicks, use current goalie-pos(even if old)
  for (int i = 0; i < 9; i++) {
    kick_vel.init_polar(test_vels_multi[i], test_dirs[i]);
    goalie_intercepts = intercept_goalie(ball_pos, kick_vel, goalie_pos, goalie_size + 0.3);
    if (goalie_intercepts <= 0) {
      multi_kick_possible[i] = 1;
      kick_found = 1;
    } else {
      multi_kick_possible[i] = 0;
    }
    /*
    LOG_DAN(0, << _2D << L2D( ball_pos.x, ball_pos.y, ball_pos.x + kick_vel.x * 3.0, 
			      ball_pos.y + kick_vel.y * 3.0, "#ff0000" ));
    */
  }

  //no multi-kick found => no single kick is possible, return!
  //if (!kick_found) return 0;
  
  for (int i = 0; i < 9; i++) {
    //only test for 1step-kick if multi-step-kick could be succesful
    /*
    if (!multi_kick_possible[i]) {
      one_step_kick_possible[i] = 0;
      continue;
      }*/

    kick_vel.init_polar(test_vels_1step[i], test_dirs[i]);
    goalie_intercepts = intercept_goalie(ball_pos, kick_vel, goalie_pos, goalie_size);
    ball_steps_to_goal[i] = goalie_intercepts;
    if (goalie_intercepts <= 0) {
      one_step_kick_possible[i] = 1;
      kick_found = 1;
    } else {
      one_step_kick_possible[i] = 0;
    }
  }

  //no kick found => return!
  if (!kick_found) return 0;

  if ((goalie_age <= 4) &&
      (goalie_age >= 1)) {
    double goalie_dir = (goalie->pos - ball_pos).arg();
    int top_of_goalie_shots = 0, bottom_of_goalie_shots = 0;
    for (int i = 0; i < 9; ++i) {
      if (multi_kick_possible[i]) {
	if (Tools::get_angle_between_null_2PI(test_dirs[i]-goalie_dir) < M_PI) {
	  ++top_of_goalie_shots;
	} else {
	  ++bottom_of_goalie_shots;
	}
      }
    }
    //shots on both sides of goalie are possible => goalie can't cover the whole goal, even if he
    //moved, declare the shots with old goalie-pos as secure!
    if ((top_of_goalie_shots > 0) && (bottom_of_goalie_shots > 0)) {
      LOG_DAN(0, << "goalie is old, but shots on both sides of goalie are possible, use old goalie-pos");
      for (int i = 0; i < 9; ++i) {
	secure_multi_kick_possible[i] = multi_kick_possible[i];
	secure_1step_kick_possible[i] = one_step_kick_possible[i];
      }
      //hack: don't consider the other shots below
      goalie_age = 0;
    }
  }

  if (goalie_age <= 4) {
    goalie_size = goalie_initial_size + 0.8 * goalie_age;
  } else {
    //no special case here
    goalie_size = goalie_initial_size + 0.8 * goalie_age;
  }

  LOG_DAN(0, << "goalie_age ist " << goalie_age);
  LOG_DAN(0, << "goalie_size ist " << goalie_size);

  if (goalie_age >= 1) {

    for (int i = 0; i < 9; ++i) {
      kick_vel.init_polar(test_vels_multi[i], test_dirs[i]);
      goalie_intercepts = intercept_goalie(ball_pos, kick_vel, goalie_pos, goalie_size);
      if (goalie_intercepts <= 0) {
	secure_multi_kick_possible[i] = 1;
      } else {
	secure_multi_kick_possible[i] = 0;
      }
    }

    for (int i = 0; i < 9; ++i) {
      if (!secure_multi_kick_possible[i]) {
	secure_1step_kick_possible[i] = 0;
	continue;
      }
      kick_vel.init_polar(test_vels_1step[i], test_dirs[i]);
      goalie_intercepts = intercept_goalie(ball_pos, kick_vel, goalie_pos, goalie_size);
      if (goalie_intercepts <= 0) {
	secure_1step_kick_possible[i] = 1;
      } else {
	secure_1step_kick_possible[i] = 0;
      }
    }

  } else {
    for (int i = 0; i < 9; ++i) {
      secure_multi_kick_possible[i] = multi_kick_possible[i];
      secure_1step_kick_possible[i] = one_step_kick_possible[i];
    }
  }

  //test if opponent could intercept
  for (int i = 0; i < 9; ++i) {
    //if (!multi_kick_possible[i]) continue;
    
    player_intercepts = intercept_opponents(test_dirs[i], test_vels_multi[i], -ball_steps_to_goal[i]);
    if (player_intercepts) {
      LOG_DAN(0, << "opponent number " << player_intercepts 
	      << " intercepts ball before goalline, don't shoot!");
      multi_kick_possible[i] = 0;
      secure_multi_kick_possible[i] = 0;
      one_step_kick_possible[i] = 0;
      secure_1step_kick_possible[i] = 0;
    }
  }


  int one_step_kick_found = 0, multi_kick_found = 0;
  int secure_1step_kick_found = 0, secure_multi_kick_found = 0;


  for (int i = 0; i < 9; ++i) {
    if (secure_multi_kick_possible[i]) ++secure_multi_kick_found;
    if (secure_1step_kick_possible[i]) ++secure_1step_kick_found;
    if (multi_kick_possible[i]) ++multi_kick_found;
    if (one_step_kick_possible[i]) ++one_step_kick_found;
  }

  //select best kick
  if (secure_multi_kick_found != 0) {

    if ((secure_multi_kick_found > secure_1step_kick_found) ||
	(secure_1step_kick_found == 0)) {
      //if ((mdpInfo::opponents_within_range(WSinfo::me->pos, 2.0) == 0) || 
      //(secure_1step_kick_found == 0) ||
      //(secure_multi_kick_found > secure_1step_kick_found)) {

      best_kick = select_best_kick(secure_multi_kick_possible, 9);
      multi_step = 1;
      velocity = test_vels_multi[best_kick];
      direction = test_dirs[best_kick];
      target = test_targets[best_kick];
      if (ret == 0) {
	if (best_kick >= 7) {
	  ret = 4;
	} else {
	  ret = 1;
	}
      }
    } else {
      best_kick = select_best_kick(secure_1step_kick_possible, 9);
      multi_step = 0;
      velocity = test_vels_1step[best_kick];
      direction = test_dirs[best_kick];
      target = test_targets[best_kick];
      if (ret == 0) {
	if (best_kick >= 7) {
	  ret = 4;
	} else {
	  ret = 1;
	}
      }
    }

  } else if (secure_1step_kick_found != 0) {

    best_kick = select_best_kick(secure_1step_kick_possible, 9);
    multi_step = 0;
    velocity = test_vels_1step[best_kick];
    direction = test_dirs[best_kick];
    target = test_targets[best_kick];
    if (ret == 0) {
      if (best_kick >= 7) {
	ret = 4;
      } else {
	ret = 1;
      }
    }

  } else if (multi_kick_found != 0) {

    if ((multi_kick_found-1 > one_step_kick_found) ||
	(one_step_kick_found == 0)) {
      best_kick = select_best_kick(multi_kick_possible, 9);
      multi_step = 1;
      velocity = test_vels_multi[best_kick];
      direction = test_dirs[best_kick];
      target = test_targets[best_kick];
      if (ret == 0) {
	ret = 2;
      }
    } else {
      best_kick = select_best_kick(one_step_kick_possible, 9);
      multi_step = 0;
      velocity = test_vels_1step[best_kick];
      direction = test_dirs[best_kick];
      target = test_targets[best_kick];
      if (ret == 0) {
	ret = 2;
      }
    }

  } else if (one_step_kick_found != 0) {

    best_kick = select_best_kick(one_step_kick_possible, 9);
    multi_step = 0;
    velocity = test_vels_1step[best_kick];
    direction = test_dirs[best_kick];
    target = test_targets[best_kick];
    if (ret == 0) {
      ret = 2;
    }

  }

  kick_vel.init_polar(test_vels_multi[best_kick], test_dirs[best_kick]);
  LOG_DAN(0, << "selected kick nr. " << best_kick << " as best kick");

  LOG_DAN(0, << _2D << L2D( ball_pos.x, ball_pos.y, ball_pos.x + kick_vel.x * 3.0, 
			    ball_pos.y + kick_vel.y * 3.0, "ffffff" ));    
  LOG_DAN(0, << "return value is " << ret);
  best_index = best_kick;
  return ret;
}

void Score::fill_velocity_arrays(double *test_vels_1step, double *test_vels_multi,
						 Vector *test_targets, int nr_of_targets) {
  double one_step_vel, multi_step_vel;

  for (int i = 0; i < nr_of_targets; ++i) {
    //Move_1or2_Step_Kick::get_vel_in_dir(3.0, test_dirs[i], one_step_vel, multi_step_vel);
    //oneortwo->get_vel_in_dir(3.0, test_dirs[i], one_step_vel, multi_step_vel);
    //oneortwo->get_vel_in_dir(3.0, test_dirs[i], one_step_vel, multi_step_vel);

    //oneortwo->kick_in_dir_with_max_vel(ANGLE(test_dirs[i]));
    oneortwo->kick_to_pos_with_max_vel(test_targets[i]);
    oneortwo->get_vel(one_step_vel, multi_step_vel);
    //oneortwo->get_vel(one_step_vel, multi_step_vel);


    //if (one_step_vel > ServerOptions::ball_speed_max * ServerOptions::ball_decay)
    if (one_step_vel > ServerOptions::ball_speed_max)
      one_step_vel = ServerOptions::ball_speed_max;
    //if (multi_step_vel > ServerOptions::ball_speed_max * ServerOptions::ball_decay)
    if (multi_step_vel > ServerOptions::ball_speed_max)
      multi_step_vel = ServerOptions::ball_speed_max;

    if (multi_step_vel < one_step_vel) {
      LOG_DAN(0, << "ERROR!!!! multi_step_vel smaller than one_step_vel!");
    }

    test_vels_1step[i] = one_step_vel;
    test_vels_multi[i] = multi_step_vel;  
    //test_vels_multi[i] = 2.5;
  }

}


void Score::fill_target_arrays(Vector *test_targets, double *test_dirs,
					       int nr_of_targets, Vector ball_pos) {

  Vector upper_corner = Vector(ServerOptions::pitch_length/2.0, ServerOptions::goal_width/2.0);
  Vector lower_corner = Vector(upper_corner.getX(), -upper_corner.getY());


  double ang_diff = Tools::get_angle_between_null_2PI( (upper_corner-WSinfo::ball->pos).arg() -
						      (lower_corner-WSinfo::ball->pos).arg() );

  double safety_threshold = DEG2RAD(3.0);
  LOG_DAN(0, "ang_diff ist " << RAD2DEG(ang_diff));
  if ((ball_pos.getX() > 25.0) && (ang_diff < DEG2RAD(10.0))) {
    safety_threshold = ang_diff / 6.0;
  } else if (ang_diff < DEG2RAD(8.0)) {
    safety_threshold = ang_diff/8.0;
  }

  double delta_ang = Tools::get_angle_between_null_2PI(ang_diff - 2.0*safety_threshold) / 6.0;
  
  double upper_secure_arg = Tools::get_angle_between_null_2PI((upper_corner-WSinfo::ball->pos).arg() -
							     safety_threshold);

  //LOG_DAN(0, << "upper secure arg is " << upper_secure_arg);

  test_dirs[0] = Tools::get_angle_between_null_2PI(upper_secure_arg - delta_ang * 3.0);
  test_dirs[1] = Tools::get_angle_between_null_2PI(upper_secure_arg - delta_ang * 2.0);
  test_dirs[2] = Tools::get_angle_between_null_2PI(upper_secure_arg - delta_ang * 4.0);
  test_dirs[3] = Tools::get_angle_between_null_2PI(upper_secure_arg - delta_ang * 1.0);
  test_dirs[4] = Tools::get_angle_between_null_2PI(upper_secure_arg - delta_ang * 5.0);
  test_dirs[7] = Tools::get_angle_between_null_2PI(upper_secure_arg);
  test_dirs[8] = Tools::get_angle_between_null_2PI(upper_secure_arg - delta_ang * 6.0);
  
  if (delta_ang > DEG2RAD(4.0)) {
    test_dirs[5] = Tools::get_angle_between_null_2PI(upper_secure_arg - DEG2RAD(2.0));
    test_dirs[6] = Tools::get_angle_between_null_2PI(upper_secure_arg - delta_ang * 6.0 + DEG2RAD(2.0));
  } else {
    test_dirs[5] = Tools::get_angle_between_null_2PI(upper_secure_arg - delta_ang/2.0);
    test_dirs[6] = Tools::get_angle_between_null_2PI(upper_secure_arg - delta_ang * 6.0 + delta_ang/2.0);
  }

  Vector st;
  for (int i = 0; i < nr_of_targets; ++i) {
    st.init_polar(1.0, test_dirs[i]);
    st = (1.0/st.getX()) * st;
    //st /= (st.x); DAS IST FALSCH(geht so nicht, siehe Implementuerung von Vector

    test_targets[i] = (ServerOptions::pitch_length/2.0 - WSinfo::ball->pos.getX()) * st + WSinfo::ball->pos;
  }

}


void Score::consider_special_cases(int goalie_age, int goalie_vel_age, Vector goalie_vel,
						   double &goalie_initial_size, Vector &goalie_pos,
						   PPlayer goalie) {
  if (goalie_vel_age == 0) {
    //if (goalie_vel.norm() > 0.1) {
    LOG_DAN(0, << "goalie_pos um " << goalie_vel << "verschoben");
    goalie_pos = goalie_pos + goalie_vel;
    //if (goalie_initial_size >= 0.3) goalie_initial_size -= 0.3;
    //else goalie_initial_size = 0.0;
    //}
  }

  if (fabs(WSinfo::ball->vel.getY()) > 1.0) {
    LOG_DAN(0, << "ball has high y-velocity, assume that goalie moves!");
    goalie_initial_size += 0.4;
  }

#if 0 // ridi goalshot
  if (goalie_age <= 1) {
    if ((goalie_pos - WSinfo::me->pos).norm() < 5.0) {
      LOG_DAN(0, << " goalie too close, reducing size to 0");
      goalie_initial_size = 0.0;
    }
  }
#endif 
}


int Score::select_best_kick(int *kick_possible, int nr_of_targets) {
  int all_possible = 1;
  for (int i = 0; i < nr_of_targets; ++i) {
    if (kick_possible[i] == 0) {
      all_possible = 0;
      break;
    }
  }
  if (all_possible) return 0;
  
  int left_side = 0, right_side = 0;
  for (int i = 1; i < nr_of_targets; ++i) {
    if (kick_possible[i]) {
      if (i%2 == 1) ++left_side;
      else right_side++;
    }
  }

  if (left_side == right_side) {
    if (WSinfo::ws->time % 2 == 0) {
      ++right_side;
    } else {
      ++left_side;
    }
  }

  if (left_side > right_side) {
    if (kick_possible[1] && kick_possible[3]) return 3;
    if (kick_possible[5]) return 5;
    else {
      for (int j = 0; j < nr_of_targets; ++j) {
	if ((kick_possible[j]) && (j%2 == 1)) return j;
      }
    }
  } else if (left_side < right_side) {
    if (kick_possible[2] && kick_possible[4]) return 4;
    if (kick_possible[6]) return 6;
    else {
      for (int j = 0; j < nr_of_targets; ++j) {
	if ((kick_possible[j]) && (j%2 == 0)) return j;
      }
    }
  }

  return -1;
}

double Score::player_action_radius_at_time(int time, PPlayer player,
							  double player_dist_to_ball,
							  int player_handicap) {
  //used for learning, otherwise players try to shoot through static defenders
  return 1.1;

  int time_L = time - player_handicap;

  if (player_dist_to_ball < 3.0) {
    if (time_L <= 2) {
      return 0.0 + player->kick_radius;
    } else {
      return player->speed_max * (time_L-2) * 0.8 + player->kick_radius;
    }
  } else {
    if (time_L <= 1) {
      return 0.0 + player->kick_radius;
    } else {
      return player->speed_max * (time_L-1) * 0.8 + player->kick_radius;
    }
  }
}

int Score::intercept_opponents(double direction, double b_v,
					       int max_steps) {
  PlayerSet pset = WSinfo::valid_opponents;
  pset.keep_players_in_cone(WSinfo::ball->pos, ANGLE(direction-DEG2RAD(20)), 
			    ANGLE(direction+DEG2RAD(20)));
  Vector player_pos, b_pos, b_vel, ball_vel;
  double player_dist_to_ball, player_action_radius;
  ball_vel.init_polar(b_v, direction);


  for (int i = 0; i < pset.num; ++i) {
    if (pset[i]->number == WSinfo::ws->his_goalie_number) continue;
    player_pos = pset[i]->pos;
    b_pos = WSinfo::ball->pos;
    b_vel = ball_vel;
    player_dist_to_ball = (pset[i]->pos - WSinfo::ball->pos).norm();

    for (int j = 1; j < 12; ++j) {
      b_pos += b_vel;
      b_vel *= ServerOptions::ball_decay;
      player_action_radius = player_action_radius_at_time(j, pset[i], player_dist_to_ball, 0);//0.4
      
      if ((b_pos - player_pos).norm() < player_action_radius) {
	if (pset[i]->number > 0) return pset[i]->number;
	else return 1;
      }
    }
  }
  return 0;
}

double Score::goalie_action_radius_at_time(int time, double goalie_size, int goalie_handicap) {
  int time_L = time - goalie_handicap;

  if (time_L < 0) return 0.0;
  switch (time_L) {
  case 0: 
    return 0.0;
  case 1:
    return goalie_size;
  case 2:
    return 0.6 + goalie_size;
  case 3:
    return 1.4 + goalie_size;
  case 4:
    return 2.4 + goalie_size;
  default:
    if (time_L < 0) return 0.0;
    else return 2.4 + 1.0 * (time_L - 4) + goalie_size;
  }
  
}


/* Berechnet den Schnittpunkt zweier Geraden */
Vector Score::intersection_point(Vector p1, Vector steigung1, 
						 Vector p2, Vector steigung2) {
  double x, y, m1, m2;
  if ((steigung1.getX() == 0) || (steigung2.getX() == 0)) {
    if (fabs(steigung1.getX()) < 0.00001) {
      return point_on_line(steigung2, p2, p1.getX());
    } else if (fabs(steigung1.getX()) < 0.00001) {
      return point_on_line(steigung1, p1, p2.getX());
    } 
  }
  m1 = steigung1.getY()/steigung1.getX();
  m2 = steigung2.getY()/steigung2.getX();
  if (m1 == m2) return Vector(-51.5, 0);
  x = (p2.getY() - p1.getY() + p1.getX()*m1 - p2.getX()*m2) / (m1-m2);
  y = (x-p1.getX())*m1 + p1.getY();
  return Vector (x, y);

}


/* Berechnet die y-Koordinate Punktes auf der Linie, der die x-Koordinate x hat
 */
Vector Score::point_on_line(Vector steigung, Vector line_point, double x) {
  //steigung.normalize();
  steigung = (1.0/steigung.getX()) * steigung;
  if (steigung.getX() > 0) {
    return (x - line_point.getX()) * steigung + line_point;
  }
  if (steigung.getX() < 0) {
    return (line_point.getX() - x) * steigung + line_point;
  }  // Zur Sicherheit, duerfte aber nie eintreten
  return line_point;
} /* point_on_line */


int Score::intercept_goalie(Vector ball_pos, Vector ball_vel, Vector goalie_pos, 
					      double goalie_size) {
  if (ball_vel.getX() < 0.0) return 1;
  if (ball_vel.norm() < 0.5) return 1;
  
  LOG_DAN(0, << "goalie_initial_size ist " << goalshot_param1);

  Vector b_pos = ball_pos;
  Vector b_vel = ball_vel;
  double goal_x = ServerOptions::pitch_length/2.0;

  double goalie_action_radius;

  int time = 1;
  bool wrong_angle = false;

  for (int i = 1; i < 50; ++i) {
    b_pos += b_vel;
    b_vel *= ServerOptions::ball_decay;
    goalie_action_radius = goalie_action_radius_at_time(time, goalie_size, 0);//0.4
    if (wrong_angle) goalie_action_radius -= 0.3;
    
    //LOG_DAN(0, << _2D << C2D(goalie_pos.x, goalie_pos.y, goalie_action_radius + 2.0, "#0000ff"));
    //LOG_DAN(0, << _2D << C2D(b_pos.x, b_pos.y, 1, "00ff00"));
    
    if (b_pos.getX() > goal_x + 0.3) {
      //LOG_DAN(0, << "time is " << time);
      return -time;

    } else if (((b_pos - goalie_pos).norm() - goalie_action_radius < 2.0) && wrong_angle) {//(time != i)) {
      return 1;
    } else if (((b_pos - goalie_pos).norm() - goalie_action_radius < 2.0) && !wrong_angle) {//(time == i)) {
      if (goalie_needs_turn_for_intercept(time, ball_pos, ball_vel, b_pos, b_vel, goalie_size)) {
	//time--;
	wrong_angle = true;
      } else return 1;
    }
    ++time;
  }
  return 1;
}

bool Score::goalie_needs_turn_for_intercept(int time, Vector initial_ball_pos, Vector initial_ball_vel, 
					      Vector b_pos, Vector b_vel, double goalie_size) {
  //test if goalie has a bad angle for the intercept
#if 0 // ridi 22.6.03 : replaced by new access to goalie
  PlayerSet alive_opps = WSinfo::alive_opponents;
  PPlayer goalie = alive_opps.get_player_by_number(WSinfo::ws->his_goalie_number);
#endif
  PPlayer goalie =WSinfo::his_goalie;


  if (goalie == NULL) return 0;
  if (goalie->age > 1) return 0;
  
  Vector goalie_ang;
  goalie_ang.init_polar(1.0, goalie->ang);
  
  Vector intersection = intersection_point(goalie->pos, goalie_ang, initial_ball_pos, initial_ball_vel);
  LOG_DAN(0, << _2D << C2D(intersection.x, intersection.y, 1, "ff0000"));
  
  //ridi: not used: Vector op_to_intersection = intersection - initial_ball_pos;
      
  double dist = (intersection-initial_ball_pos).norm();
  Vector ball_in_goal = Vector(point_on_line(initial_ball_vel, initial_ball_pos,ServerOptions::pitch_length/2.0));
  //LOG_DAN(0, "Ball ueberschreitet Torlinie an " << ball_in_goal.x << " " << ball_in_goal.y);
  if (dist - 2.0 > (ball_in_goal - initial_ball_pos).norm() &&
      ((b_pos - goalie->pos).norm() > 2.5)) {
    LOG_DAN(0, "Schnittpunkt zu weit entfernt!");
    return 1;
    //--time;
  } else {
    
    int time_L = time;
    Vector b_pos_L = b_pos;
    Vector b_vel_L = b_vel;
    
    for (int j = 0; j < 10; ++j) {
      if ((b_pos_L - initial_ball_pos).norm() > dist - 2.0) {
	double goalie_rad = goalie_action_radius_at_time(time_L, goalie_size, 0);
	double goalie_rad2 = goalie_action_radius_at_time(time_L+1, goalie_size, 0);
	Vector g_pos = intersection-goalie->pos;
	Vector g_pos2 = g_pos;
	g_pos.normalize(goalie_rad);
	g_pos += goalie->pos;
	
	g_pos2.normalize(goalie_rad2);
	g_pos2 += goalie->pos;
	
	LOG_DAN(0, << _2D << C2D(b_pos_L.x, b_pos_L.y, 1, "ff00ff"));
	if ((goalie_rad + 2.0 < (intersection-goalie->pos).norm()) &&
	    !is_pos_in_quadrangle(b_pos_L, goalie->pos, g_pos, 4.0) &&
	    (b_pos_L.distance(g_pos) > 2.2) &&
	    !is_pos_in_quadrangle(b_pos_L+b_vel_L, goalie->pos, g_pos2, 4.0) &&
	    (b_pos_L.distance(g_pos2) > 2.2) &&
	    ((b_pos-goalie->pos).norm() > 2.5) &&
	    ((goalie_rad*(intersection-goalie->pos)+goalie->pos - b_pos_L).norm() > 2.5)) {
	  LOG_DAN(0, << "goalie has bad angle for intercept => needs one extra step");
	  LOG_DAN(0, << _2D << C2D(g_pos.x, g_pos.y, 1, "ffff00"));
	  //--time;
	  return 1;
	  //break;
	} else return 0;
      } 
      
      b_pos_L += b_vel_L;
      b_vel_L *= ServerOptions::ball_decay;
      ++time_L;
    }
    return 0;
  }
}


bool Score::is_pos_in_quadrangle(Vector pos, Vector p1, Vector p2, Vector p3, Vector p4) {
  if ( Tools::point_in_triangle(pos, p1,p2,p3) ||
       Tools::point_in_triangle(pos, p1,p3,p4) ) { 
    return true;
    }
  return false;
}

bool Score::is_pos_in_quadrangle(Vector pos, Vector p1, Vector p2, double width) {
  Vector tmp= p2-p1;
  Vector norm;
  norm.setX( -tmp.getY() );
  norm.setY( tmp.getX() );
  norm.normalize(0.5*width);
  Vector g1= p1+ norm;
  Vector g2= p1- norm;
  Vector g3= p2- norm;
  Vector g4= p2+ norm;

  return is_pos_in_quadrangle(pos,g1,g2,g3,g4);
}

