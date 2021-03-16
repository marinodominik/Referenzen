/*
 * neuro_dribble_2017.cpp
 *
 *  Created on: 05.02.2017
 *      Author: tobias
 */

#include "neuro_dribble_2017.h"
#include "../../basics/log_macros.h"
#include "../../basics/tools.h"
#include "../../basics/ws_info.h"
#include "../../basics/ws_memory.h"
#include "../../basics/ws.h"
#include "../../basics/globaldef.h"
#include <stdio.h>
#include <math.h>

namespace NS_ND2017
{

CommandProvider::~CommandProvider(){}
SituationAssessment::~SituationAssessment(){}

//############################################################################
// CommandAnalysis
//############################################################################
const double CommandAnalysis::NEURODRIBBLE2017_BALL_SAFETY_MARGIN = 0.1;

CommandAnalysis::CommandAnalysis()
{

}

CommandAnalysis::~CommandAnalysis()
{
	// TODO Auto-generated destructor stub
}

double CommandAnalysis::get_best_command(Situation current_situation, Cmd& cmd, CommandProvider& provider, SituationAssessment& sit)
{
	double current_assessment, best_assessment;

	Vector my_curr_pos, my_curr_vel, ball_curr_pos, ball_curr_vel, my_next_pos, my_next_vel, ball_next_pos, ball_next_vel;
	ANGLE my_curr_ang, my_next_ang;
	int best_iteration;

	current_assessment = -1000;

	my_curr_pos = current_situation.get_player_position();
	my_curr_vel = current_situation.get_player_velocity();
	my_curr_ang = current_situation.get_player_angle();
	ball_curr_pos = current_situation.get_ball_position();
	ball_curr_vel = current_situation.get_ball_velocity();
	my_next_pos = Vector(0,0);
	my_next_vel = Vector(0,0);
	my_next_ang = 0;
	ball_next_pos = Vector(0,0);
	ball_next_vel = Vector(0,0);

	best_assessment = -1000;


	for(int i = 0; i < provider.get_max_iterations(); i++)
	{
		Cmd current_cmd = Cmd();
		provider.provide(current_cmd, i);
		Tools::model_cmd_main(my_curr_pos, my_curr_vel, my_curr_ang,
				ball_curr_pos, ball_curr_vel,
				current_cmd.cmd_body,
				my_next_pos, my_next_vel, my_next_ang,
				ball_next_pos, ball_next_vel);

		Situation next = Situation(my_next_pos, my_next_vel, my_next_ang, ball_next_pos, ball_next_vel, current_situation.get_target());

/*		if ( my_next_pos.distance( ball_next_pos ) >   WSinfo::me->kick_radius
		                                             + WSinfo::me->radius
		                                             + ServerOptions::ball_size
		                                             - 0.0)
		  continue;*/

		current_assessment = sit.transition(current_situation, next) + sit.assess(next);

		if(current_assessment > best_assessment)
		{
			best_assessment = current_assessment;
			best_iteration = i;
		}
	}

	provider.provide(cmd, best_iteration);

	return best_assessment;
}

//############################################################################
// DashProvider
//############################################################################
DashProvider::DashProvider(double min_power, double power_step, double max_power, double min_angle, double angle_step, double max_angle)
{
	this->min_power = min_power;
	this->power_step = power_step;
	this->max_power = max_power;
	this->min_angle = min_angle;
	this->max_angle = max_angle;
	this->angle_step = angle_step;

	this->power_iterations = ( this->max_power - this->min_power ) / this->power_step;

	if(this->angle_step != 0)
		this->angle_iterations = ( this->max_angle - this->min_angle ) / this->angle_step;
	else
		this->angle_iterations = 0;

	this->power_iterations = fabs(this->power_iterations) + 1;
	this->angle_iterations = fabs(this->angle_iterations) + 1;
}

DashProvider::~DashProvider() {
	// TODO Auto-generated destructor stub
}

int DashProvider::get_max_iterations()
{
	return this->power_iterations * this->angle_iterations;
}

void DashProvider::provide(Cmd& cmd, int iteration)
{
	double power, angle;

	power = this->min_power + ( this->power_step * ( iteration % this->power_iterations ) );
	angle = this->min_angle + ( this->angle_step * ( iteration / this->power_iterations ) );

	cmd.cmd_body.set_dash(power, DEG2RAD(angle));
}

bool DashProvider::is_of_type(Cmd& cmd)
{
	return cmd.cmd_body.get_type() == Cmd_Body::TYPE_DASH;
}

void DashProvider::copy(Cmd& copy_target, Cmd& copy_source)
{
	double power, angle;
	copy_source.cmd_body.get_dash(power, angle);
	copy_target.cmd_body.set_dash(power, angle);
}

std::string DashProvider::to_string(Cmd& cmd)
{
	double power, angle;
	cmd.cmd_body.get_dash(power, angle);
	return "Command: Dash(" + val2str(power) + ", " + val2str(RAD2DEG(angle)) + ")";
}

//############################################################################
// DribbleNetAssessment
//############################################################################
DribbleNetAssessment::DribbleNetAssessment(DribbleNetControl* control)
{
	this->net_ctrl = control;
}

DribbleNetAssessment::~DribbleNetAssessment(){
	//TODO: Implement?
	//remember: do not delete control
}

double DribbleNetAssessment::assess(Situation situation)
{
	double assessment = this->net_ctrl->forward(situation);

	//Da assessment hier Kosten orientiert ist, assess aber für bessere Situationen höhere Werte zurückgeben soll,
	//müssen die Kosten hier in Belohnungen umgerechnet werden.
	return assessment * -1;
}

double DribbleNetAssessment::transition(Situation situation, Situation next_situation)
{
	return this->net_ctrl->calculate_cost(situation, next_situation) * -1;
}

//############################################################################
// DribbleNetControl
//############################################################################
std::string DribbleNetControl::path = ND2017_PATH;
std::string DribbleNetControl::default_file_name = ND2017_DEFAULT_FILE_NAME;
std::string DribbleNetControl::file_ending = ND2017_FILE_ENDING;
double DribbleNetControl::ball_size = 0.085;

DribbleNetControl::DribbleNetControl(const char* file_name)
{
	this->net = new Net();

	char* name = new char[200];
	strcpy(name, file_name);
	this->delta_null = 0.001;

	if( this->load_net(name) )
	{
		LOG_POL(0, << "DribbleNetControl loaded " << name);
		//this->delta_null = this->net->update_params[0];
	}
	else
	{
		LOG_POL(0, << "DribbleNetControl creates " << name);

		int* net_layers;
		float* uparams;

		int i = 0;
		uparams = new float[MAX_PARAMS];
		uparams[i++] = this->delta_null;
		uparams[i++] = 10.0;
		while( i < MAX_PARAMS ) { uparams[i++] = 0.0; }

		this->num_layers = 3;
		this->num_input = 8;
		this->num_hidden = 22;
		this->num_output = 1;

		i = 0;
		net_layers = new int[this->num_layers];
		net_layers[i++] = this->num_input;
		net_layers[i++] = this->num_hidden;
		net_layers[i++] = this->num_output;

		i = 1;
		this->net->create_layers(this->num_layers, net_layers);
		this->net->set_layer_act_f(i++, LOGISTIC);
		this->net->set_layer_act_f(i++, LINEAR);

		this->net->connect_layers();
		this->net->init_weights(0, 0.5);
		this->net->set_update_f(RPROP, uparams);

		delete[] uparams;
		delete[] net_layers;

		this->save_net(name);
	}

	delete[] name;
}

DribbleNetControl::~DribbleNetControl()
{
	delete this->net;
}

void DribbleNetControl::set_target_angle(ANGLE angle)
{
	this->target_angle = ANGLE(angle);
}

bool DribbleNetControl::load_net(const char* file_name)
{
	char* name = new char[200];
	strcpy(name, file_name);
	bool returnValue = this->net->load_net(name) == 0;
	LOG_POL(1, "Loading " << name << " success: " << returnValue);

	if( returnValue )
	{
		this->num_layers = this->net->topo_data.layer_count;
		this->num_input = this->net->topo_data.in_count;
		this->num_hidden = this->net->topo_data.hidden_count;
		this->num_output = this->net->topo_data.out_count;
	}

	delete[] name;
	return returnValue;
}

bool DribbleNetControl::save_net(const char* file_name)
{
	char* name = new char[200];
	strcpy(name, file_name);
	bool returnValue = this->net->save_net(name) == 0;
	LOG_POL(1, "Saving " << name << " success: " << returnValue);
	delete[] name;
	return returnValue;
}

void DribbleNetControl::receive_input(Situation situation, bool report)
{
	ANGLE to_target = figure_out_how_to_correct(situation.get_player_angle(), situation.get_target());
	Vector relative_player_velocity, relative_ball_position, relative_ball_velocity, target;
	double rel_b_p_x, rel_b_p_y;
	rel_b_p_x = situation.get_ball_position().getX() - situation.get_player_position().getX();
	rel_b_p_y = situation.get_ball_position().getY() - situation.get_player_position().getY();
	relative_ball_position = Vector(rel_b_p_x, rel_b_p_y);
	relative_player_velocity = situation.get_player_velocity();
	relative_player_velocity.ROTATE(situation.get_player_angle() * (-1));
	relative_ball_velocity = situation.get_ball_velocity();
	relative_ball_velocity.ROTATE(situation.get_player_angle() * (-1));
	relative_ball_position.ROTATE(situation.get_player_angle() * (-1));
	target = Vector(to_target).normalize(1);

	int i = 0;
	this->net->in_vec[i++] = relative_player_velocity.getX();
	this->net->in_vec[i++] = relative_player_velocity.getY();
	this->net->in_vec[i++] = relative_ball_position.getX();
	this->net->in_vec[i++] = relative_ball_position.getY();
	this->net->in_vec[i++] = relative_ball_velocity.getX();
	this->net->in_vec[i++] = relative_ball_velocity.getY();
	this->net->in_vec[i++] = target.getX();
	this->net->in_vec[i++] = target.getY();

	if(report)
	{
		LOG_POL(2, << "--Dribble Net reporting--");
		LOG_POL(2, << "Player Position: " << situation.get_player_position());
		LOG_POL(2, << "Player Velocity: " << situation.get_player_velocity());
		LOG_POL(2, << "Player Angle: " << RAD2DEG(situation.get_player_angle().get_value_mPI_pPI()));
		LOG_POL(2, << "Ball Position: " << situation.get_ball_position());
		LOG_POL(2, << "Ball Velocity: " << situation.get_ball_velocity());
		LOG_POL(2, << "Rel Player Velocity: " << relative_player_velocity);
		LOG_POL(2, << "Rel Ball Position: " << relative_ball_position);
		LOG_POL(2, << "Rel Ball Velocity: " << relative_ball_velocity);
		LOG_POL(2, << "To Target: " << RAD2DEG(to_target.get_value_mPI_pPI()));
		LOG_POL(2, << "--End control log--");
		for(int j = 0; j < this->num_input; j++)
		{
			LOG_POL(2, << "In Vec " << j << ": " << this->net->in_vec[j]);
		}
	}
}

double DribbleNetControl::forward(Situation situation)
{
	this->receive_input(situation);
	this->net->forward_pass(this->net->in_vec, this->net->out_vec);
	return this->net->out_vec[0];
}

void DribbleNetControl::backward(Situation situation, double target_value, bool report)
{
	double current = this->forward(situation);
	this->net->out_vec[0] = current - target_value;

	if( report )
	{
		LOG_POL(0, << "Target Value = " << target_value << " Current Value = " << current);
		LOG_POL(0, << "out_vec: " << this->net->out_vec[0] );
	}

	this->receive_input(situation, report);

	this->net->backward_pass(this->net->out_vec, this->net->in_vec);
}

double DribbleNetControl::calculate_cost(Situation situation, Situation next_situation)
{
	double cost = 0;
	bool ball_in_kick_range, ball_collision;

	ball_in_kick_range = kickable(next_situation, WSinfo::me->kick_radius);
	ball_collision = collision(next_situation, WSinfo::me->radius) || collision(situation, next_situation, WSinfo::me->radius);

	//if(ball_in_kick_range && !ball_collision)
	//{
		Vector perfect_spot;
		double perfect_distance;
		Vector p_pos, p_vel, b_pos, b_vel;
		ANGLE p_ang;
		Cmd_Body perfect_cmd = Cmd_Body();
		perfect_cmd.set_dash(100, 0);
		Tools::model_cmd_main(situation.get_player_position(), situation.get_player_velocity(), situation.get_player_angle(),
				situation.get_ball_position(), situation.get_ball_velocity(),
				perfect_cmd,
				p_pos, p_vel ,p_ang,
				b_pos, b_vel);
		perfect_distance = p_pos.distance(situation.get_player_position());

		perfect_spot = Vector(situation.get_target());
		perfect_spot.normalize(perfect_distance);
		perfect_spot.setX( perfect_spot.getX() + situation.get_player_position().getX() );
		perfect_spot.setY( perfect_spot.getY() + situation.get_player_position().getY() );

		/*
		//Every cost for a given start situation should assume the same here.
		LOG_POL(2, << _2D << C2D(perfect_spot.x, perfect_spot.y, WSinfo::me->kick_radius, "55ffff"));
		LOG_POL(2, << _2D << P2D(perfect_spot.x, perfect_spot.y, "55ffff"));
		LOG_POL(2, << "My pos: " << situation.get_player_position() << " perfect pos: " << perfect_spot);
		*/

		cost += next_situation.get_player_position().distance(perfect_spot) - situation.get_player_position().distance(perfect_spot);
		cost *= 10;
		//cost += next_situation.get_player_angle().diff(this->target_angle);

	//}
	//else
		//cost = 10;

	if( !ball_in_kick_range )
	{
		cost = 10;
	}

	if( ball_collision )
	{
		cost = 5;
	}

	//cost *= 10;

	return cost;
}

void DribbleNetControl::update_weights()
{
	this->net->update_weights();
}

void DribbleNetControl::report(Situation situation)
{
	this->receive_input(situation, true);
}

ANGLE DribbleNetControl::figure_out_how_to_correct(ANGLE my_angle, ANGLE other_angle)
{
	Vector my_v, o_v;
	//my_v = Vector(my_angle);
	o_v = Vector(other_angle);

	//my_v.ROTATE(my_angle * (-1));
	o_v.ROTATE(my_angle * (-1));

	return o_v.ARG();
}

bool DribbleNetControl::collision(Situation situation, double radius)
{
	bool returnValue = situation.get_player_position().distance(situation.get_ball_position()) <= radius + ball_size;
	return returnValue;
}

bool DribbleNetControl::collision(Situation situation, Situation next_situation, double radius)
{
	bool returnValue = false;
	Vector trajectory_player = next_situation.player_position - situation.player_position;
	Vector trajectory_ball = next_situation.ball_position - situation.ball_position;
	trajectory_player = trajectory_player.normalize(situation.player_position.distance(next_situation.ball_position));
	trajectory_ball = trajectory_ball.normalize(situation.ball_position.distance(next_situation.player_position));

	returnValue |= ( situation.player_position + trajectory_player ).distance(next_situation.ball_position) <= radius + ball_size;
	returnValue |= ( situation.ball_position + trajectory_ball ).distance(next_situation.player_position) <= radius + ball_size;

	return returnValue;
}

bool DribbleNetControl::kickable(Situation situation, double kick_radius)
{
	return situation.get_player_position().distance(situation.get_ball_position()) <= kick_radius - ball_size;

}

//############################################################################
// DribbleNetTrainControl
//############################################################################
DribbleNetTrainControl::DribbleNetTrainControl(DribbleNetControl* control, CommandProvider** providers, int num_com, SituationAssessment* sit)
{
	this->control = control;
	this->training_data = new TrainingsData(providers, num_com, sit, control);
	this->buffer_state = Situation();
	this->buffered = false;
	this->session = 1;
	this->fresh = true;
}

DribbleNetTrainControl::~DribbleNetTrainControl()
{
	delete this->training_data;
}

void DribbleNetTrainControl::add_transition(Situation situation, Situation next_situation)
{
	TemporalDifference new_member = TemporalDifference(situation, next_situation, this->control);
	LOG_POL(3, << "TDValue: " << new_member.value);
	this->training_data->add(new_member);
}

void DribbleNetTrainControl::clear_buffer()
{
	this->buffered = false;
}

bool DribbleNetTrainControl::reasonable_to_add(Situation situation)
{
	bool kickable, collision;

	kickable = DribbleNetControl::kickable(situation, WSinfo::me->kick_radius);
	collision = DribbleNetControl::collision(situation, WSinfo::me->radius);
	/*
	if( kickable )
	{
		std::queue<TemporalDifference> helper = this->state_queue;
		while( !helper.empty() )
		{
			TemporalDifference state = helper.front();
			helper.pop();
			if( state.situation.equals(situation) && state.next_situation.equals(next_situation) )
				return false;
		}
	}
	 */

	return kickable && !collision;
}

void DribbleNetTrainControl::add_situation(Situation situation)
{
	if( !this->reasonable_to_add(situation) && !this->reasonable_to_add(this->buffer_state) )
	{
		this->clear_buffer();
	}
	else
	{
		if( this->buffered  )
		{
			this->add_transition(this->buffer_state, situation);
		}

		this->buffer_state = situation;
		this->buffered = true;
	}
}

void DribbleNetTrainControl::train()
{
	std::queue<TemporalDifference> trainings_queue = std::queue<TemporalDifference>();

	int entries = 20000;
	double init_error, current_error;

	if( this->training_data->get_total_size() < entries )
	{
		entries = this->training_data->get_total_size();
		for( int i = 0; i < entries; i++ )
		{
			trainings_queue.push(this->training_data->draw_next());
		}
	}
	else
	{
		while( (unsigned) entries > trainings_queue.size() )
		{
			trainings_queue.push(this->training_data->draw_fresh_or_random());
		}
	}

	init_error = current_error = this->calculate_error(trainings_queue);
	LOG_POL(0, << "DribbleNetTrainControl is training with " << entries << " entries");
	this->clear_buffer();
	LOG_POL(2, << "Initial error: " << init_error);
	printf("\nInitial Error: ");
	printf("%s", val2str(init_error).data());

	for( int repetition = 0; repetition < 300 /*200*/; repetition++ )
	{
		std::queue<TemporalDifference> helper = trainings_queue;

		while( !helper.empty() )
		{
			TemporalDifference state = helper.front();
			helper.pop();
			bool report = repetition == 90 && helper.size() == 0;
			this->control->backward(state.situation, state.value, report);
		}

		this->control->update_weights();

		current_error = this->calculate_error(trainings_queue);

		if(repetition % 20 == 0)
		{
			LOG_POL(2, << "Error after " << repetition << " repetitions: " << current_error);
			printf("\nError: ");
			printf("%s", val2str(current_error).data());
		}
	}

	this->training_data->clean();

	char* new_name = new char[200];
	strcpy(new_name, (DribbleNetControl::path + DribbleNetControl::default_file_name + "_" + val2str(session) + DribbleNetControl::file_ending).c_str());
	this->control->save_net(new_name);

	delete[] new_name;
	session++;
}

double DribbleNetTrainControl::calculate_error(std::queue<TemporalDifference> queue)
{
	std::queue<TemporalDifference> helper = queue;
	double error, num_states;

	num_states = helper.size();
	error = 0;

	while( !helper.empty() )
	{
		TemporalDifference state = helper.front();
		helper.pop();
		double output = this->control->forward(state.situation);
		error += ( state.value - output ) * ( state.value - output );
	}

	return error / num_states;
}

void DribbleNetTrainControl::eval_next(int step)
{
	this->session += step;

	char* new_name = new char[200];
	strcpy(new_name, (DribbleNetControl::path + DribbleNetControl::default_file_name + "_" + val2str(session) + DribbleNetControl::file_ending).c_str());
	this->control->load_net(new_name);

	delete[] new_name;
}

void DribbleNetTrainControl::eval_all(int step)
{
	if( this->fresh == true )
	{
		this->session = 0;
		this->fresh = false;

		char* new_name = new char[200];
		strcpy(new_name, (DribbleNetControl::path + DribbleNetControl::default_file_name + DribbleNetControl::file_ending).c_str());
		this->control->load_net(new_name);

		delete[] new_name;
	}
	else
	{
		this->eval_next(step);
	}
}

//############################################################################
// EvalControl
//############################################################################
EvalControl::EvalControl(int number_of_episodes_per_sequence)
{
	this->episode_counter = 0;
	this->sequence_counter = 0;
	this->time_step_counter = 0;
	this->in_episode = false;
	this->in_sequence = false;
	this->number_of_episodes_per_sequence = number_of_episodes_per_sequence;
	this->lost_ball = 0;
	this->distances = std::queue<double>();
	this->x_distances = std::queue<double>();
	this->durations = std::queue<double>();
	this->distance = 0;
	this->x_distance = 0;
}

EvalControl::~EvalControl()
{

}

void EvalControl::next_state(Situation situation, double sequence_duration)
{
	if( this->in_sequence == false )
	{
		this->start_sequence();
	}
	else if( this->in_episode == false )
	{
		this->start_episode();
	}

	if( this->time_step_counter > 0 )
	{
		double v_distance = situation.player_position.distance(this->last_state.player_position),
				norm = this->last_state.player_velocity.norm();

		if( v_distance > norm + 2 )
		{
			this->distance += norm;
		}
		else
		{
			this->distance += v_distance;
		}

		if( situation.player_position.getX() > this->last_state.player_position.getX() )
		{
			double abs_old, abs_new;
			abs_old = abs(this->last_state.player_position.getX());
			abs_new = abs(situation.player_position.getX());
			this->x_distance += abs(abs_old - abs_new);
		}
	}

	this->durations.push(sequence_duration);
	this->last_state = situation;
	this->time_step_counter++;
}

void EvalControl::start_episode()
{
	LOG_POL(0, << "Started episode");
	this->episode_counter++;
	this->time_step_counter = 0;
	this->distance = 0;
	this->x_distance = 0;
	this->in_episode = true;
}

void EvalControl::start_sequence()
{
	LOG_POL(0, << "Started sequence");
	this->sequence_counter++;
	this->in_sequence = true;
	this->episode_counter = 0;
	this->start_episode();
}

void EvalControl::end_episode()
{
	//this->distances[this->episode_counter - 1] = this->first_state.player_position.distance(this->last_state.player_position);
	this->distances.push(this->distance);
	this->x_distances.push(this->x_distance);
	this->in_episode = false;
	if( !DribbleNetControl::kickable(this->last_state, WSinfo::me->kick_radius) )
		this->lost_ball++;
}

void EvalControl::end_sequence(bool was_eval)
{
	if( this->in_episode == true )
	{
		this->end_episode();
	}

	this->in_sequence = false;

	Analysis distance_a = Analysis(this->distances);
	Analysis x_distance_a = Analysis(this->x_distances);
	Analysis duration_a = Analysis(this->durations);

	std::string file;

	if( was_eval == true )
	{
		//this->sequence_counter--;
		file = "data/nets_neuro_dribble2017/eval.txt";
	}
	else
	{
		file = "data/nets_neuro_dribble2017/training.txt";
	}

	while( this->distances.empty() == false )
	{
		this->distances.pop();
	}

	while( this->durations.empty() == false )
	{
		this->durations.pop();
	}

	while( this->x_distances.empty() == false )
	{
		this->x_distances.pop();
	}

	std::ofstream ofs(file.c_str(), std::ios::app);

	ofs << "Sequence: " << this->sequence_counter << " Average Distance: " << distance_a.average << " Maximum: " << distance_a.max1 << " 2nd Maximum: " << distance_a.max2 << " Minimum: " << distance_a.min << " Standard Deviation: " << distance_a.standard_deviation << " Ball lost: " << this->lost_ball << " Episodes in sequence: " << this->episode_counter << std::endl;

	ofs << "x Average Distance: " << x_distance_a.average << " Maximum: " << x_distance_a.max1 << " 2nd Maximum: " << x_distance_a.max2 << " Minimum: " << x_distance_a.min << " Standard Deviation: " << x_distance_a.standard_deviation << std::endl;

	ofs << "Average Duration: " << duration_a.average << " Maximum: " << duration_a.max1 << " Minimum: " << duration_a.min << " Standard Deviation: " << duration_a.standard_deviation << std::endl;

	if( this->number_of_episodes_per_sequence != this->episode_counter )
	{
		ofs << "Previous Data invalid! Number of episodes should be " << this->number_of_episodes_per_sequence << std::endl;
	}

	if( distance_a.size != this->episode_counter || x_distance_a.size != this->episode_counter )
	{
		ofs << "Previous Data invalid! Not all data was stored." << std::endl;
	}

	ofs << "----" << std::endl;

	ofs.close();

	this->lost_ball = 0;
}

//############################################################################
// KickProvider
//############################################################################
KickProvider::KickProvider(double min_power, double power_step, double max_power, double min_angle, double angle_step, double max_angle)
{
	this->min_power = min_power;
	this->power_step = power_step;
	this->max_power = max_power;
	this->min_angle = min_angle;
	this->max_angle = max_angle;
	this->angle_step = angle_step;

	this->power_iterations = ( this->max_power - this->min_power ) / this->power_step;

	if(this->angle_step != 0)
		this->angle_iterations = ( this->max_angle - this->min_angle ) / this->angle_step;
	else
		this->angle_iterations = 0;

	this->power_iterations = fabs(this->power_iterations) + 1;
	this->angle_iterations = fabs(this->angle_iterations) + 1;
}

KickProvider::~KickProvider() {
	// TODO Auto-generated destructor stub
}

int KickProvider::get_max_iterations()
{
	return this->power_iterations * this->angle_iterations;
}

void KickProvider::provide(Cmd& cmd, int iteration)
{
	double power, angle;

	power = this->min_power + ( this->power_step * ( iteration % this->power_iterations ) );
	angle = this->min_angle + ( this->angle_step * ( iteration / this->power_iterations ) );

	cmd.cmd_body.set_kick(power, DEG2RAD(angle));
}

bool KickProvider::is_of_type(Cmd& cmd)
{
	return cmd.cmd_body.get_type() == Cmd_Body::TYPE_KICK;
}

void KickProvider::copy(Cmd& copy_target, Cmd& copy_source)
{
	double power, angle;
	copy_source.cmd_body.get_kick(power, angle);
	copy_target.cmd_body.set_kick(power, angle);
}

std::string KickProvider::to_string(Cmd& cmd)
{
	double power, angle;
	cmd.cmd_body.get_kick(power, angle);
	return "Command: Kick(" + val2str(power) + ", " + val2str(RAD2DEG(angle)) + ")";
}

//############################################################################
// RandomDribbleAssessment
//############################################################################
RandomDribbleAssessment::RandomDribbleAssessment()
{
}

RandomDribbleAssessment::~RandomDribbleAssessment()
{
}

double RandomDribbleAssessment::assess(Situation situation)
{
	if( !DribbleNetControl::kickable(situation, WSinfo::me->kick_radius) || DribbleNetControl::collision(situation, WSinfo::me->radius) )
	{
		return -1000;
	}
	else
	{
		return rand() % 1000;
	}
}

double RandomDribbleAssessment::transition(Situation situation, Situation next_situation)
{
	if( situation.get_player_angle().diff(next_situation.get_player_angle()) > DEG2RAD(15) )
	{
		return -1000;
	}
	else
	{
		return 0;
	}
}

//############################################################################
// TrainingsData
//############################################################################
TrainingsData::TrainingsData(CommandProvider** providers, int num_com, SituationAssessment* sit, DribbleNetControl* control)
{
	this->providers = providers;
	this->num_com = num_com;
	this->analysis = new CommandAnalysis();
	this->sit = sit;
	this->control = control;
	this->old_states = StateType();
	this->fresh_states = StateType();
	this->used_states = StateType();
}

TrainingsData::~TrainingsData()
{
	delete this->analysis;
}

TemporalDifference TrainingsData::extract_adjusted_element_at(int index)
{
	int current_index = 0;
	int fresh, old;
	TemporalDifference state;
	fresh = this->fresh_states.size();
	old = fresh + this->old_states.size();
	/*
	if(index < fresh)
	{
		state = this->fresh_states[index];
		this->fresh_states.erase(this->fresh_states.begin() + index);
	}
	else
	{
		state = this->old_states[index - fresh];
		this->old_states.erase(this->old_states.begin() + ( index - fresh ));
		state = this->simulate(state);
	}
	*/
	if(index < fresh)
	{
		current_index = fresh - 1;
	}
	else
	{
		current_index = old - 1;
	}

	do
	{
		if( index < fresh )
		{
			state = this->fresh_states.front();
			this->fresh_states.pop();
			if( current_index != index )
			{
				this->fresh_states.push(state);
				current_index--;
			}
		}
		else if ( index < old  )
		{
			state = this->old_states.front();
			this->old_states.pop();
			if( current_index == index )
			{
				state = this->simulate(state);
			}
			else
			{
				this->old_states.push(state);
				current_index--;
			}
		}
		else
		{
			break;
		}
	}
	while( current_index != index );

	this->used_states.push(state);

	//this->used_states.push_back(state);
	return state;
}

int TrainingsData::get_total_size()
{
	return this->fresh_states.size() + this->old_states.size();
}

void TrainingsData::clean()
{
	while( this->fresh_states.empty() == false )
	{
		TemporalDifference state = this->fresh_states.front();
		this->fresh_states.pop();
		this->old_states.push(state);
	}

	while( this->used_states.empty() == false )
	{
		TemporalDifference state = this->used_states.front();
		this->used_states.pop();
		this->old_states.push(state);
	}
}

void TrainingsData::add(TemporalDifference state)
{
	this->fresh_states.push(state);
}

TemporalDifference TrainingsData::draw_random()
{
	int i = rand() % this->get_total_size();
	int fresh = this->fresh_states.size();

	if( i >= fresh )
	{
		i = ( this->get_total_size() - 1 ) - ( ( rand() % ( this->old_states.size() - 1 ) ) % 10000 );
	}


	return this->extract_adjusted_element_at(i);
}

TemporalDifference TrainingsData::draw_fresh_or_random()
{
	if( this->fresh_states.size() > 0 )
	{
		return this->extract_adjusted_element_at(this->fresh_states.size() - 1);
	}
	else
	{
		return this->draw_random();
	}
}

TemporalDifference TrainingsData::simulate(TemporalDifference state)
{
	Cmd* cmds = new Cmd[this->num_com];
	int* assessments = new int[this->num_com];
	int chosen;
	chosen = 0;

	TemporalDifference new_state;

	for(int j = 0; j < this->num_com; j++)
	{
		cmds[j] = Cmd();
		assessments[j] = this->analysis->get_best_command(state.situation, cmds[j], *this->providers[j], *this->sit);

		if(assessments[j] > assessments[chosen])
		{
			chosen = j;
		}
	}

	Vector bnp, bnv, pnp, pnv;
	ANGLE pna;

	Tools::model_cmd_main(state.situation.get_player_position(), state.situation.get_player_velocity(), state.situation.get_player_angle(),
			state.situation.get_ball_position(), state.situation.get_ball_velocity(),
			cmds[chosen].cmd_body,
			pnp, pnv, pna, bnp, bnv);

	Situation next = Situation(pnp, pnv, pna, bnp, bnv, state.situation.get_target());

	new_state = TemporalDifference(state.situation, next, this->control);

	delete[] assessments;
	delete[] cmds;

	return new_state;
}

TemporalDifference TrainingsData::draw_next()
{
	if( this->get_total_size() == 0 )
	{
		return TemporalDifference();
	}

	return this->extract_adjusted_element_at(this->get_total_size() - 1);
}

//############################################################################
// TurnProvider
//############################################################################
TurnProvider::TurnProvider(double min, double step, double max)
{
	this->min = min;
	this->step = step;
	this->max = max;

	this->iterations = ( max - min ) / step;
	this->iterations = fabs(this->iterations) + 1;
}

TurnProvider::~TurnProvider() {
	// TODO Auto-generated destructor stub
}

int TurnProvider::get_max_iterations()
{
	return this->iterations;
}

void TurnProvider::provide(Cmd& cmd, int iteration)
{
	double angle;

	angle = this->min + ( step * iteration );

	cmd.cmd_body.set_turn(DEG2RAD(angle));
}

bool TurnProvider::is_of_type(Cmd& cmd)
{
	return cmd.cmd_body.get_type() == Cmd_Body::TYPE_TURN;
}

void TurnProvider::copy(Cmd& copy_target, Cmd& copy_source)
{
	double angle;
	copy_source.cmd_body.get_turn(angle);
	copy_target.cmd_body.set_turn(angle);
}

std::string TurnProvider::to_string(Cmd& cmd)
{
	double angle;
	cmd.cmd_body.get_turn(angle);
	return "Command: Turn(" + val2str(RAD2DEG(angle)) + ")";
}

} // end of namespace NS_ND2017

//############################################################################
// N E U R O D R I B B L E 2 0 1 7
//############################################################################
using namespace NS_ND2017;

bool NeuroDribble2017::initialized = false;

NeuroDribble2017::NeuroDribble2017()
{
	this->control = new DribbleNetControl();
	this->num_com = 3;
	this->providers = new CommandProvider*[num_com];
	this->analysis = CommandAnalysis();
	this->sit = new DribbleNetAssessment(this->control);
	this->random_sit = new RandomDribbleAssessment();

	int i = 0;
	this->providers[i++] = new KickProvider(5, 5, 100, -180, 5, 175);
	this->providers[i++] = new DashProvider(5, 5, 100, -180, 5, 175);
	this->providers[i++] = new TurnProvider(-180, 5, 180);

	this->training = false;
	this->trainer = new DribbleNetTrainControl(this->control, this->providers, this->num_com, this->sit);
	this->last_sit = Situation();
}

NeuroDribble2017::~NeuroDribble2017()
{
	delete this->sit;

	for(int i = 0; i < this->num_com; i++)
		delete this->providers[i];

	delete[] this->providers;
	delete this->control;
	delete this->random_sit;

	delete this->trainer;
}

bool NeuroDribble2017::init(const char* conf_file, int argc, const char* const* argv)
{
	if(initialized)
		return true;

	bool returnValue = true;

	initialized = returnValue;
	return returnValue;
}

void NeuroDribble2017::reset_intention()
{

}

void NeuroDribble2017::set_target(ANGLE target)
{
	this->target = target;
	this->control->set_target_angle(target);
}

bool NeuroDribble2017::get_cmd(Cmd& cmd)
{
	int chosen, random;
	double* assessments = new double[this->num_com];
	Cmd* cmds = new Cmd[this->num_com];
	bool set = false;
	Vector pos, vel, b_pos, b_vel;
	ANGLE ang;

	for(int i = 0; i < this->num_com; i++)
	{
		assessments[i] = -1000;
	}

	chosen = 0;

	random = rand() % 100;

	pos  = WSinfo::me->pos;
	vel  = WSinfo::me->vel;
	ang = WSinfo::me->ang;
	b_pos  = WSinfo::ball->pos;
	b_vel  = WSinfo::ball->vel;

	Situation current = Situation(pos, vel, ang, b_pos, b_vel, this->target);

	if( this->training )
	{
		this->trainer->add_situation(current);
		this->control->report(current);

		Situation next;

		if(random >= 95)
		{
			LOG_POL(1, << "Selecting random action");

			chosen = rand() % this->num_com;
			cmds[chosen] = Cmd();
		 	assessments[chosen] = this->analysis.get_best_command(current, cmds[chosen], *this->providers[chosen], *this->random_sit);

			set = true;
		}
	}

	if( !set )
	{
		LOG_POL(1, << "Selecting neuro action");
		for(int i = 0; i < this->num_com; i++)
		{
			cmds[i] = Cmd();
			assessments[i] = analysis.get_best_command(current, cmds[i], *this->providers[i], *this->sit);
			if(assessments[i] > assessments[chosen])
			{
				chosen = i;
			}
		}
	}

	this->providers[chosen]->copy(cmd, cmds[chosen]);

	LOG_POL(0, << "Selected i " << chosen << " with assessment " << assessments[chosen]);
	LOG_POL(0, << "Provides " << this->providers[chosen]->get_max_iterations() << " iterations" );
	LOG_POL(0, << this->providers[chosen]->to_string(cmd));

	delete[] assessments;
	delete[] cmds;

	return true;
}

void NeuroDribble2017::start_training()
{
	LOG_POL(0, << "NeuroDribble started training");

	this->training = true;

	Vector pos, vel, b_pos, b_vel;
	ANGLE ang;

	pos  = WSinfo::me->pos;
	vel  = WSinfo::me->vel;
	ang = WSinfo::me->ang;
	b_pos  = WSinfo::ball->pos;
	b_vel  = WSinfo::ball->vel;

	Situation current = Situation(pos, vel, ang, b_pos, b_vel, this->target);

	if( !this->last_sit.equals(Situation()) )
	{
		LOG_POL(1, "Cost for current: " << this->control->calculate_cost(this->last_sit, current));
		LOG_POL(1, "Net for current: " << this->control->forward(current));
	}

	this->last_sit = current;
}

void NeuroDribble2017::execute_training()
{
	LOG_POL(0, << "NeuroDribble executed training");
	this->training = false;
	this->trainer->train();
	this->last_sit = Situation();
}

void NeuroDribble2017::pause_training()
{
	LOG_POL(0, << "NeuroDribble paused training");
	this->training = false;
	this->trainer->clear_buffer();
	this->last_sit = Situation();
}

void NeuroDribble2017::eval_next(int step)
{
	this->training = false;
	this->trainer->eval_next(step);
}

// Add-On by TG17
bool NeuroDribble2017::is_safe()
{
    // Consideration of existing near-by opponents for which NeuroDribble2017
    // has, unfortunately, never been trained.
    bool returnValue = true;
    double maxDist = 4.0;
    PlayerSet harmingOpps = WSinfo::valid_opponents;
    harmingOpps.keep_players_in_circle( WSinfo::me->pos, maxDist);
    LOG_POL(0,<<_2D<<VC2D(WSinfo::me->pos,maxDist,"aaffaal"));
    for ( int p = 0; p < harmingOpps.num; p++)
    {
      PPlayer opp = harmingOpps[p];
      LOG_POL(0, << "NeuroDribble harm check for opp " << opp->number);
      // danger component 1: distance
      double minDist = 1.0;
      double dang1 = 1.0 - Tools::max(0.0, (opp->pos.distance(WSinfo::me->pos)-minDist) / (maxDist-minDist));
      LOG_POL(0, << "NeuroDribble harm check dist " << opp->pos.distance(WSinfo::me->pos)
        << " -> " << dang1);
      // danger component 2: angular deviation from target dir
      ANGLE meToHim( opp->pos.getX() - WSinfo::me->pos.getX(),
                     opp->pos.getY() - WSinfo::me->pos.getY() );
      ANGLE deltaTDir = target - meToHim;
      double dang2 = 1.0 - (fabs(deltaTDir.get_value_mPI_pPI()) / PI);
      dang2 *= dang2; // square it
      LOG_POL(0, << "NeuroDribble harm check ang dev tdir " << RAD2DEG(deltaTDir.get_value_mPI_pPI())
        << " -> " << dang2);
      // danger component 3: his body orientation
      ANGLE deltaBody = opp->ang - meToHim;
      double dang3 = pow( fabs(deltaBody.get_value_mPI_pPI()) / PI, 1.5 );
      LOG_POL(0, << "NeuroDribble harm check ang dev body " << RAD2DEG(deltaBody.get_value_mPI_pPI())
        << " -> " << dang3);
      // danger component 4: his vel
      int maxMove = 1.0;
      Vector oppNextNoAction = opp->pos + opp->vel;
      double approach =   WSinfo::me->pos.distance( opp->pos )
                        - WSinfo::me->pos.distance( oppNextNoAction );
      double dang4 = Tools::min(1.0, approach / maxMove) / 2.0 + 0.5;
      LOG_POL(0, << "NeuroDribble harm check approach (vel="<<opp->vel.norm()
        <<") " << approach << " -> " << dang4);
      // weighted danger sum
      double dangSum =   0.40 * dang1
                       + 0.20 * dang2
                       + 0.20 * dang3
                       + 0.20 * dang4;
      LOG_POL(0, << "NeuroDribble harm check weighted dang sum ### " << dangSum << " ###");
      LOG_POL(0,<<_2D<<VC2D(opp->pos,1.0,"ffffaa"));
      LOG_POL(0,<<_2D<<VSTRING2D(opp->pos, dangSum, "ffffaa"));
      if (dangSum > 0.45) returnValue = false;
    }
    LOG_POL(0, << "NeuroDribble safeness check (dir="
      << RAD2DEG(this->target.get_value_mPI_pPI())<<"): " << returnValue);
    return returnValue;
}
