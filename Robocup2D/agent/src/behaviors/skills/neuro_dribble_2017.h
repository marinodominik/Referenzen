/*
 * neuro_dribble_2017.h
 *
 *  Created on: 05.02.2017
 *      Author: tobias
 */

#ifndef BS2K_BEHAVIORS_SKILLS_NEURO_DRIBBLE_2017_H_
#define BS2K_BEHAVIORS_SKILLS_NEURO_DRIBBLE_2017_H_

#include "../base_bm.h"
#include <queue>
#include <sstream>
#include "../../../lib/src/n++.h"

namespace NS_ND2017
{

//############################################################################
// Situation
//############################################################################
struct Situation
{
public:
	Situation()
	{
		this->player_position = Vector(0, 0);
		this->player_velocity = Vector(0, 0);
		this->player_angle = ANGLE(0);
		this->ball_position = Vector(0, 0);
		this->ball_velocity = Vector(0, 0);
		this->target = ANGLE(0);
	}

	Situation(Vector player_position, Vector player_velocity, ANGLE player_angle, Vector ball_position, Vector ball_velocity, ANGLE target)
	{
		this->player_position = player_position;
		this->player_velocity = player_velocity;
		this->player_angle = player_angle;
		this->ball_position = ball_position;
		this->ball_velocity = ball_velocity;
		this->target = target;
	}

	Situation(const Situation &other)
	{
		this->player_position = other.player_position;
		this->player_velocity = other.player_velocity;
		this->player_angle = other.player_angle;
		this->ball_position = other.ball_position;
		this->ball_velocity = other.ball_velocity;
		this->target= other.target;
	}

	~Situation()
	{

	}

	bool equals(Situation other)
	{
		bool equal = true;

		equal &= this->player_position.distance(other.get_player_position()) <= 0.1;
		equal &= this->player_velocity.distance(other.get_player_velocity()) <= 0.1;
		equal &= this->player_angle.diff(other.get_player_angle()) <= DEG2RAD(1);
		equal &= this->ball_position.distance(other.get_ball_position()) <= 0.1;
		equal &= this->ball_velocity.distance(other.get_ball_velocity()) <= 0.1;
		equal &= this->target.diff(other.get_target()) <= DEG2RAD(1);

		return equal;
	}

	Vector get_player_position()
	{
		return player_position;
	}

	Vector get_player_velocity()
	{
		return player_velocity;
	}

	Vector get_ball_position()
	{
		return ball_position;
	}

	Vector get_ball_velocity()
	{
		return ball_velocity;
	}

	ANGLE get_player_angle()
	{
		return player_angle;
	}

	ANGLE get_target()
	{
		return target;
	}

	Vector player_position, player_velocity, ball_position, ball_velocity;
	ANGLE player_angle, target;
};

//############################################################################
// CommandProvider
//############################################################################
class CommandProvider
{
public:
	virtual ~CommandProvider();
	virtual void provide(Cmd& cmd, int iteration) = 0;
	virtual int get_max_iterations() = 0;
	virtual bool is_of_type(Cmd& cmd) = 0;
	virtual void copy(Cmd& copy_target, Cmd& copy_source) = 0;
	virtual std::string to_string(Cmd& cmd) = 0;
	std::string val2str(double val)
	{
		std::ostringstream strs;
		strs << val;
		return strs.str();
	}
}; // end of class CommandProvider

//############################################################################
// SituationAssessment
//############################################################################
class SituationAssessment
{
public:
	virtual ~SituationAssessment();
	virtual double assess(Situation situation) = 0;
	virtual double transition(Situation situation, Situation next_situation) = 0;
};

//############################################################################
// CommandAnalysis
//############################################################################
class CommandAnalysis
{
  const static double NEURODRIBBLE2017_BALL_SAFETY_MARGIN;
public:
	CommandAnalysis();
	virtual ~CommandAnalysis();
	double get_best_command(Situation current_situation, Cmd& cmd, CommandProvider& provider, SituationAssessment& sit);
}; // end of class CommandAnalysis

//############################################################################
// DashProvider
//############################################################################
class DashProvider : public CommandProvider
{
public:
	DashProvider(double min_power, double power_step, double max_power, double min_angle, double angle_step, double max_angle);
	virtual ~DashProvider();
	virtual void provide(Cmd& cmd, int iteration);
	virtual int get_max_iterations();
	virtual bool is_of_type(Cmd& cmd);
	virtual void copy(Cmd& copy_target, Cmd& copy_source);
	virtual std::string to_string(Cmd& cmd);
private:
	double min_power, power_step, max_power;
	double min_angle, max_angle, angle_step;
	int power_iterations, angle_iterations;
}; // end of class DashProvider

//############################################################################
// DribbleNetControl
//############################################################################
#define ND2017_PATH "../data/nets_neuro_dribble2017/"
#define ND2017_DEFAULT_FILE_NAME "dribble2017"
#define ND2017_FILE_ENDING ".net"
#define ND2017_DEFAULT_FILE "../data/nets_neuro_dribble2017/dribble2017.net"

class DribbleNetControl
{
public:
	DribbleNetControl(const char* file_name = ND2017_DEFAULT_FILE);
	virtual ~DribbleNetControl();
	void set_target_angle(ANGLE angle);
	double forward(Situation situation);
	void backward(Situation situation, double target_error, bool report = false);
	double calculate_cost(Situation situation, Situation next_situation);
	bool load_net(const char* file_name = ND2017_DEFAULT_FILE);
	bool save_net(const char* file_name = ND2017_DEFAULT_FILE);
	void update_weights();
	void report(Situation situation);
	static std::string path;
	static std::string default_file_name;
	static std::string file_ending;
	static bool collision(Situation situation, double radius);
	static bool collision(Situation situation, Situation next_situation, double radius);
	static bool kickable(Situation situation, double kick_radius);
private:
	double delta_null;
	ANGLE target_angle;
	Net* net;
	void receive_input(Situation situation, bool report = false);
	ANGLE figure_out_how_to_correct(ANGLE angle1, ANGLE angle2);
	int num_layers, num_input, num_output, num_hidden;
	static double ball_size;
};


//############################################################################
// TemporalDifference
//############################################################################
struct TemporalDifference
{
public:
	TemporalDifference()
	{
		this->situation = Situation();
		this->situation = Situation();
		this->value = 0;
	}

	TemporalDifference(Situation situation, Situation next_situation, DribbleNetControl* control)
	{
		double val1, val2, cost;

		this->situation = situation;
		this->next_situation = next_situation;

		val1 = control->forward(situation);
		val2 = control->forward(next_situation);
		cost = control->calculate_cost(situation, next_situation);

		double alpha = 1.0;
		double gamma = 0.7;
		this->value = ( 1 - alpha ) * val1 + alpha * ( cost + gamma * val2 );
	}

	TemporalDifference(const TemporalDifference& other)
	{
		this->situation = other.situation;
		this->next_situation = other.next_situation;
		this->value = other.value;
	}

	~TemporalDifference()
	{

	}

	bool equals(const TemporalDifference& other)
	{
		bool equals = true;

		equals &= this->situation.equals(other.situation);
		equals &= this->next_situation.equals(other.next_situation);

		return equals;
	}

	Situation situation, next_situation;
	double value;
}; // end of struct TemporalDifference

//############################################################################
// Data
//############################################################################
struct Data
{
public:
	Data()
	{
		this->state = TemporalDifference();
		this->used = false;
	}

	Data(TemporalDifference state)
	{
		this->state = TemporalDifference(state);
		this->used = false;
	}

	Data(const Data &other)
	{
		this->state = TemporalDifference(other.state);
		this->used = other.used;
	}

	virtual ~Data()
	{

	}

	TemporalDifference state;
	bool used;
}; // end of struct Data

//############################################################################
// DribbleNetAssessment
//############################################################################
class DribbleNetAssessment: public SituationAssessment
{
public:
	DribbleNetAssessment(DribbleNetControl* control);
	virtual ~DribbleNetAssessment();
	virtual double assess(Situation situation);
	virtual double transition(Situation situation, Situation next_situation);
private:
	DribbleNetControl* net_ctrl;
}; // end of class DribbleNetAssessment

//############################################################################
// TrainingsData
//############################################################################
typedef std::queue<TemporalDifference> StateType;

class TrainingsData
{
public:
	TrainingsData(CommandProvider** providers, int num_com, SituationAssessment* sit, DribbleNetControl* control);
	virtual ~TrainingsData();
	TemporalDifference extract_adjusted_element_at(int index);
	TemporalDifference draw_next();
	TemporalDifference draw_random();
	TemporalDifference draw_fresh_or_random();
	void add(TemporalDifference state);
	TemporalDifference simulate(TemporalDifference state);
	int get_total_size();
	void clean();
private:
	int num_com;
	CommandProvider** providers;
	CommandAnalysis* analysis;
	SituationAssessment* sit;
	StateType fresh_states, old_states, used_states;
	DribbleNetControl* control;
};

//############################################################################
// DribbleNetTrainControl
//############################################################################
class DribbleNetTrainControl
{
public:
	DribbleNetTrainControl(DribbleNetControl* control, CommandProvider** providers, int num_com, SituationAssessment* sit);
	virtual ~DribbleNetTrainControl();
	void add_transition(Situation situation, Situation next_situation);
	bool reasonable_to_add(Situation situation);
	void add_situation(Situation situation);
	void clear_buffer();
	void train();
	void eval_next(int step);
	void eval_all(int step);
	double calculate_error(std::queue<TemporalDifference> queue);
	std::string val2str(double val)
	{
		std::ostringstream strs;
		strs << val;
		return strs.str();
	}
private:
	DribbleNetControl* control;
	int session;
	Situation buffer_state;
	bool buffered, fresh;
	TrainingsData* training_data;
};

//############################################################################
// Analysis
//############################################################################
struct Analysis
{
public:
	Analysis()
	{
		this->size = 0;
		this->max1 = -10000;
		this->max2 = -10000;
		this->min = 10000;
		this->average = 0;
		this->standard_deviation = 0;
	}

	Analysis(const Analysis &other)
	{
		this->size = other.size;
		this->max1 = other.max1;
		this->max2 = other.max2;
		this->min = other.min;
		this->average = other.average;
		this->standard_deviation = other.standard_deviation;
	}

	Analysis(const std::queue<double> &values)
	{
		this->size = values.size();
		this->max1 = -10000;
		this->max2 = -10000;
		this->min = 10000;
		this->average = 0;
		this->standard_deviation = 0;

		std::queue<double> helper = values;

		while( !helper.empty() )
		{
			double c_d = helper.front();
			helper.pop();
			this->average += c_d;

			if( c_d < this->min )
				this->min = c_d;

			if( c_d > this->max1 )
				this->max1 = c_d;

			if( c_d > this->max2 && c_d < this->max1 )
				this->max2 = c_d;
		}

		this->average /= this->size;

		helper = values;

		while( !helper.empty() )
		{
			double c_d = helper.front();
			helper.pop();
			double variance = c_d - this->average;
			variance *= variance;
			this->standard_deviation += variance;
		}

		this->standard_deviation /= this->size;

		this->standard_deviation = sqrt(this->standard_deviation);
	}

	int size;
	double max1, max2, min, average, standard_deviation;
};

//############################################################################
// EvalControl
//############################################################################
class EvalControl
{
public:
	EvalControl(int number_of_episodes_per_sequence);
	virtual ~EvalControl();
	void next_state(Situation situation, double sequence_duration);
	void end_episode();
	void end_sequence(bool was_eval);
	std::string val2str(double val)
	{
		std::ostringstream strs;
		strs << val;
		return strs.str();
	}
private:
	int episode_counter, sequence_counter, time_step_counter, lost_ball, number_of_episodes_per_sequence;
	Situation last_state;
	std::queue<double> distances, x_distances, durations;
	double distance, x_distance;
	bool in_episode, in_sequence;
	void start_episode();
	void start_sequence();
};

//############################################################################
// KickProvider
//############################################################################
class KickProvider: public CommandProvider
{
public:
	KickProvider(double min_power, double power_step, double max_power, double min_angle, double angle_step, double max_angle);
	virtual ~KickProvider();
	virtual void provide(Cmd& cmd, int iteration);
	virtual int get_max_iterations();
	virtual bool is_of_type(Cmd& cmd);
	virtual void copy(Cmd& copy_target, Cmd& copy_source);
	virtual std::string to_string(Cmd& cmd);
private:
	double min_power, power_step, max_power;
	double min_angle, max_angle, angle_step;
	int power_iterations, angle_iterations;
};

//############################################################################
// RandomDribbleAssessment
//############################################################################
class RandomDribbleAssessment : public SituationAssessment
{
public:
	RandomDribbleAssessment();
	virtual ~RandomDribbleAssessment();
	virtual double assess(Situation situation);
	virtual double transition(Situation situation, Situation next_situation);
};


//############################################################################
// TurnProvider
//############################################################################
class TurnProvider : public CommandProvider
{
public:
	TurnProvider(double min, double step, double max);
	virtual ~TurnProvider();
	virtual void provide(Cmd& cmd, int iteration);
	virtual int get_max_iterations();
	virtual bool is_of_type(Cmd& cmd);
	virtual void copy(Cmd& copy_target, Cmd& copy_source);
	virtual std::string to_string(Cmd& cmd);
private:
	double min, step, max;
	int iterations;
};

} // end of namespace NS_ND2017

//############################################################################
// N E U R O D R I B B L E 2 0 1 7
//############################################################################

using namespace NS_ND2017;

class NeuroDribble2017: public BodyBehavior
{
public:
	NeuroDribble2017();
	virtual ~NeuroDribble2017();
	static bool init( char const * conf_file, int argc, char const* const* argv );
	void reset_intention();
	bool get_cmd( Cmd &cmd );
	void set_target(ANGLE target);
	void start_training();
	void pause_training();
	void execute_training();
	void eval_next(int step);
	bool is_safe();
private:
	static bool initialized;
	bool training;
	int num_com;
	Situation last_sit;
	ANGLE target;
	CommandProvider** providers;
	CommandAnalysis analysis;
	SituationAssessment* sit,* random_sit;
	DribbleNetTrainControl* trainer;
	DribbleNetControl* control;
};

#endif /* BS2K_BEHAVIORS_SKILLS_NEURO_DRIBBLE_2017_H_ */
