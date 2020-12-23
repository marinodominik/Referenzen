#ifndef _DRIBBLE_BETWEEN_H
#define _DRIBBLE_BETWEEN_H

// includes
#include "../../basics/Cmd.h"
#include "dribble_around.h"
#include "dribble_between_actions.h"
#include "positioning.h"
#include "ws_memory.h"
#include "log_macros.h"


#define MIN_STAMINA_DRIBBLE ((0.05+ServerOptions::effort_dec_thr)*ServerOptions::stamina_max) 
                            //TG09: alt: 1600

	

/**
 * DribbleBetween
 *
 * Uses DribbleAround to find a way between opponents on the way to goal
 * @author Hannes Schulz <mail@hannes-schulz.de>
 * @version 0.1
 *
 */

class DribbleBetween:public BodyBehavior
{
	private:
		static DribbleBetween* myInstance;              ///< Singleton
		DribbleBetween();                               ///< Private constructor
		DribbleBetween(const DribbleBetween&);          ///< Copying also prohibited!
		DribbleBetween operator=(const DribbleBetween&);///< Copying still prohibited!

		/// the DribbleAround behavior we're using
		DribbleAround* dribbleAround;

		Cmd cachedCmd;
		int cachedCmdTime;

		enum Mode{
			DB_NO_OPP,
			DB_ONE_OPP,
			DB_TWO_OPP,
			DB_TOO_MANY_OPP};
		Mode mode;

		/// get the relevant opponents
		void setRelevantOpponents();

		/// select point to dribble to
		Vector getTargetPos();

		/// if I'm attacked go straight, do not turn
		bool iAmAttacked();

		/// judge how good a dribble dir is
		/// @param dribbleAngle relative to body
		float getValueForDribbleDir(float dribbleAngle);

		/// target to dribble to
		Vector dribbleTo;
		/// dribbleAround target
		Vector dribbleAroundTarget;
		float dribbleAroundTargetValue;

		/// target last time
		Vector lastTarget;

		/// the opponents we're playing against
		PlayerSet opps;

		/// safes result of is_dribble_between_possible
		bool isDribblingSafe;
		bool dribblingInsecure;

		/// last time when we dribbled
		int lastDribbleTime;

		/// how often we dribblet to current target
		int targetTimeCounter;

		/// The statistics are saved in here
#define DRIBBLE_STATS_BUFFER_LENGTH 10
		class Stats {
			private:
				struct DribbleState {
					PPlayer closestOpp;
					float xDistToClosestOpp;
					DribbleAction actionTaken;
					bool actionSucceeded;
					int time;
					bool didDribble;
				};
				// The ring buffer of dribbleStates
				DribbleState dribbleStates[DRIBBLE_STATS_BUFFER_LENGTH];
				// The current Position in the ring buffer
				int bufferPos;

				DribbleBetween* db;

			public:
				// update from viewpoint of is_dribble_between_possible
				void updateTestDribbling();

				// update from viewpoint of get_cmd
				void updateGetCmd();

				// get some statistics output
				void getStatistics();

				// constructor
				Stats();

		} stats;

	public:
		/// get the only instance of this class (Singleton pattern!)
		static DribbleBetween* getInstance();

		/// Try to dribble.
		/// @return true   if successful
		/// @return false  otherwise
		bool get_cmd(Cmd& cmd);

		/// Set a point to dribble to
		void set_target(const Vector&);
		
		/// Get the point to dribble to
		Vector get_target();

		/// say whether dribbling is generally possible
		bool is_dribble_safe(bool = true);
		bool is_dribble_insecure();

		/// say whether neck request was set
		bool is_neck_req_set();
		ANGLE get_neck_req();

		virtual ~DribbleBetween();

		static bool init(char const * conf_file, int argc, char const* const* argv) {
			cout<<"\n DribbleBetween: init"<<endl;
			return (DribbleAround::init(conf_file,argc,argv));
		}
};

#endif /* _DRIBBLE_BETWEEN_H */
