#ifndef __POINTTO_BMP_H__
#define __POINTTO_BMP_H__

#include "../basics/Cmd.h"
#include "tools.h"
#include "ws_info.h"
#include "base_bm.h"
#include "log_macros.h"

enum POINTTO_MODE { 
      current_stamina, 
      stamina_capacity 
};  

class Pointto10 : public PointToBehavior {
            
	private:
		static double pointtoDist;
        bool pointto_stamina_capacity(Cmd & cmd);
        bool pointto_current_stamina(Cmd & cmd);
        
        static Angle getPointtoFromCurrStaminaRel2Me(double stamina, ANGLE neck_ang);
        static Angle getPointtoFromCurrStaminaCapacityRel2Me(double stamina, ANGLE neck_ang);
        // JTS 10: pointto related fields and vars
        static int pt_current_stamina_discretization [];
        static int pt_curr_stamina_discretization_N;
        static Angle pt_curr_stamina_ang_inc;
        static Angle pt_curr_stamina_start_ang;
  
        static Angle pt_stamina_cap_start_ang;
        static Angle pt_stamina_cap_ang_inc;
        static int pt_stamina_cap_discretization_step;
        static int pt_stamina_cap_discretization_max;
	public:
        static POINTTO_MODE mode;
        static bool init(char const * conf_file, int argc, char const* const* argv) {
            return true;
		}
        
        static Angle getPointtoFromCurrStamina(double stamina);
        static int getCurrStaminaFromPointto(ANGLE pointto);
        
        static Angle getPointtoFromCurrStaminaCapacity(double stamina);
        static int getCurrStaminaCapacityFromPointto(ANGLE pointto);
        static double get_stamina_capacity_from_base_encoding(char c);
		
        Pointto10();
		virtual ~Pointto10();
		bool get_cmd(Cmd & cmd); 
};

#endif
