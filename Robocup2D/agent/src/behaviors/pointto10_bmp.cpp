#include "pointto10_bmp.h"

#define DRAW(XXX)  LOG_POL(0,<<_2D<<XXX)
#define DRAW_LINE(P,Q,C) DRAW(L2D((P).x,(P).y,(Q).x,(Q).y,#C))

// the mode variable decides which mode we use to encode stamina
POINTTO_MODE Pointto10::mode = stamina_capacity;

double Pointto10::pointtoDist = 1000000.;
int Pointto10::pt_current_stamina_discretization[] = { 1250, 1300, 1350, 1400, 1500, 1600, 1800, 2000, 2250, 2500, 3000, 3500, 4000  };
int Pointto10::pt_curr_stamina_discretization_N = 13;
Angle Pointto10::pt_curr_stamina_ang_inc(DEG2RAD(22.5));
Angle Pointto10::pt_curr_stamina_start_ang(DEG2RAD(45.f));
int Pointto10::pt_stamina_cap_discretization_step = 400;
Angle Pointto10::pt_stamina_cap_ang_inc(DEG2RAD(1.f));
Angle Pointto10::pt_stamina_cap_start_ang(5.0*Pointto10::pt_stamina_cap_ang_inc);
int Pointto10::pt_stamina_cap_discretization_max = int(fabs( (2.f * PI -   2.f * Pointto10::pt_stamina_cap_start_ang)
                                                         / Pointto10::pt_stamina_cap_ang_inc 
                                                         * Pointto10::pt_stamina_cap_discretization_step ));

Pointto10::Pointto10()
{

}

Pointto10::~Pointto10()
{

}

/* static methods */
Angle Pointto10::getPointtoFromCurrStamina(double stamina) {
  Angle result;
  int best = 0;
  for ( int i = 1; i <  Pointto10::pt_curr_stamina_discretization_N; i++)
  {
  	if(    fabs( Pointto10::pt_current_stamina_discretization[i] - stamina )
  	    <= fabs( Pointto10::pt_current_stamina_discretization[best] - stamina ) 
  	  )
  	{
  		best = i;
  	}
  }
  LOG_POL(10, "Pointto10 getPointtoFromStamina closest stamina: "  << Pointto10::pt_current_stamina_discretization[best] << " index: " << best);
  result = ANGLE(Pointto10::pt_curr_stamina_start_ang + best  * Pointto10::pt_curr_stamina_ang_inc).get_value_mPI_pPI();
  LOG_POL(10, "Pointto10 getPointtoFromStamina pointing to: " << RAD2DEG(result));
  return result;
}

Angle Pointto10::getPointtoFromCurrStaminaRel2Me(double stamina, ANGLE neck_ang) {
  ANGLE result(Pointto10::getPointtoFromCurrStamina(stamina));
  LOG_POL(10, "Pointto10 getPointtoFromStamina provided neck_ang: "  << RAD2DEG(neck_ang.get_value_mPI_pPI()) 
  			<< " computed pointto_ang: " << RAD2DEG(result.get_value_mPI_pPI()));
  result -= neck_ang;
  LOG_POL(10, "Pointto10 getPointtoFromStamina result was: "  << RAD2DEG(result.get_value_mPI_pPI()));
  return result.get_value_mPI_pPI();
}

double Pointto10::get_stamina_capacity_from_base_encoding(char c) {
    char uchar_base = c - 'A';  
    int base = (int) uchar_base;
    LOG_POL(1, << "Pointto10 int base is " << base);
    return (base * 1.f) * Pointto10::pt_stamina_cap_discretization_step;
}

int Pointto10::getCurrStaminaFromPointto(ANGLE pointto) {
	Angle angValue = pointto.get_value_0_p2PI();
	int staminaIndex = int( floor(fabs( (angValue - Pointto10::pt_curr_stamina_start_ang) 
                                        / Pointto10::pt_curr_stamina_ang_inc) + 0.5) );
	LOG_POL(10, "Pointto10 getStaminaFromPointto  pointto was " << RAD2DEG(angValue))
	if ( staminaIndex < 0 )
	{
		LOG_POL(0, "Pointto10 getStaminaFromPointto ERROR: no valid index " << staminaIndex);
		staminaIndex = 0;
	}
	if ( staminaIndex >= Pointto10::pt_curr_stamina_discretization_N )
	{
		LOG_POL(0, "Pointto10 getStaminaFromPointto ERROR: no valid index " << staminaIndex);
		staminaIndex = Pointto10::pt_curr_stamina_discretization_N - 1;
	}
	return Pointto10::pt_current_stamina_discretization[staminaIndex];
}

Angle Pointto10::getPointtoFromCurrStaminaCapacity(double stamina) {
  int base = 0;
  Angle result;
  if ( stamina > Pointto10::pt_stamina_cap_discretization_max ) 
  {
    result = - pt_stamina_cap_start_ang;
  }
  else
  {
    base = floor(stamina / pt_stamina_cap_discretization_step);
    result = ANGLE(Pointto10::pt_stamina_cap_start_ang + 
                   base * Pointto10::pt_stamina_cap_ang_inc).get_value_mPI_pPI();;
  }
  //std::cout << "Pointto10 getPointtoFromStamina pointing to: " << RAD2DEG(result) << " stamina_cap: " << stamina <<  " base: " << base << std::endl;
  return result;
}

Angle Pointto10::getPointtoFromCurrStaminaCapacityRel2Me(double stamina, ANGLE neck_ang) {
  ANGLE result(Pointto10::getPointtoFromCurrStaminaCapacity(stamina));
  result -= neck_ang;
  LOG_POL(10, "Pointto10 getPointtoFromStaminaCapacityRel2Me result was: "  << RAD2DEG(result.get_value_mPI_pPI()));
  return result.get_value_mPI_pPI();
}

int Pointto10::getCurrStaminaCapacityFromPointto(ANGLE pointto) {
   Angle angValue = pointto.get_value_0_p2PI();
   int base = int(fabs((angValue - Pointto10::pt_stamina_cap_start_ang) / Pointto10::pt_stamina_cap_ang_inc));
   int result = base * pt_stamina_cap_discretization_step;
   return result;
}

bool Pointto10::get_cmd(Cmd & cmd) 
{
    if ( Pointto10::mode == current_stamina )
        return pointto_current_stamina(cmd);
	else if ( Pointto10::mode == stamina_capacity )
        return pointto_stamina_capacity(cmd);
    else
        return false;
}

bool Pointto10::pointto_stamina_capacity(Cmd & cmd)
{
    double actStamina = WSinfo::me->stamina_capacity + WSinfo::me->stamina;
	if ( cmd.cmd_point.is_cmd_set())
	{ // cmd is already set
		LOG_POL(10,"Pointto10 : WARNING cmd_point is already set");
		return false;
	}
    cmd.cmd_point.set_pointto( Pointto10::pointtoDist, //0.0);
				               Pointto10::getPointtoFromCurrStaminaCapacityRel2Me(actStamina, WSinfo::me->neck_ang));
    return true;
}


bool Pointto10::pointto_current_stamina(Cmd & cmd)
{
    double actStamina = WSinfo::me->stamina;
	//ANGLE neck_ang = WSinfo::me->neck_ang;
	if ( cmd.cmd_point.is_cmd_set())
	{ // cmd is already set
		LOG_POL(10,"Pointto10 : WARNING cmd_point is already set");
		return false;
	}
	if (   cmd.cmd_body.is_cmd_set() 
		&& cmd.cmd_body.get_type() == Cmd_Body::TYPE_DASH
	   )
	{ // I plan to dash -> therefore communicate stamina after dash
		double dash_cost;
		cmd.cmd_body.get_dash(dash_cost);
		if ( dash_cost < 0. )
			dash_cost = fabs(dash_cost) * 2.;
		actStamina -= dash_cost;
		actStamina += WSinfo::me->stamina_inc_max; 
		if ( actStamina < 0. )
			actStamina = 0.;
		if ( actStamina > ServerOptions::stamina_max )
			actStamina = ServerOptions::stamina_max;
		LOG_POL(10,"Pointto10 : planning to dash using next turn stamina " << actStamina);
	}
	else
	{
		LOG_POL(10,"Pointto10 : no dash next turn using stamina " << actStamina);
	}
	// otherwise code stamina information
	LOG_POL(10,"Pointto10  neck ang is:  " << RAD2DEG(WSinfo::me->neck_ang.get_value_mPI_pPI()))
	//Vector tmp;
	//tmp.init_polar(2., WSinfo::me->neck_ang);
	//DRAW_LINE(WSinfo::me->pos, WSinfo::me->pos + tmp, "#0000FF");
	cmd.cmd_point.set_pointto( Pointto10::pointtoDist, //0.0);
				               Pointto10::getPointtoFromCurrStaminaRel2Me(actStamina, WSinfo::me->neck_ang));
	return true;
}
