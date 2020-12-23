#include "shadow.h"
#include "log_macros.h"
#include "tools.h"

Shadow::Shadow()
{
	prediction_depth = 1;
	set_goal(0);
}

Shadow::~Shadow()
{

}

	//Setter

void Shadow::set_point(const Vector point)
{
	this->point = Vector(point);

	schatten = Triangle2d(point, GOAL_RIGHT_CORNER, GOAL_LEFT_CORNER);
}




void Shadow::set_prediction_depth(const int depth)
{
	prediction_depth = depth;
}

void Shadow::set_goalie(const Vector goalie)
{
	this->goalie = Vector(goalie);
}

void Shadow::set_goal(const int goal)
{
	if(goal==0)
	{
		GOAL_LEFT_CORNER=HIS_GOAL_LEFT_CORNER;
		GOAL_RIGHT_CORNER=HIS_GOAL_RIGHT_CORNER;
		GOAL_CENTER=HIS_GOAL_CENTER;
	}
	if(goal==1)
	{
		GOAL_LEFT_CORNER=MY_GOAL_LEFT_CORNER;
		GOAL_RIGHT_CORNER=MY_GOAL_RIGHT_CORNER;
		GOAL_CENTER=MY_GOAL_CENTER;
	}
}

	//Getter

Vector Shadow::get_left_open_point()
{
	Vector goalieleft=this->goalie;
	goalieleft.setY(goalieleft.getY()-get_goalie_intercept_distance(point.distance(goalie)));

	left_open_point = goalieleft;

	//log
	LOG_POL(0,<<_2D<<VL2D(point,GOAL_LEFT_CORNER,"3333ff"));

	return left_open_point;
}

Vector Shadow::get_right_open_point()
{
	Vector goalieright=this->goalie; 
	goalieright.setY(goalieright.getY()+get_goalie_intercept_distance(point.distance(goalie)));
	
	right_open_point = goalieright;

	//log
	LOG_POL(0,<<_2D<<VL2D(point,GOAL_RIGHT_CORNER,"3333ff"));

	return right_open_point;
}

void Shadow::get_open_points(Vector &left, Vector &right)
{
	get_right_open_point();
	get_left_open_point();
	left = Vector(left_open_point);
	right = Vector(right_open_point);
}

double Shadow::get_cover()
{
	double cover=(100-((get_right_open_area() + get_left_open_area()) / (GOAL_RIGHT_CORNER.distance(GOAL_LEFT_CORNER)))*100);

	LOG_POL(0,<<_2D<<VSTRING2D(GOAL_CENTER,cover,"ff0000"));

	return cover;
}

double Shadow::get_cover_center()
{
	double cover=(100-((get_right_open_area() + get_left_open_area()) / (GOAL_RIGHT_CORNER.distance(GOAL_LEFT_CORNER)))*100);
	//TODO: anpassen
	double a = get_right_open_area() - get_left_open_area();
	if(a<0.0) a=-a;
	cover = cover - (a*1);
	if(cover<0.0) cover=0.0;

	LOG_POL(0,<<_2D<<VSTRING2D(GOAL_CENTER,cover,"ff0000"));

	return cover;
}

double Shadow::get_right_open_area()
{
	get_right_open_point();
	Vector rightsp;
	Vector rightst=right_open_point;//Steigung
	rightst.subFromX( point.getX() );
	rightst.subFromY( point.getY() );
	rightsp=Tools::intersection_point(point,rightst,GOAL_RIGHT_CORNER,Vector(0.1,1000000));
	if(rightsp.getY()<GOAL_RIGHT_CORNER.getY())
	{
		rightsp.setY(GOAL_RIGHT_CORNER.getY());
	}
	if(rightsp.getY()>GOAL_LEFT_CORNER.getY()){rightsp.setY(GOAL_LEFT_CORNER.getY());}
	if(point.getX()>goalie.getX()&&GOAL_CENTER.getX()>0){rightsp=GOAL_LEFT_CORNER;}
	if(point.getX()<goalie.getX()&&GOAL_CENTER.getX()<0){rightsp=GOAL_LEFT_CORNER;}
	rightsp.setX(GOAL_LEFT_CORNER.getX());

	//log
	LOG_POL(0,<<_2D<<VL2D(rightsp,right_open_point,"3333ff"));
	LOG_POL(0,<<_2D<<VL2D(goalie,right_open_point,"e7fe2e"));
	LOG_POL(0,<<_2D<<VSTRING2D(GOAL_LEFT_CORNER,rightsp.distance(GOAL_LEFT_CORNER),"ff0000"));
	return rightsp.distance(GOAL_LEFT_CORNER);
}

double Shadow::get_left_open_area()
{
	get_left_open_point();

	Vector leftsp; 
	Vector leftst=left_open_point;//Steigung
	leftst.subFromX( point.getX() );
	leftst.subFromY( point.getY() );
	leftsp=Tools::intersection_point(point,leftst,GOAL_RIGHT_CORNER,Vector(0.1,1000000));
	if(leftsp.getY()>GOAL_LEFT_CORNER.getY()){leftsp.setY(GOAL_LEFT_CORNER.getY());}
	if(leftsp.getY()<GOAL_RIGHT_CORNER.getY()){leftsp.setY(GOAL_RIGHT_CORNER.getY());}
	if(point.getX()>goalie.getX()&&GOAL_CENTER.getX()>0){leftsp=GOAL_LEFT_CORNER;}
	if(point.getX()<goalie.getX()&&GOAL_CENTER.getX()<0){leftsp=GOAL_LEFT_CORNER;}
	leftsp.setX(GOAL_LEFT_CORNER.getX());
	//log
	LOG_POL(0,<<_2D<<VL2D(leftsp,left_open_point,"3333ff"));
	LOG_POL(0,<<_2D<<VL2D(goalie,left_open_point,"e7fe2e"));
	LOG_POL(0,<<_2D<<VSTRING2D(GOAL_RIGHT_CORNER,leftsp.distance(GOAL_RIGHT_CORNER),"ff0000"));
	return leftsp.distance(GOAL_RIGHT_CORNER);
}

void Shadow::get_open_areas(double &left, double &right)
{
	left = get_left_open_area();
	right = get_right_open_area();
}

double Shadow::get_goalie_intercept_distance(int steps)
{
	double a;
	steps=steps/3;//max ball speed
	a=0.3824*steps; //side dash
	a=a + 0.75;//catch

LOG_POL(0,<<_2D<<STRING2D(0,0,a,"ff0000"));

	return a;
}
