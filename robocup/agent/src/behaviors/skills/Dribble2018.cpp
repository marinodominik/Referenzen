//
// Created by dominik on 03.06.18.
//

#include "Dribble2018.h"
#include "math.h"


bool Dribble2018::initialized = false;
bool cmd_set = false;
bool isVector = false;
Vector targetP;
bool first = false;
ANGLE targetD;
Vector lastPosition;
bool two2kick = false;


//Konstruktor und Dekonstruktor
Dribble2018::Dribble2018(){
    ivpGo2PosBehavior = new GoToPos2016();
    ivpOneStepKickBehavior = new OneStepKick();
    ivpNeuroKickBehavior = new NeuroKickWrapper();
    targetP = MY_GOAL_CENTER;
    //nextPositionPlayer = calcNextPosition(WSinfo::me->pos, WSinfo::me->vel);
    //nextPostionBall = calcNextPostion(WSinfo::ball->pos, WSinfo::ball->vel);
}

Dribble2018::~Dribble2018(){

}


void Dribble2018::reset(){

}

bool Dribble2018::is_safe() {
    if (WSinfo::ball->pos.distance(WSinfo::me->pos)<=1.0){
        return true;
    }else{
        first = true;
        return false;
    }

}


bool Dribble2018::is_safe_to_kick(){
    Vector ball = WSinfo::ball->pos;
    Vector ballNext = WSinfo::ball->pos+WSinfo::ball->vel*ServerOptions::ball_decay;
    Vector me = WSinfo::me->pos;
    Vector meNext = WSinfo::me->pos+WSinfo::me->vel;

    if(ballNext.distance(meNext)<1 /*&& bedingung für geschwindigkeit? vllt TODO*/){
        return true;
    }else{
        return false;
    }

}


bool Dribble2018::get_cmd(Cmd &cmd){
    Vector ballNext = nextBallPosition(WSinfo::ball->pos, WSinfo::ball->vel);
    Vector meNext = nextPlayerPosition(WSinfo::me->pos, WSinfo::me->vel);

    if(first){
        Vector naechstePosition(WSinfo::ball->pos.getX() + WSinfo::ball->vel.getX()*ServerOptions::ball_decay,
                                WSinfo::ball->pos.getY() + WSinfo::ball->vel.getY()*ServerOptions::ball_decay);
        //berechne die stärke
        //double staerke = 100.0/3.0;
        //double durch = (WSinfo::ball->vel.getX()+WSinfo::ball->vel.getY()); ///2.0;
        //staerke = staerke*durch;
        // ANGLE angleBall = Tools::my_angle_to(WSinfo::ball->vel);

        //cmd.cmd_body.set_kick(durch, angleBall.get_value_mPI_pPI()+PI);
        //cmd_set=cmd.cmd_body.check_cmd();
        int a = 1;

        ANGLE angForPos;
        angForPos.set_value(DEG2RAD(25));

        //std::cout<<" Angle a:"<<angForPos.get_value_mPI_pPI()<<std::endl;
        //a.set_value(DEG2RAD(-90));
        Vector newBallPos = createVector(25, 0.8); //setXY(me.getX()+0.8 *cos(target2Me-angForPos), me.getY()+0.8*sin(target2Me-angForPos));

        ivpNeuroKickBehavior->kick_to_pos_with_final_vel(0.1, WSinfo::me->pos);

        cmd_set = ivpNeuroKickBehavior->get_cmd(cmd);
        first=false;
        return cmd_set;
    }


    if (is_safe_to_kick() ){ // && me.distance(ball)>0.66
        if(two2kick){

            //std::cout << "stoppeball;" <<std::endl;
            stopBall(cmd);
            two2kick= false;



        }else if(Tools::my_angle_to(targetP)>DEG2RAD(10) && ballNext.distance(meNext) <1 ){
           cmd.cmd_body.set_turn(Tools::my_angle_to(targetP).get_value_mPI_pPI());


        }else if(checkBlindSpot()){
            //schieße den ball zu einer vorgegebenen Position, sodass
            //der ball nicht mehr im blind spot liegt
            //überprüfung welcher der position links oder rechts näher ist und dann wird zu dem Vektor geschossen
            Vector links = createVector(90, 0.4);
            Vector rechts = createVector(-90, 0.4);
            two2kick = true;
            std::cout <<"blindspot" << std::endl;
            if(checkNearSideForDribble(links, rechts)){
                ivpOneStepKickBehavior->kick_in_dir_with_initial_vel(0.3, links);
                //ivpOneStepKickBehavior->kick_to_pos_with_initial_vel(0.3, newBallPos)
                cmd_set = ivpOneStepKickBehavior->get_cmd(cmd);
            }else{
                //kick rechts
                ivpOneStepKickBehavior->kick_to_pos_with_final_vel(0.3, rechts);
                //ivpOneStepKickBehavior->kick_to_pos_with_initial_vel(0.3, newBallPos)
                cmd_set = ivpOneStepKickBehavior->get_cmd(cmd);
          }

        }else if(check2stepKick()){
            Vector links = createVector(45, 0.4);
            Vector rechts = createVector(-45, 0.4);
            two2kick = false;


            if(checkNearSideForDribble(links, rechts)){
                ivpOneStepKickBehavior->kick_in_dir_with_initial_vel(0.2, links);
                //ivpOneStepKickBehavior->kick_to_pos_with_initial_vel(0.3, newBallPos)
                cmd_set = ivpOneStepKickBehavior->get_cmd(cmd);
            }else{
                //kick rechts
                ivpOneStepKickBehavior->kick_to_pos_with_final_vel(0.2, rechts);
                //ivpOneStepKickBehavior->kick_to_pos_with_initial_vel(0.3, newBallPos)
                cmd_set = ivpOneStepKickBehavior->get_cmd(cmd);
            }

        } else if(checkBallInTheWay()){
            Vector links = createVector(45, 0.3);
            Vector rechts = createVector(-45, 0.3);
            two2kick = true;
            if(checkNearSideForDribble(links, rechts)){

                ivpOneStepKickBehavior->kick_in_dir_with_initial_vel(0.2, links);
                //ivpOneStepKickBehavior->kick_to_pos_with_initial_vel(0.3, newBallPos)
                cmd_set = ivpOneStepKickBehavior->get_cmd(cmd);
            }else{
                //kick rechts
                ivpOneStepKickBehavior->kick_to_pos_with_final_vel(0.2, rechts);
                //ivpOneStepKickBehavior->kick_to_pos_with_initial_vel(0.3, newBallPos)
                cmd_set = ivpOneStepKickBehavior->get_cmd(cmd);
            }


        }else if(checkNotSaveForDribbleBecause2FarAway()){
            Vector links = createVector(90, 0.3);
            Vector rechts = createVector(-90, 0.3);
            two2kick = true;
            if(checkNearSideForDribble(links, rechts)){
                ivpOneStepKickBehavior->kick_in_dir_with_initial_vel(0.2, links);
                //ivpOneStepKickBehavior->kick_to_pos_with_initial_vel(0.3, newBallPos)
                cmd_set = ivpOneStepKickBehavior->get_cmd(cmd);
            }else{
                //kick rechts
                ivpOneStepKickBehavior->kick_to_pos_with_final_vel(0.2, rechts);
                //ivpOneStepKickBehavior->kick_to_pos_with_initial_vel(0.3, newBallPos)
                cmd_set = ivpOneStepKickBehavior->get_cmd(cmd);
            }

        } else if(ballNext.distance(meNext)>0.9 == false && WSinfo::ball->pos.distance(targetP) > WSinfo::me->pos.distance(targetP) && WSinfo::ball->pos.distance(WSinfo::me->pos) >=0.7) {
            Vector links = createVector(10, 0.7);
            Vector rechts = createVector(-10, 0.7);
            std::cout << "kick now" << std::endl;
            if(checkNearSideForDribble(links, rechts)){
                ivpOneStepKickBehavior->kick_in_dir_with_initial_vel(0.50, links);
                //ivpOneStepKickBehavior->kick_to_pos_with_initial_vel(0.3, newBallPos)
                cmd_set = ivpOneStepKickBehavior->get_cmd(cmd);
            }else{
                //kick rechts
                ivpOneStepKickBehavior->kick_to_pos_with_final_vel(0.50, rechts);
                //ivpOneStepKickBehavior->kick_to_pos_with_initial_vel(0.3, newBallPos)
                cmd_set = ivpOneStepKickBehavior->get_cmd(cmd);
            }


        }else if(checkPerfektBallPosition()) {
            //std::cout << "perfekt line" << std::endl;
            Vector links = createVector(10, 0.7);
            Vector rechts = createVector(-10, 0.7);

            if (checkNearSideForDribble(links, rechts)) {
                ivpOneStepKickBehavior->kick_in_dir_with_initial_vel(0.5, links);
                //ivpOneStepKickBehavior->kick_to_pos_with_initial_vel(0.3, newBallPos)
                cmd_set = ivpOneStepKickBehavior->get_cmd(cmd);
            } else {
                //kick rechts
                ivpOneStepKickBehavior->kick_to_pos_with_final_vel(0.5, rechts);
                //ivpOneStepKickBehavior->kick_to_pos_with_initial_vel(0.3, newBallPos)
                cmd_set = ivpOneStepKickBehavior->get_cmd(cmd);
            }
        }else if(Tools::my_angle_to(targetP) < 10 && ballNext.distance(targetP) < WSinfo::ball->pos.distance(targetP) && ballNext.distance(WSinfo::me->pos) <1){
            ivpGo2PosBehavior->setAngleDiff(DEG2RAD(2.5)); // Einstellung um Spieler besser auszurichten :
            ivpGo2PosBehavior->set_target(targetP, 0.7);
            cmd_set = ivpGo2PosBehavior->get_cmd(cmd);

        }
}





/*
        //Ball behind Player
        if(ball.distance(behindPlayer)<ball.distance(behindRight) && ball.distance(behindPlayer)<ball.distance(behindeLeft)) {
            angForPos.set_value(DEG2RAD(75));
            std::cout << " Angle a:" << angForPos.get_value_mPI_pPI() << std::endl;
            //a.set_value(DEG2RAD(-90));
            newBallPos.setXY(me.getX() + 0.6 * cos(target2Me - angForPos), me.getY() + 0.6 * sin(target2Me - angForPos));

            ivpOneStepKickBehavior->kick_to_pos_with_final_vel(0.67, newBallPos);
            //ivpOneStepKickBehavior->kick_to_pos_with_initial_vel(0.3, newBallPos)
            cmd_set = ivpOneStepKickBehavior->get_cmd(cmd);
        }

            //BallRightSide
        else if (ball.distance(behindRight)<ball.distance(behindeLeft)){
            angForPos.set_value(DEG2RAD(25));

            std::cout<<" Angle a:"<<angForPos.get_value_mPI_pPI()<<std::endl;
            //a.set_value(DEG2RAD(-90));
            newBallPos.setXY(me.getX()+1.0 *cos(target2Me-angForPos), me.getY()+1.0*sin(target2Me-angForPos));

            if (fabs(WSinfo::me->vel.getY())+fabs(WSinfo::me->vel.getX())<0.3){
                ivpOneStepKickBehavior->kick_to_pos_with_final_vel(0.67,newBallPos);
            }else{
                ivpOneStepKickBehavior->kick_to_pos_with_final_vel(0.67,newBallPos);
            }

            //ivpOneStepKickBehavior->kick_to_pos_with_initial_vel(0.3, newBallPos);
            cmd_set = ivpOneStepKickBehavior->get_cmd(cmd);

            //BallLeftSide
        }else {
            angForPos.set_value(DEG2RAD(-25));

            std::cout<<" Angle a:"<<angForPos.get_value_mPI_pPI()<<std::endl;
            newBallPos.setXY(me.getX()+1.0 *cos(target2Me-angForPos), me.getY()+1.0*sin(target2Me-angForPos));

            if (fabs(WSinfo::me->vel.getY())+fabs(WSinfo::me->vel.getX())<0.3){
                ivpOneStepKickBehavior->kick_to_pos_with_final_vel(0.67,newBallPos);
            }else{
                ivpOneStepKickBehavior->kick_to_pos_with_final_vel(0.67,newBallPos);
            }
            //ivpOneStepKickBehavior->kick_to_pos_with_initial_vel(0.7, newBallPos);
            cmd_set = ivpOneStepKickBehavior->get_cmd(cmd);
        }



    }else{

        std::cout<<" ausrichten "<<targetP.getY()<<std::endl;
        ivpGo2PosBehavior->setAngleDiff(DEG2RAD(2.5)); // Einstellung um Spieler besser auszurichten :
        ivpGo2PosBehavior->set_target(targetP, 0.7);
        cmd_set = ivpGo2PosBehavior->get_cmd(cmd);

    }*/
    return cmd_set;
}

//Neu
bool Dribble2018::checkBlindSpot(){
    Vector ball= WSinfo::ball->pos;
    Vector me = WSinfo::me->pos;


    Vector blindSpot = createVector((-180), 0.2);

    if(ball.distance(blindSpot) <= 0.2 || ball.distance(me) <=0.1){
        return true;
    }else{
        return false;
    }

}

bool Dribble2018::checkBallInTheWay() {
    Vector ball= WSinfo::ball->pos;
    Vector me = WSinfo::me->pos;

    Vector beforePlayer = createVector(0, 0.2);

    if(ball.distance(beforePlayer) <= 0.2 || ball.distance(me) <=0.1){
        return true;
    }else return false;
}

bool Dribble2018::checkNearSideForDribble(Vector links, Vector rechts) {
    Vector ball= WSinfo::ball->pos;
    if(ball.distance(links) <= ball.distance(rechts)){
        return true;
    }else return false;
}


bool Dribble2018::check2stepKick() {
    Vector ball= WSinfo::ball->pos;

    Vector twoKickArea1 = createVector((180), 0.6);
    Vector twoKickArea2 = createVector(-180, 1.0);

    if(ball.distance(twoKickArea1) <=0.2 || ball.distance(twoKickArea2) <=0.2){
        return true;
    }else{
        return false;
    }
}

bool Dribble2018::checkPerfektBallPosition() {
    Vector ball= WSinfo::ball->pos;

    Vector linksFeld = createVector((135), 0.6);
    Vector rechtsFeld = createVector((-135), 0.6);

    if(ball.distance(linksFeld) <= 0.25 || ball.distance(rechtsFeld) <= 0.25){
        return true;
    }else return false;
}

bool Dribble2018::checkNotSaveForDribbleBecause2FarAway(){
    Vector ball= WSinfo::ball->pos;

    Vector linksFeld = createVector((90), 1);
    Vector rechtsFeld = createVector((-90), 1);


    if(ball.distance(linksFeld) <= 0.25 || ball.distance(rechtsFeld) <= 0.25){
        return true;
    }else return false;


}

//neu ende


void Dribble2018::set_target(ANGLE targetDir){
    targetD = targetDir;
    if(WSinfo::me->pos.distance(lastPosition)>3.0){
        first=true;
    }else{
        first=false;
    }
    PPlayer me = WSinfo::me;
    //me.ang=targetDir;

    ANGLE angForPos;
    angForPos.set_value(DEG2RAD(90));
    Vector pos;
    pos.setXY(me->pos.getX()+50.0 *cos(angForPos-targetDir - DEG2RAD(90)), me->pos.getY()+50.0*sin(angForPos-targetDir - DEG2RAD(90)));


    lastPosition=WSinfo::me->pos;



    targetP=pos;



}

void Dribble2018::set_target(Vector targetPos){
    if(WSinfo::me->pos.distance(lastPosition)>3.0){
        first=true;
    }else{
        first=false;
    }
    lastPosition=WSinfo::me->pos;
    ::targetP = targetPos;
    //isVector = true;
}


Vector Dribble2018::nextBallPosition(Vector pos, Vector vel) {
    Vector naechstePosition(pos.getX() + vel.getX()*ServerOptions::ball_decay,
                            pos.getY() + vel.getY()*ServerOptions::ball_decay);
    return naechstePosition;
}

Vector Dribble2018::nextPlayerPosition(Vector pos, Vector vel) {
    Vector naechstePosition(pos.getX() + vel.getX(), pos.getY() + vel.getY());
    return naechstePosition;
}




ANGLE Dribble2018::kickAngle(Vector target){
    ANGLE test = WSinfo::ball->pos.ANGLE_to(target);
    return test;
}

bool Dribble2018::ballOutOfControll() {

    Vector nextBall = nextBallPosition(WSinfo::ball->pos, WSinfo::ball->vel);
    Vector nextPlayer = nextPlayerPosition(WSinfo::me->pos, WSinfo::ball->vel);

    if(nextBall.distance(nextPlayer)>=0.95){
        return true;
    }else return false;

}



bool Dribble2018::init(char const * conf_file, int argc, char const* const* argv){
    if(initialized) return true;

    bool res = OneOrTwoStepKick::init(conf_file,argc,argv) &&
               OneStepKick::init(conf_file,argc,argv);
    if(!res) exit(1);

    //success_sqrdist = SQUARE(WSinfo::me->kick_radius - SAFETY_MARGIN);

    initialized = true;
    INFO_OUT << "\nDribbleStraight behavior initialized.";
    return initialized;
}



Vector Dribble2018::createVector(double winkel, double distanz) {
    /*Diese Funktion erstellt den Vektor in mithilfe von angle2target funktion
     * Das bedeutet,die erste koordinate ist dazu da, um den winkel in richtung target +-einen neuen
     * Winkel der übergeben wird zu berechnen um dann den Vektor zum Beispiel blindspot zu berechnen*/

    Vector me = WSinfo::me->pos;
    ANGLE target2Me = angle2Target(targetP);
    float alpha = bogen2Grad(target2Me.get_value());
    float x, y;

    if((alpha+winkel)==90.0){

        x = me.getX() + (0 * distanz);
        y = me.getY() + (1 * distanz);

    }else if((alpha+winkel)==180.0){

        x = me.getX() + (-1 * distanz);
        y = me.getY() + (0 * distanz);
    }else if((alpha+winkel) == 270.0){

        x = me.getX() + (0 * distanz);
        y = me.getY() + (-1 * distanz);
    } else if((alpha+winkel)==360.0 || alpha+winkel==0.0) {
        x = me.getX() + (1 * distanz);
        y = me.getY() + (0 * distanz);
    }else {
        x = me.getX() + (cos((alpha + winkel) * M_PI / 180) * distanz);
        y = me.getY() + (sin((alpha + winkel) * M_PI / 180) * distanz);
    }

    Vector zielPunkt(x, y);
   // std::cout << "zielpunkt: " <<zielPunkt << std::endl;
    return zielPunkt;
}

ANGLE Dribble2018::angle2Target(Vector target) {
    //immer ausgehend von der x-Achse
    Vector me = WSinfo::me->pos;
    Vector targetDirectionPlayer = target-me;

    ANGLE target2Me = targetDirectionPlayer.ARG();
    //std::cout << "target 2 me: " << target2Me << std::endl;
    /*Hier wird der Winkel von  der x-Achse bis zum Target zuückgegen,
     * Wichtig, das ist nicht der winkel zum Turn, sodass der Spieler zum target schaut*/
    return target2Me;
}


float Dribble2018::bogen2Grad(double winkel) {
    float alpha = (360*winkel)/(2*3.141592654);
    return alpha;
}



void Dribble2018::stopBall(Cmd &cmd){

    //POL("stopBall: was called");
    Vector mePos = WSinfo::me->pos;
    Vector nextMePos = mePos + WSinfo::me->vel*ServerOptions::player_decay;
    Vector ballVel = WSinfo::ball->vel;
    Vector ballPos = WSinfo::ball->pos;
    Vector nextBallPos = ballPos+ballVel*ServerOptions::ball_decay;
    const MyState state=get_cur_state();

    if(fabs(ballVel.getX()) <= 0.02 && fabs(ballVel.getY()) <= 0.02){
        //POL("stopBall: a stop is not needed");
    }else{
        //POL("stopBall: Trying to stop the ball");
        const Vector kick_vec=ballPos-nextBallPos;

        double kick_decay = get_kick_decay(state);
        double power = kick_vec.norm()/(ServerOptions::kick_power_rate * kick_decay);

        ANGLE kickDir = kick_vec.ARG();
        kickDir -= WSinfo::me->ang;

        cmd.cmd_body.set_kick(power, kickDir);
        //cmd_set=cmd.cmd_body.check_cmd();
    }

}



// Aus OneStepKick probz an den Programmierer//

void Dribble2018::get_ws_state(MyState &state) {
    state.my_pos = WSinfo::me->pos;
    state.my_vel = WSinfo::me->vel;
    state.my_angle = WSinfo::me->ang;
    state.ball_pos = WSinfo::ball->pos;
    state.ball_vel = WSinfo::ball->vel;

    PlayerSet pset= WSinfo::valid_opponents;
    pset.keep_and_sort_closest_players_to_point(1, state.ball_pos);
    if ( pset.num ){
        state.op_pos= pset[0]->pos;
        state.op_bodydir =pset[0]->ang;
        state.op_bodydir_age = pset[0]->age_ang;
        state.op = pset[0];
    }
    else{
        state.op_pos= Vector(1000,1000); // outside pitch
        state.op_bodydir = ANGLE(0);
        state.op_bodydir_age = 1000;
        state.op = 0;
    }

}

MyState Dribble2018::get_cur_state() {
    MyState cur_state;
    get_ws_state(cur_state);
    return cur_state;
}


double Dribble2018::get_kick_decay(const MyState &state) {
    Vector tmp = state.ball_pos - state.my_pos;
    tmp.rotate(-state.my_angle.get_value()); //normalize to own dir
    tmp.setY( fabs(tmp.getY()) );
    double decay=1.0-tmp.arg()/(4.0*PI) -
                 (tmp.norm()- ServerOptions::ball_size - WSinfo::me->radius)/
                 (4.0* (WSinfo::me->kick_radius-WSinfo::me->radius-ServerOptions::ball_size));
    return decay;
}



