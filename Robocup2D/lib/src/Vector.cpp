#include "Vector.h"
#include <iomanip>

#define min(x, y) (x > y ? y : x)
#define max(x, y) (x > y ? x : y)

#define EPS 1.0e-10

double Vector::getX() const
{
	return this->x;
}

void Vector::setX( double newX )
{
	this->x = newX;//roundf( newX * 100 ) / 100;
}

Vector& Vector::addToX( double summand )
{
	setX( getX() + summand );
	return *this;
}

Vector& Vector::subFromX( double subtrahend )
{
	setX( getX() - subtrahend );
	return *this;
}

Vector& Vector::mulXby( double multiplier )
{
	setX( getX() * multiplier );
	return *this;
}

Vector& Vector::divXby( double divisor )
{
	setX( getX() / divisor );
	return *this;
}

double Vector::getY() const
{
	return this->y;
}

void Vector::setY( double newY )
{
	this->y = newY;//roundf( newY * 100 ) / 100;
}

Vector& Vector::addToY( double summand )
{
	setY( getY() + summand );
	return *this;
}

Vector& Vector::subFromY( double subtrahend )
{
	setY( getY() - subtrahend );
	return *this;
}

Vector& Vector::mulYby( double multiplier )
{
	setY( getY() * multiplier );
	return *this;
}

Vector& Vector::divYby( double divisor )
{
	setY( getY() / divisor );
	return *this;
}

void Vector::setXY( double newX, double newY )
{
	setX( newX );
	setY( newY );
}

void Vector::clone( const Vector &origin )
{
	this->setXY( origin.getX(), origin.getY() );
}

Vector::Vector()
{
	setXY( 0, 0 );
}

Vector::Vector(const Vector &origin)
{
	clone( origin );
}

Vector::Vector(const double valX, const double valY){
	setXY( valX, valY );
}

Vector::Vector(const ANGLE &ang){
	setXY( cos(ang), sin(ang) );
}

Vector& Vector::operator=(Vector vec){
	this->clone(vec);
	return *this;
}

Vector Vector::operator+(const Vector &otherVec) const {
	return Vector((this->getX() + otherVec.getX()), (this->getY() + otherVec.getY()));
}

Vector Vector::operator-(const Vector &otherVec) const {
	return Vector((this->getX() - otherVec.getX()), (this->getY() - otherVec.getY()));
}

Vector Vector::operator*(const double &multiplier) const {
	return Vector((this->getX() * multiplier), (this->getY() * multiplier));
}

Vector operator*(const double &multiplier, const Vector &vec){
	return Vector((multiplier * vec.getX()), (multiplier * vec.getY()));
}

Vector Vector::operator/(const double &divisor) const {
	return Vector((this->getX() / divisor), (this->getY() / divisor));
}

Vector Vector::operator-() const {
	return Vector(-this->getX(), -this->getY());
}

void Vector::operator+=(Vector vec){
	this->setX( this->getX() + vec.getX() );
	this->setY( this->getY() + vec.getY() );
}

void Vector::operator-=(Vector vec){
	this->setX( this->getX() - vec.getX() );
	this->setY( this->getY() - vec.getY() );
}

void Vector::operator*=(double val){
	this->setX( this->getX() * val );
	this->setY( this->getY() * val );
}

void Vector::operator/=(double val){
	this->setX( this->getX() / val );
	this->setY( this->getY() / val );
}

bool Vector::operator<=( const Vector &otherVec ) const {
	return this->sqr_norm() <= otherVec.sqr_norm();
}

bool Vector::operator<(  const Vector &otherVec ) const {
	return this->sqr_norm() <  otherVec.sqr_norm();
}

bool Vector::operator>=( const Vector &otherVec ) const {
	return this->sqr_norm() >= otherVec.sqr_norm();
}

bool Vector::operator>( const Vector &otherVec ) const {
	return this->sqr_norm() >  otherVec.sqr_norm();
}

bool Vector::isBetween( Vector vec1, Vector vec2 ) const {
	return (( vec1.getX() <= this->getX() && this->getX() <= vec2.getX() ) || ( vec1.getX() >= this->getX() && this->getX() >= vec2.getX() ))
	    && (( vec1.getY() <= this->getY() && this->getY() <= vec2.getY() ) || ( vec1.getY() >= this->getY() && this->getY() >= vec2.getY() ));
}

double Vector::distance(const Vector &orig) const{
	return (*this - orig).norm();
}

double Vector::sqr_distance(const Vector &orig) const{
	return (*this - orig).sqr_norm();
}

double Vector::angle(const Vector &dir) const {
	double ang = dir.arg() - arg();
	return normalize_angle(ang);
}

ANGLE Vector::ANGLE_to(const Vector &dir) const {
	return dir.ARG() - ARG();
}

Vector& Vector::rotate(const double &ang){
	double c = cos(ang);
	double s = sin(ang);

	/* Rotation Matrix (counterclockwise)
		 [ cos	-sin]
		 [ sin	 cos] */
	double old_x = getX();
	setX( (c * getX()) - (s * getY()) );
	setY( (s * old_x)  + (c * getY()) );

	return *this;
}

Vector& Vector::ROTATE(const ANGLE &ang){
	rotate(ang.get_value());

	return *this;
}

double Vector::norm() const {
	return sqrt( ( getX() * getX() ) + ( getY() * getY() ) );
}

double Vector::sqr_norm() const {
	return ( ( getX() * getX() ) + ( getY() * getY() ) );
}

Vector& Vector::normalize(double l){
	*this *= (l / max(norm(), EPS));
	return *this;
}

double Vector::arg() const {
	if (0.0 == getX() && 0.0 == getY()) return 0.0;
	double tmp = atan2(getY(), getX());
	if (tmp < 0.0)
		return tmp + 2 * M_PI;
	else
		return tmp;
}

ANGLE Vector::ARG() const {
	return ANGLE(getX(), getY());
}

double Vector::angle() {
	return arg();
}

double Vector::normalize_angle(const double &a) const {
	double tmp = fmod(a, 2 * M_PI);
	if (tmp < 0.0){
		tmp += 2 * M_PI;
	}
	return tmp;
}

double Vector::dot_product(const Vector &orig) const{
	return (this->getX() * orig.getX()) + (this->getY() * orig.getY());
}

Vector& Vector::init_polar(const double &val, const double &ang){
	setX( val * cos(ang) );
	setY( val * sin(ang) );
	return *this;
}

Vector& Vector::init_polar(const double &val, const ANGLE &ang){
	init_polar(val, ang.get_value());
	return *this;
}

std::ostream& operator<<( std::ostream &outStream, const Vector &vector )
{
    return outStream << "( " << vector.getX() << " / " << vector.getY() << " )";
//    return outStream << "( " << std::setw(6) << std::setprecision(2) << std::fixed << vector.getX() << " / " << std::setw(6) << std::setprecision(2) << std::fixed << vector.getY() << " )";
}
