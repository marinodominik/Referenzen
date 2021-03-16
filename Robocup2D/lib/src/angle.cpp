/*
 * Copyright (c) 1999 - 2000, Artur Merke
 *
 * This file is part of FrameView2d.
 *
 * FrameView2d is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2, or (at your option)
 * any later version.
 *
 * FrameView2d is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with FrameView2d; see the file COPYING.  If not, write to
 * the Free Software Foundation, 675 Mass Ave, Cambridge, MA 02139, USA.
 */

#include "angle.h"

const double ANGLE::TwoPi = 2*M_PI;

ANGLE::ANGLE()
{
    set_value(0.0);
}

ANGLE::ANGLE(const ANGLE &otherAng)
{
    set_value(otherAng.get_value());
}

ANGLE::ANGLE(double newVal)
{
    set_value(newVal);
}

ANGLE::ANGLE(double xVal, double yVal)
{
    set_value(atan2(yVal, xVal));
}

ANGLE::ANGLE(const Vector &vec)
{
    set_value(atan2(vec.getY(), vec.getX()));
}

ANGLE::~ANGLE()
{

}

void ANGLE::set_value(double newVal)
{
    if(newVal >= 0 && newVal < TwoPi)
    {
        val = newVal;
        return;
    }
    val = fmod(newVal, TwoPi);
    if(val < 0.0)
    {
        val += TwoPi;
    }
}

double ANGLE::get_value() const
{
    return get_value_0_p2PI();
}

double ANGLE::get_value_0_p2PI() const
{
    return val;
}

double ANGLE::get_value_mPI_pPI() const
{
    return val <= M_PI ? val : val - TwoPi;
}

double ANGLE::diff(const ANGLE &otherAng) const
{
    double res, myVal = get_value(), otherVal = otherAng.get_value();
    if(myVal > otherVal)
        res = myVal - otherVal;
    else
        res = otherVal - myVal;
    if(res > M_PI)
        res = TwoPi - res;
    return res;
}

ANGLE& ANGLE::operator=(const ANGLE &otherAng)
{
    set_value(otherAng.get_value());
    return *this;
}

void ANGLE::operator+=(const ANGLE &otherAng)
{
    set_value(get_value() + otherAng.get_value());
}

void ANGLE::operator-=(const ANGLE &otherAng)
{
    set_value(get_value() - otherAng.get_value());
}

void ANGLE::operator*=(const double &multiplier)
{
    set_value(get_value() * multiplier);
}

void ANGLE::operator/=(const double &divisor)
{
    set_value(get_value() / divisor);
}

ANGLE ANGLE::operator+() const
{
    return ANGLE( get_value() );
}

ANGLE ANGLE::operator-() const
{
    return ANGLE( TwoPi - get_value() );
}

ANGLE ANGLE::operator+(const ANGLE &otherAng) const
{
    return ANGLE( get_value() + otherAng.get_value() );
}

ANGLE ANGLE::operator-(const ANGLE &otherAng) const
{
    return ANGLE( get_value() - otherAng.get_value() );
}

ANGLE ANGLE::operator*(const double &multiplier) const
{
    return ANGLE( get_value() * multiplier );
}

ANGLE operator*(const double &multiplier, const ANGLE &ang)
{
    return ANGLE( ang.get_value() * multiplier );
}

ANGLE ANGLE::operator/(const double &divisor) const
{
    return ANGLE( get_value() / divisor );
}

ANGLE operator/(const double &divisor, const ANGLE &ang)
{
    return ANGLE( ang.get_value() / divisor );
}

bool ANGLE::operator==(const ANGLE &otherAng) const
{
	return this->get_value() == otherAng.get_value();
}

bool ANGLE::operator!=(const ANGLE &otherAng) const
{
	return this->get_value() != otherAng.get_value();
}

bool ANGLE::operator<(const ANGLE &otherAng) const
{
	return this->get_value() < otherAng.get_value();
}

bool ANGLE::operator>(const ANGLE &otherAng) const
{
	return this->get_value() > otherAng.get_value();
}

bool ANGLE::operator<=(const ANGLE &otherAng) const
{
	return this->get_value() <= otherAng.get_value();
}

bool ANGLE::operator>=(const ANGLE &otherAng) const
{
	return this->get_value() >= otherAng.get_value();
}

std::istream& operator>>(std::istream& inStream, ANGLE &ang)
{
    return inStream >> ang.val;
}

std::ostream& operator<<(std::ostream& outStream, const ANGLE &ang)
{
    return outStream << ang.get_value();
}

double sin(const ANGLE &ang)
{
    return sin(ang.get_value());
}

double cos(const ANGLE &ang)
{
    return cos(ang.get_value());
}
