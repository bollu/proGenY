#pragma once
#include <algorithm>
#include "vector.h"
#include "../IO/strHelper.h"

#define PRINTANGLE(angle) IO::infoLog<<#angle<<" "<angle.debugStr();

namespace util{
	const double PI = 3.141592653589793238462;
	const float PIBy2 = PI / 2.0;
	const double TwoPI = 2.0 * PI;
	const double RAD2DEG = 180.0 / util::PI;
	const double DEG2RAD = util::PI / 180.0;


	template<typename T>
	static int sgn(T num){
		return 
		num > 0 ? 1 :
		num < 0 ? -1  :
		0;
	}

	
	template <typename T>
	T clamp(const T& n, const T& lower, const T& upper) {
  	return std::max(lower, std::min(n, upper));
	}

	template <typename T>
	static T lerp(const T &begin, const T &end, float t){
		return begin * (1.0 - t)  +  end * t;
	}

	/*
	static float randFloat(const float range = 1000){
		return (rand() % (int)range) / (float)(range - 1.0);
	}
	*/
	
	/*!An agnostic representation of Angle
	can be used to represent angles in both degrees and
	radians. Internally, data is stored in radians
	for faster trig computations. 
	*/
	class Angle{
	private:
		Angle(float rad){
			this->setRad(rad);
		}

		float angleInRad;
	public:
		Angle(){
			this->angleInRad = 0;
		};

		/*!convert a vector to an Angle in polar coordinates
		creates an Angle that holds the angle formed by the vector
		with the x-axis in anti-clockwise direction
	
		@param [in] vec the vector to be converted to an Angle
		\return the specified Angle
		*/
		Angle(vector2 vec){
			this->setRad(vec.toAngle());
		}

		/*!creates an Angle specified in Degrees

		@param [in] deg the value of the Angle in degrees
		\return the Angle in degrees
		*/
		static Angle Deg(float deg){
			return Angle(deg * DEG2RAD);
		};

		/*!creates an Angle specified in Radians

		@param [in] rad the value of the Angle in radians
		\return the Angle in radians
		*/
		static Angle Rad(float rad){
			return Angle(rad);
		};

		/*!returns a string representing the Angle held in degrees
		is useful for debugging
		*/
		std::string debugStr(){
			return "dAngle: " + util::strHelper::toStr(RAD2DEG * this->angleInRad);
		}

		/*!returns the sine of the angle*/
		float sin(){
			return ::sin(this->angleInRad);
		}

		/*!returns the cosine of the angle*/
		float cos(){
			return ::cos(this->angleInRad);
		};

		/*!returns the tan of the angle*/
		float tan(){
			return ::tan(this->angleInRad);
		}

		/*returns the value of the angle in degrees
		this is always [0, 360)*/
		float toDeg(){
			return this->angleInRad * RAD2DEG;
		};

		/*returns the value of the angle in radians
		this is always [0, 2 PI)*/
		float toRad(){
			return this->angleInRad;
		};

		/*!returns a unit vector that makes the corresponding angle with x-axis*/
		vector2 toVector(){
			return vector2(this->cos(), this->sin());
		}

		/*!set the Angle in radians
		@param angleInRad - the angle in Radians. It need not be within [0, 2 PI).
		If the angle is outside the domain, it is normalized internally
		*/
		void setRad(float angleInRad){
			this->angleInRad = angleInRad;

			while(this->angleInRad >= TwoPI){
				this->angleInRad -= TwoPI;
			}
			while(this->angleInRad < 0){
				this->angleInRad += TwoPI;
			}

		}

		/*!returns the position vector of the point
		 cut by the angle on a circle of given radius
		
		basically, returns the Cartesian coordinate(x,y) of the point (r, theta) in
		polar coordinates 

		@param [in] radius the radius of the circle 
		 */
		vector2 polarProjection(float radius){
			return vector2(this->cos() * radius, this->sin() * radius);
		}

		Angle operator + (const Angle & other) const{
			return Angle(this->angleInRad + other.angleInRad);
		}

		Angle operator - (const Angle & other) const{
			return Angle(this->angleInRad - other.angleInRad);
		}

		Angle operator * (const float multiplier) const{
			return Angle(this->angleInRad * multiplier);

		}

		Angle operator / (const float multiplier) const{
			return Angle(this->angleInRad / multiplier);
		}

		void operator += (const Angle &other){
			this->setRad(this->angleInRad + other.angleInRad);
		}

		void operator -= (const Angle &other){
			this->setRad(this->angleInRad - other.angleInRad);
		}

		void  operator *= (const float multiplier){
			this->setRad(this->angleInRad * multiplier);
		}


		void operator /= (const float multiplier){
			this->setRad(this->angleInRad / multiplier);
		}


	};
};
