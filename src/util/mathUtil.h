#pragma once
#include <algorithm>
#include "../core/vector.h"
#include "strHelper.h"

#define PRINTANGLE(angle) util::msgLog(std::string(#angle) + angle.debugStr());

namespace util{
	const float PI = 3.1415926f;
	const float PIBy2 = PI / 2.0;
	const float TwoPI = 2.0 * PI;
	const float RAD2DEG = 180.0 / util::PI;
	const float DEG2RAD = 1.0f / RAD2DEG;


	class Angle{
	private:
		Angle(float rad){
			this->setRad(this->angleInRad);
		}

		float angleInRad;
	public:
		Angle(){
			this->angleInRad = 0;
		};

		Angle(vector2 vec){
			this->setRad(vec.toAngle());
		}

		static Angle Deg(float deg){
			return Angle(deg * DEG2RAD);
		};

		static Angle Rad(float rad){
			return Angle(rad);
		};

		std::string debugStr(){
			return "dAngle: " + util::strHelper::toStr(RAD2DEG * this->angleInRad);
		}

		float sin(){
			return ::sin(this->angleInRad);
		}

		float cos(){
			return ::cos(this->angleInRad);
		};

		float tan(){
			return ::tan(this->angleInRad);
		}

		float toDeg(){
			return this->angleInRad * RAD2DEG;
		};

		float toRad(){
			return this->angleInRad;
		};

		void setRad(float angleInRad){
			this->angleInRad = angleInRad;

			while(this->angleInRad >= TwoPI){
				this->angleInRad -= TwoPI;
			}
			while(this->angleInRad < 0){
				this->angleInRad += TwoPI;
			}

		}

		//returns the vector cut by this angle on a circle of given radius
		vector2 polarProjection(float radius){
			return vector2(this->cos() * radius, this->sin() * radius);
		}

		Angle operator + (const Angle & other){
			return Angle(this->angleInRad + other.angleInRad);
		}

		Angle operator - (const Angle & other){
			return Angle(this->angleInRad - other.angleInRad);
		}

		Angle operator * (const float multiplier){
			return Angle(this->angleInRad * multiplier);

		}

		Angle operator / (const float multiplier){
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