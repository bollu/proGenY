#pragma once
#include <algorithm>


namespace util{
	const float PI = 3.1415926f;
	const float RAD2DEG = 180.0 / util::PI;
	const float DEG2RAD = 1.0f / RAD2DEG;


	class Angle{
	private:
		Angle(float deg){
			this->angleInDeg = deg;
		}

		float angleInDeg;
	public:
		Angle(){
			this->angleInDeg = 0;
		};

		static Angle Deg(float deg){
			return Angle(deg);
		};

		static Angle Rad(float rad){
			return Angle(rad * RAD2DEG);
		};

		float toDeg(){
			return this->angleInDeg;
		};

		float toRad(){
			return this->angleInDeg * DEG2RAD;
		};
	};
};