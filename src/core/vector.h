#pragma once
#include <math.h>

#ifndef PRINTVECTOR2
	#define PRINTVECTOR2(vec) std::cout<<"\n\t"<<#vec<<" X = "<<vec.x<<" Y = "<<vec.y<<std::endl;
#endif

//HACK!--------------------------------------
#include "../include/Box2D/Common/b2Math.h"
#include "../include/SFML/System/Vector2.hpp"
//class b2Vec2;



class vector2{
public:
	float x, y;
 
	vector2(){ this->x = this->y = 0;};
	~vector2(){};

	template<typename T>
	static vector2 cast(const T &otherVec){
		return vector2(otherVec.x, otherVec.y);
	}

	template<typename T>
	T cast(){
		return T(this->x, this->y);
	}

	/*inline*/ vector2(float x, float y){
		this->x = x; this->y = y;
	};

	inline vector2(const vector2& other){
		this->x = other.x;
		this->y = other.y;
	}
	//creates a normalized vector which forms an angle **in radians** with the x-axis
	static vector2 createVectorFromAngle(float angle){
		return vector2(cos(angle),  sin(angle));
	};

	//get the normalized vector
	vector2 Normalize(){	
		float length = this->Length(); 
		length  = (length == 0 ? 1 : length);
		return (vector2(this->x / length, this->y / length));
	};


	//convert the vector to an angle 
	float toAngle() {
		return atan2(this->y, this->x);
	}

	//clamp this vector between -vec and +vec
	vector2 clamp(vector2 minVec, vector2 maxVec){
		vector2 clampedVec; clampedVec.x = x; clampedVec.y = y;
		if(clampedVec.x < minVec.x) clampedVec.x = minVec.x;
		if(clampedVec.x > maxVec.x) clampedVec.x = maxVec.x;

		if(clampedVec.y < minVec.y) clampedVec.y = minVec.y;
		if(clampedVec.y > maxVec.y) clampedVec.y = maxVec.y;

		return clampedVec;
	};

	//get the length of the vector
	inline float Length(){ return (sqrt(x * x  +  y * y)); };
	/// Get the length squared. For performance, use this instead of vector2::Length (if possible).
	inline float LengthSquared() const{ return (x * x + y * y); };

	//---------operator overloads-------------------------------------------

	//negate this vector
	inline vector2 operator -(){	return vector2(-x, -y); };
	// Add a vector to this vector.
	inline void operator += (const vector2& v){ x += v.x; y += v.y; };
	// Subtract a vector from this vector.
	inline void operator -= (const vector2& v){ x -= v.x; y -= v.y; };
	// Multiply this vector by a scalar.
	inline void operator *= (float a){ x *= a; y *= a; };

	inline vector2 operator + (const vector2& a) const { return vector2(x + a.x, y + a.y); };
	inline vector2 operator - (const vector2& a) const { return vector2(x - a.x, y - a.y); };
	inline vector2 operator / (const vector2& a) const { return vector2(x / a.x, y / a.y); };
	inline vector2 operator * (float scale)		 const { return vector2(x * scale, y * scale); };

	inline bool operator > (const vector2& a) const  { return (this->x > a.x && this->y > a.y); };
	inline bool operator < (const vector2& a) const  { return (this->x < a.x && this->y < a.y); };
	inline bool operator >= (const vector2& a) const { return (this->x >= a.x && this->y >= a.y); };
	inline bool operator <= (const vector2& a) const { return (this->x <= a.x && this->y <= a.y); };
	inline bool operator == (const vector2& a) const { return (this->x == a.x && this->y == a.y); };
	//templatized function to typecast an object to another one by using the constructor -_^
	 //template<class otherPhyVect> inline operator otherPhyVect(){ return otherPhyVect(this->x, this->y); }

	inline operator b2Vec2(){ return b2Vec2(this->x, this->y); }
	
	template <typename T>
	inline operator sf::Vector2<T>(){ return sf::Vector2<T>(this->x, this->y); }

};

template<typename TYPE>
inline vector2 operator * (const TYPE s, const vector2& a) { return vector2(a.x * s  , a.y * s);    };	

template<typename TYPE>
inline vector2 operator * (const vector2& a, const TYPE s) { return vector2(a.x * s  , a.y * s);    };	

inline bool    operator == (const vector2&a , vector2& b) { return (a.x == a.y) && (b.x == b.y);  };

//------------------------------------------------------------------------------------------------
class vector3{
public:
	float x, y, z;

	vector3(){ this->x = this->y = this->z = 0;};
	vector3(float x, float y, float z){
		this->x = x; this->y = y;
		this->z = z;
	};

	vector3(vector2 vec2, float z){
		this->x = vec2.x;
		this->y = vec2.y;
		this->z = z;
	}
	//get the normalized vector
	vector3 Normalize(){	
		float length = this->Length(); 
		length  = length == 0 ? 1 : length;
		return (vector3(this->x / length, this->y / length, this->z / length));
	};

	//convert this vector to radians
	/*float rad(){
		vector2 v = vector2(this->x, this->y);
		v.Normalize();
		return atan2(v.y,v.x);
	};*/


	//get the length of the vector
	inline float Length(){ return (sqrt(x * x  +  y * y + z * z)); };
	/// Get the length squared. For performance, use this instead of vector2::Length (if possible).
	inline float LengthSquared() const{ return (x * x + y * y + z * z); };

	//---------operator overloads-------------------------------------------

	//negate this vector
	inline vector3 operator -(){	return vector3(-x, -y, -z); };
	// Add a vector3 to this vector.
	inline void operator += (const vector3& v){ x += v.x; y += v.y; z += v.z; };
	// Add a vector2 to this vector.
	inline void operator += (const vector2& v){ x += v.x; y += v.y; };

	// Subtract a vector3 from this vector.
	inline void operator -= (const vector3& v){ x -= v.x; y -= v.y; z -= v.z;};
	// Subtract a vector2 from this vector.
	inline void operator -= (const vector2& v){ x -= v.x; y -= v.y;};

	// Multiply this vector by a scalar.
	inline void operator *= (float a){ x *= a; y *= a; z *= a; };

	inline vector3 operator + (const vector3& a) { return vector3(x + a.x, y + a.y, z + a.z); };
	inline vector2 operator + (const vector2& a) { return vector2(x + a.x, y + a.y); };

	inline vector3 operator - (const vector3& a) { return vector3(x - a.x, y - a.y, z - a.z); };
	inline vector2 operator - (const vector2& a) { return vector2(x - a.x, y - a.y); };


	inline bool operator > (const vector3& a) { return (this->x > a.x && this->y > a.y && this->z > a.z); };
	inline bool operator < (const vector3& a) { return (this->x < a.x && this->y < a.y && this->z < a.z); };
	inline bool operator >= (const vector3& a) { return (this->x >= a.x && this->y >= a.y && this->z >= a.z); };
	inline bool operator <= (const vector3& a) { return (this->x <= a.x && this->y <= a.y  && this->z <= a.z); };

	//templatized function to typecast an object to another one by using the constructor -_^
	
	//template<class otherPhyVect> inline operator otherPhyVect(){ return otherPhyVect(this->x, this->y, this->z); }
	template<class otherPhyVect2> inline operator otherPhyVect2(){ return otherPhyVect2(this->x, this->y); }
	
	//conversion from vector3 to vector2
	inline operator vector2(){ return vector2(this->x, this->y); }

};
inline vector3 operator * (float s, const vector3& a)		   { return vector3(a.x * s  , a.y * s, a.z * s);    };	
inline vector3 operator * (const vector3& a, float s)		   { return vector3(a.x * s  , a.y * s, a.z * s);	 };	
inline bool    operator == (const vector3&a , vector3& b)	   { return (a.x == a.y) && (b.x == b.y) && (a.z == b.z);  };




//template<class Type = float>
//class vector2{
//public:
//	float x, y;
//	
//	vector2(){ this->x = this->y = 0; };
//	vector2(float x, float y){
//		this->x = x; this->y = y;
//	};
//	get the normalized vector
//	vector2 Normalize(){	
//		float length = this->Length(); 
//		length  = length == 0 ? 1 : length;
//		return (vector2(this->x / length, this->y / length));
//	};
//
//	convert this vector to radians
//	float rad(){
//		vector2 v = vector2(this->x, this->y);
//		v.Normalize();
//		return atan2(v.y,v.x);
//	};
//
//	clamp this vector between -vec and +vec
//	vector2 clamp(vector2 minVec, vector2 maxVec){
//		vector2 clampedVec; clampedVec.x = x; clampedVec.y = y;
//		if(clampedVec.x < minVec.x) clampedVec.x = minVec.x;
//		if(clampedVec.x > maxVec.x) clampedVec.x = maxVec.x;
//
//		if(clampedVec.y < minVec.y) clampedVec.y = minVec.y;
//		if(clampedVec.y > maxVec.y) clampedVec.y = maxVec.y;
//
//		return clampedVec;
//	};
//
//	get the length of the vector
//	inline float Length(){ return (sqrt(x * x  +  y * y)); };
//	 Get the length squared. For performance, use this instead of vector2::Length (if possible).
//	inline float LengthSquared() const{ return (x * x + y * y); };
//
//	---------operator overloads-------------------------------------------
//
//	negate this vector
//	inline vector2 operator -(){	return vector2(-x, -y); };
//	 Add a vector to this vector.
//	inline void operator += (const vector2& v){ x += v.x; y += v.y; };
//	 Subtract a vector from this vector.
//	inline void operator -= (const vector2& v){ x -= v.x; y -= v.y; };
//	 Multiply this vector by a scalar.
//	inline void operator *= (float a){ x *= a; y *= a; };
//
//	inline vector2 operator + (const vector2& a) { return vector2(x + a.x, y + a.y); };
//	inline vector2 operator - (const vector2& a) { return vector2(x - a.x, y - a.y);};
//	inline vector2 operator / (const vector2& a) { return vector2(x / a.x, y / a.y); };
//	
//	
//	inline bool    operator == (const vector2&a)	   { return (a.x == this->x) && (a.y == this->y);  };
//
//	templatized function to typecast an object to another one by using the constructor -_^
//	template<class otherPhyVect> inline operator otherPhyVect(){ return otherPhyVect(this->x, this->y); }
//
//};
//
//inline vector2 operator * (const float s, vector2&a)		   { return vector2(a.x * s  , a.y * s);    };	
//inline vector2 operator * (vector2&a, const float s)		   { return vector2(a.x * s  , a.y * s);    };	