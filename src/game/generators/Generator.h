#pragma once
#include <random>

class Generator{
public:
		virtual ~Generator(){};

protected:
	Generator(){};
	
	std::mt19937 generator;

	int _genInt(int min, int max){
		std::uniform_int_distribution<int> dist(min, max);    
		return dist(this->generator);
	};



	float _genFloat(float min, float max){
		std::uniform_real_distribution<float> dist(min, max);     
		return dist(this->generator);
	};

	float  _normGenFloat(float mean, float deviation ){
		std::normal_distribution<float> dist(mean, deviation);     
		return dist(this->generator);
	}

	int  _normGenInt(int mean, int deviation ){
		std::normal_distribution<float> dist(mean, deviation);     
		return std::ceil(dist(this->generator));
	}
};