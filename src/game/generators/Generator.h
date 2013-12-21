#pragma once
#include <random>

static int genInt(std::mt19937 &generator, int min, int max){
	std::uniform_int_distribution<int> dist(min, max);    
	return dist(generator);
};


static float genFloat(std::mt19937 &generator, float min, float max){
	std::uniform_real_distribution<float> dist(min, max);     
	return dist(generator);
};

/*
static float normGenFloat(std::mt19937 &generator, float mean, float deviation ){
	std::normal_distribution<float> dist(mean, deviation);     
	return dist(generator);
}
*/

static int  normGenInt(std::mt19937 &generator, int mean, int deviation ){
	std::normal_distribution<float> dist(mean, deviation);     
	return std::ceil(dist(generator));
}