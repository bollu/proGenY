#pragma once
#include "distanceSensor.h"
#include "../World/worldProcess.h"
using namespace AI;


distanceSensor::distanceSensor(Object *owner, b2World *world, std::string targetClass) :
	Sensor(owner, "distanceSensor"){

};


void distanceSensor::Sense(){
	
};
