
#include "AirMoveProcessor.h"

void airMoveProcessor::_onObjectAdd(Object *obj){};

void airMoveProcessor::_Process(Object *obj, float dt){
	
};


//airMoveData--------------------------------------------------------------
void airMoveData::setDir(vector2 dir){
	this->dir = dir;
};
void airMoveData::setSpeed(float speed){
	this->speed = speed;
};
