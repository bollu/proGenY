#pragma once
#include "worldProcess.h"


worldProcess::worldProcess(processMgr &processManager, Settings &settings, eventMgr &eventManager) :
Process("worldProcess"){
		//world = new b2World(settings.getPrimitive<vector2>("worldGravity")->getVal());

	vector2 *gravity = settings.getPrimitive<vector2>(Hash::getHash("gravity"));
	this->stepSize = *settings.getPrimitive<float>(Hash::getHash("stepSize"));

	this->velIterations = *settings.getPrimitive<int>(Hash::getHash("velIterations"));
	this->collisionIterations = *settings.getPrimitive<int>(Hash::getHash("collisionIterations"));

	world = new b2World(*gravity);


	this->dtAccumilator = 0;
	maxAccumilation = this->stepSize * 10;
	
};


float worldProcess::getStepSize(){
	return this->stepSize;
}


float worldProcess::getMaxAccumilation(){
	return this->maxAccumilation;
}

void worldProcess::Start(){
	simulationThread = std::thread(&worldProcess::_Simulate, this);
	simulationThread.detach();

};

/*!pauses the simulation */
void worldProcess::Pause(){
	this->paused = true;
};

	 /*!Resumes the simulation*/
void worldProcess::Resume(){
	this->paused = false;
};

void worldProcess::Update(float dt){

	worldAccess.lock();

	this->dtAccumilator += dt;

	worldAccess.unlock();

};

void worldProcess::Draw(){};


void worldProcess::Shutdown(){

	this->simulationThread.join();
	delete(this->world);
};



#include <thread>
#include <chrono>
void worldProcess::_Simulate(){

	while(1){

	if(!this->worldAccess.try_lock()){
		continue;
	}
	
	
	if(dtAccumilator >= maxAccumilation){
	 	//just step the world once
		dtAccumilator = stepSize;
	}

	while(dtAccumilator >= stepSize){
		world->Step(stepSize, velIterations, collisionIterations);

	 	//now decrement the accumilator by the step size. repeat until you have less accumilated
	 	//than dtAccum step size
		dtAccumilator -= stepSize;
	}

	this->worldAccess.unlock();

	}

};


vector2 worldProcess::getGravity(){
	worldAccess.lock();
	return vector2::cast(this->world->GetGravity());
};

b2Body *worldProcess::createBody(const b2BodyDef* def){
	return this->world->CreateBody(def);
};

void worldProcess::destroyBody(b2Body* body){
	this->world->DestroyBody(body);
};

 void worldProcess::setContactListener(b2ContactListener* listener){
 	this->world->SetContactListener(listener);
 };
