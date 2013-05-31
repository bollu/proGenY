#pragma once
#include "../../include/Box2D/Box2D.h"
#include "Process.h"
#include "processMgr.h"



class worldProcess : public Process{
private:
	b2World *world;

	bool paused = false;

	float stepSize;

	//the accumilator used to accumilate partial step sizes, so that once every sepSize time has
	//passed, we can step the world 
	float dtAccumilator;

	//if the amount accumilated passes maxAccumilation, the accumilator "lapses". this is to make sure
	//that it dosen't get sucked into a "spiral of doom" where it updates more as it's accumilated a lot,
	//and this causes even more accumilation, and so on and so forth
	float maxAccumilation;

	float velIterations, collisionIterations;
public:
	worldProcess(processMgr &processManager, Settings &settings, eventMgr &eventManager) :
	 Process("worldProcess"){
		//world = new b2World(settings.getProp<vector2>("worldGravity")->getVal());
		world = new b2World(vector2(0, -9.8));

		this->stepSize = 1.0 / 60.0;

		this->dtAccumilator = 0;

		maxAccumilation = this->stepSize * 10;

		velIterations = collisionIterations = 5;

	};

	 void Pause(){
	 	this->paused = true;
	 };

	 void Resume(){
	 	this->paused = false;
	 };

	 void Update(float dt){

	 	dtAccumilator += dt;

	 	if(this->paused){
	 		return;
	 	}

	 	if(dtAccumilator >= maxAccumilation){
	 		//just step thw world once
	 		dtAccumilator = stepSize;
	 	}

	 	while(dtAccumilator >= stepSize){
	 		world->Step(stepSize, velIterations, collisionIterations);

	 		//now decrement the accumilator by the step size. repeat until you have less accumilated
	 		//than a step size
	 		dtAccumilator -= stepSize;
	 	}


	 };

	 void Draw(){};


	 void Shutdown(){};

	 b2World *getWorld(){
	 	return this->world;
	 }
} ;