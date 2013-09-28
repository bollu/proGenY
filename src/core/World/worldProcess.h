#pragma once
#include "../../include/Box2D/Box2D.h"
#include "../controlFlow/Process.h"
#include "../controlFlow/processMgr.h"



/*!encapsulates the box2d simulation as a Process

the worldProcess is responsible for the physics engine. So,
it must ensure that the physics runs smoothly and is never corrupt.
*/
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
		//world = new b2World(settings.getPrimitive<vector2>("worldGravity")->getVal());
		
	 	vector2 *gravity = settings.getPrimitive<vector2>(Hash::getHash("gravity"));
	 	this->stepSize = *settings.getPrimitive<float>(Hash::getHash("stepSize"));

	 	this->velIterations = *settings.getPrimitive<int>(Hash::getHash("velIterations"));
	 	this->collisionIterations = *settings.getPrimitive<int>(Hash::getHash("collisionIterations"));

		world = new b2World(*gravity);


		this->dtAccumilator = 0;
		maxAccumilation = this->stepSize * 10;

	};

	/*!returns the size of each simuation step
	\return the step size of the simulation
	*/
	float getStepSize(){
		return this->stepSize;
	}

	/*!returns the maximum accumulation of time possible
		before it lapses

	if time accumulates such that it becomes 
	greater than this limit, rater than trying to catch up,
	just lapse and start counting from the beginning again.
	This is useful when, for example, we switch windows. on returning,
	a lot of time will have elapsed. simulating those many steps will 
	cause the whole simulation to "explode". So, just lapse and start counting anew

	\return the maximum accumulation of time 
	*/
	float getMaxAccumilation(){
		return this->maxAccumilation;
	}

	/*!pauses the simulation */
	 void Pause(){
	 	this->paused = true;
	 };

	 /*!Resumes the simulation*/
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


	 void Shutdown(){
	 	delete(this->world);
	 };

	 /*!returns the internal box2d World
	 \return the box2d world pointer */
	 b2World *getWorld(){
	 	return this->world;
	 }
};
