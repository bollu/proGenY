#pragma once
#include "../../include/Box2D/Box2D.h"
#include "../controlFlow/Process.h"
#include "../controlFlow/processMgr.h"

#include <thread>
#include <atomic>
#include <mutex>

/*!encapsulates the box2d simulation as a Process

the worldProcess is responsible for the physics engine. So,
it must ensure that the physics runs smoothly and is never corrupt.
*/
class worldProcess : public Process{

public:

	worldProcess(processMgr &processManager, Settings &settings, eventMgr &eventManager);

	/*!returns the size of each simulation step
	\return the step size of the simulation
	*/
	float getStepSize();

	/*!returns the maximum accumulation of time possible
		before it lapses

	if time accumulates such that it becomes 
	greater than this limit, rater than trying to catch up,
	just lapse and start counting from the beginning again.
	This is useful when, for example, we switch windows. on returning,
	a lot of time will have elapsed. simulating those many steps will 
	cause the whole simulation to explode. So, just lapse and start counting anew

	\return the maximum accumulation of time 
	*/
	float getMaxAccumilation();

	void Start();

	/*!pauses the simulation */
	 void Pause();
	 /*!Resumes the simulation*/
	 void Resume();

	 void Update(float dt);
	 void Draw();
	 void Shutdown();

	 /*!returns the internal box2d World
	 \return the box2d world pointer */
	// b2World *getWorld();

	 void _Simulate();


	 vector2 getGravity();
	 b2Body *createBody(const b2BodyDef* def);
	 void destroyBody(b2Body* body);
	 void setContactListener(b2ContactListener* listener);

private:
	b2World *world = NULL;

	bool paused = false;
	
	std::mutex worldAccess;          
	
	float stepSize;

	//the accumilator used to accumulate partial step sizes, so that once every sepSize time has
	//passed, we can step the world 
	float dtAccumilator;

	//if the amount accumilated passes maxAccumilation, the accumilator "lapses". this is to make sure
	//that it dosen't get sucked into a "spiral of doom" where it updates more as it's accumilated a lot,
	//and this causes even more accumilation, and so on and so forth
	float maxAccumilation;

	float velIterations, collisionIterations;

	std::thread simulationThread; 


};
