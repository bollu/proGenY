#include "worldProcess.h"


worldProcess::worldProcess(processMgr &processManager, Settings &settings, EventManager &eventManager) :
Process("worldProcess"){
		//world = new b2World(settings.getPrimitive<vector2>("worldGravity")->getVal());

	vector2 *gravity = settings.getSetting<vector2>(Hash::getHash("gravity"));
	this->stepSize = *settings.getSetting<float>(Hash::getHash("stepSize"));

	this->velIterations = *settings.getSetting<int>(Hash::getHash("velIterations"));
	this->collisionIterations = *settings.getSetting<int>(Hash::getHash("collisionIterations"));

	world = new b2World(*gravity);
	//world->SetContinuousPhysics(true);

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
	//HACK!;
	/*simulationThread = std::thread(&worldProcess::_Simulate, this);
	simulationThread.detach();
	*/
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

	//worldAccess.lock();

	this->dtAccumilator += dt;

	//worldAccess.unlock();

	//HACK!;
	this->_Simulate();

};

void worldProcess::Draw(){};


void worldProcess::Shutdown(){

	//this->worldAccess.lock();

	delete(this->world);
	this->world = (b2World*)(0xDEADBEEF);

	//HACK!
	//this->simulationThread.join();
	
	//this->worldAccess.unlock();
};



#include <thread>
#include <chrono>
void worldProcess::_Simulate(){

	//while(world != (b2World*)(0xDEADBEEF)){

		//this->worldAccess.lock();
		
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

		
		//this->worldAccess.unlock();

		/*std::chrono::milliseconds dura( static_cast<int>(stepSize * 1000) );
    	std::this_thread::sleep_for( dura );
 	
	}*/	

};


vector2 worldProcess::getGravity(){
	//worldAccess.lock();
	vector2 gravity =  vector2::cast(this->world->GetGravity());
	//worldAccess.unlock();
	return gravity;
};

b2Body *worldProcess::createBody(const b2BodyDef* def){
	//worldAccess.lock();
	b2Body *body = this->world->CreateBody(def);
	//worldAccess.unlock();

	return body;
};

void worldProcess::destroyBody(b2Body* body){
	//worldAccess.lock();
	this->world->DestroyBody(body);
	//worldAccess.unlock();
};

void worldProcess::setContactListener(b2ContactListener* listener){
	//worldAccess.lock();
	this->world->SetContactListener(listener);
	//worldAccess.unlock();
};
