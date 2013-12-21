#pragma once
#include "Process.h"
#include "processMgr.h"
#include "../IO/Hash.h"

#include "State.h"

#include <map>

/*!Manages State. a simplified Finite State Machine implementation.

stateProcess manages State classes. it's responsibility is to 
transition between different States and to execute the current State

\sa State
*/  

class stateProcess : public Process{
public:
	stateProcess(processMgr &_processManager, Settings &_settings, EventManager &_eventManager) :
	 Process("stateProcess"), processManager(_processManager), settings(_settings), eventManager(_eventManager){

	 	this->transitioning = false;
	 	this->currentState = NULL;
	 
	 };

	 /*!Adds a state to the State Machine

	 @param [in] state A state to be added
	 @param [in] currentState whether state is the default state to begin with 
	 */
 	 void addState(State *state, bool currentState){
	 	this->states[state->getHashedName()] = state;

	 	state->Init(processManager, settings, eventManager);

	 	this->currentState = state;
	 };

	 /*!Updates the current State */
	 void Update(float dt){
	 	assert(this->currentState != NULL);
	 	this->currentState->Update(dt);
	 };
	 /*!Draws the current State */
	 void Draw(){
	 	assert(this->currentState != NULL);
	 	this->currentState->Draw();
	 };

	 /*!Saves and destroys all states owned*/
	 void Shutdown(){
	 	assert(this->currentState != NULL);
		
	 };


private:
	processMgr &processManager;
	Settings &settings;
	EventManager &eventManager;

	std::map<const Hash*, State*> states;
	State *currentState;

	//whether another state is being transitioned to.
	bool transitioning;

};
