#pragma once
#include "Process.h"
#include "processMgr.h"
#include "../Hash.h"

#include "../State/State.h"

#include <map>


class stateProcess : public Process{
public:
	stateProcess(processMgr &_processManager, Settings &_settings, eventMgr &_eventManager) :
	 Process("stateProcess"), processManager(_processManager), settings(_settings), eventManager(_eventManager){

	 	this->transitioning = false;
	 	this->currentState = NULL;
	 
	 };

	 void addState(State *state, bool currentState){
	 	this->states[state->getHashedName()] = state;

	 	state->Init(processManager, settings, eventManager);

	 	this->currentState = state;
	 };

	 void Update(){
	 	assert(this->currentState != NULL);
	 	this->currentState->Update();
	 };

	 void Draw(){
	 	assert(this->currentState != NULL);
	 	this->currentState->Draw();
	 };

	 void Shutdown(){
	 	assert(this->currentState != NULL);
		
	 };


private:
	processMgr &processManager;
	Settings &settings;
	eventMgr &eventManager;

	std::map<const Hash*, State*> states;
	State *currentState;

	//whether another state is being transitioned to.
	bool transitioning;

};