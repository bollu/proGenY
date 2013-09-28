#pragma once
#include "../../core/controlFlow/State.h"
#include "../../core/controlFlow/dummyStateSaveLoader.h"

class mainMenuState : public State{
public:
	mainMenuState() : State("mainMenuState"){};
	
	void Update(float dt){};
	
	void Draw(){};

	stateSaveLoader *createSaveLoader(){
		return new dummyStateSaveLoader();
	}

private:
	
};
