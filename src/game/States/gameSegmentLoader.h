#pragma once
#include "../../core/controlFlow/State.h"
#include "../../core/controlFlow/dummyStateSaveLoader.h"



#include "../level/gameSegment.h"

class gameSegmentLoader : public State{
private:
	gameSegment *segment;

	std::string segmentName;

	bool doneLoading;
public:
	gameSegmentLoader() : State("gameSegmentLoader"){}
	
	void Update(float dt){};
	void Draw(){};

	
	stateSaveLoader *createSaveLoader(){
		return new dummyStateSaveLoader();
	}

	gameSegment *getLoadedSegment();
};
