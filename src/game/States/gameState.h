#pragma once
#include "../../core/controlFlow/State.h"
#include "../../core/controlFlow/dummyStateSaveLoader.h"
#include "../../core/componentSys/ObjectManager.h"
#include "../../core/componentSys/ObjectMgrProcess.h"
   

#include "../gameStateHelpers/playerEventHandler.h"
#include "../gameStateHelpers/gunsManager.h"
#include "../gameStateHelpers/playerController.h"

class gameState : public State{
public:
	gameState() : State("gameState"){};
	
	void Update(float dt){
		this->_playerController->Update(dt);
	};
	
	void Draw(){};

	stateSaveLoader *createSaveLoader(){
		return new dummyStateSaveLoader();
	}

protected:
	void _Init();

	void _generateBoundary(vector2 levelDim);
	void _generateTerrain(unsigned long long seed, vector2& playerInitPos, vector2& levelDim);
	Object* _createPlayer(vector2 playerInitPos, vector2 levelDim);
	void _createEnemies(vector2 levelDim, Object *player);
	void _createDummy(vector2 levelDim);

	ObjectManager *objectManager;
	viewProcess *viewProc;

	playerController *_playerController;
};
