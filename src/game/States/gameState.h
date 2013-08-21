#pragma once
#include "../../core/State/State.h"
#include "../../core/State/dummyStateSaveLoader.h"
#include "../../core/objectMgr.h"
#include "../../core/Process/objectMgrProcess.h"


#include "../factory/objectFactory.h"
#include "../gameStateHelpers/playerEventHandler.h"
#include "../gameStateHelpers/gunsManager.h"
#include "../gameStateHelpers/playerController.h"

class gameState : public State
{
public:
	gameState () : State( "gameState" ){}

	void Update ( float dt ){
		this->_playerController->Update( dt );
	}


	void Draw (){}

	stateSaveLoader *createSaveLoader (){
		return ( new dummyStateSaveLoader() );
	}


protected:
	void _Init ();

	void _initFactory ();
	void _generateBoundary ( vector2 levelDim );
	void _generateTerrain ( unsigned long long seed, vector2 playerInitPos, vector2 levelDim );
	void _createPlayer ( vector2 playerInitPos, vector2 levelDim );
	void _createEnemies ( vector2 levelDim );
	void _createDummy ( vector2 levelDim );

	objectMgr *objectManager;
	viewProcess *viewProc;
	playerController *_playerController;
	objectFactory objFactory;
};