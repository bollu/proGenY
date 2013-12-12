#pragma once
#include "playerController.h"
#include "gunsManager.h"
#include "playerEventHandler.h"

#include "../../core/Object.h"
#include "../../core/renderUtil.h"

#include "../../core/Messaging/eventMgr.h"
#include "../ObjProcessors/groundMoveProcessor.h"
#include "../ObjProcessors/cameraProcessor.h"
#include "../factory/playerCreator.h"
#include "../../core/objectMgr.h"

#include "../factory/objectFactory.h"


playerController::playerController(eventMgr *eventManager, objectMgr *objectManager, viewProcess *viewProc){
	
	eventManager_ = eventManager;
	objectManager_ = objectManager;
	viewProc_ = viewProc;

	//gunsMgr_ = new gunsManager(*_eventManager);
	gunsMgr_ = NULL;
	playerHandler_ = NULL;

	
};

void playerController::createPlayer(vector2 levelDim, vector2 initPos, playerCreator *creator,
	playerHandlerData playerData){


	//camera data--------------------------------------

	CameraData camera;
	camera.enabled = true;
	
	camera.maxCoord = levelDim;//(viewProc_->game2ViewCoord(levelDim));
	PRINTVECTOR2(camera.maxCoord);

	camera.maxMoveAmt = vector2(32, 64);
	camera.boxHalfW = 100;
	camera.boxHalfH = 100;

	creator->setCameraData(camera);

	//player_ created--------------------------------------------
	_createPlayer(initPos, creator);
	util::infoLog<<"player_ created";

	//create guns
	_createGunsManager(player_);

	//create playerEventHandler
	playerData.player = player_;
	_createPlayerEventHandler(playerData);
	util::infoLog<<"player_ event handler created";

	//add player_
	objectManager_->addObject(player_);
	util::infoLog<<"added to object manager";

};

void playerController::addGun(Object *gun, bool currentGun){
	assert(gunsMgr_ != NULL);
	gunsMgr_->addGun(gun, currentGun);
};

Object *playerController::getPlayer(){
	return player_;
};

void playerController::Update(float dt)
{
	assert(playerHandler_ != NULL);
	playerHandler_->Update();
};

void playerController::_createPlayer(vector2 initPos, playerCreator *creator){
	player_ = creator->createObject(initPos);
};

void playerController::_createGunsManager(Object *player){
 	gunsMgr_ = new gunsManager(*eventManager_, *objectManager_, viewProc_, player);
};

void playerController::_createPlayerEventHandler(playerHandlerData &playerData)
{
	playerHandler_ = new playerEventHandler(eventManager_, playerData); 
};
