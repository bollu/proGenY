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


playerController::playerController(eventMgr *eventManager, objectMgr *objectManager,
	objectFactory *factory, viewProcess *viewProc){
	
	this->_eventManager = eventManager;
	this->_objectManager = objectManager;
	this->_objectFactory = factory; 
	this->viewProc = viewProc;

	//this->gunsMgr = new gunsManager(*_eventManager);
	this->gunsMgr = NULL;
	this->playerHandler = NULL;

	
};

void playerController::createPlayer(vector2 levelDim, vector2 initPos, playerCreator *creator,
	playerHandlerData playerData){


	//camera data--------------------------------------

	cameraData camera;
	camera.enabled = true;
	camera.maxCoord = (viewProc->game2ViewCoord(levelDim));
	camera.maxMoveAmt = vector2(30, 60);
	camera.boxHalfW = 360;
	camera.boxHalfH = 200;

	creator->Init(camera);

	//player created--------------------------------------------
	this->_createPlayer(initPos, creator);
	util::infoLog<<"player created";

	//create guns
	this->_createGunsManager(this->player);

	//create playerEventHandler
	playerData.player = this->player;
	this->_createPlayerEventHandler(playerData);
	util::infoLog<<"player event handler created";

	//add player
	this->_objectManager->addObject(this->player);
	util::infoLog<<"added to object manager";

};

void playerController::addGun(Object *gun, bool currentGun){
	assert(this->gunsMgr != NULL);
	this->gunsMgr->addGun(gun, currentGun);
};

Object *playerController::getPlayer(){
	return this->player;
};

void playerController::Update(float dt)
{
	assert(this->playerHandler != NULL);
	this->playerHandler->Update();
};

void playerController::_createPlayer(vector2 initPos, playerCreator *creator){
	this->player = creator->createObject(initPos);
};

void playerController::_createGunsManager(Object *player){
 	this->gunsMgr = new gunsManager(*this->_eventManager, *this->_objectFactory, 
 			*this->_objectManager, player);
};

void playerController::_createPlayerEventHandler(playerHandlerData &playerData)
{
	this->playerHandler = new playerEventHandler(this->_eventManager, playerData); 

};
