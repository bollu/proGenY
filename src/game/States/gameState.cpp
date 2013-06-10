#pragma once
#include "gameState.h"
#include "../../core/Object.h"
#include "../../core/renderUtil.h"

#include "../ObjProcessors/groundMoveProcessor.h"
#include "../ObjProcessors/cameraProcessor.h"
#


void gameState::_generateTerrain(unsigned long long seed, vector2 playerInitPos){
	Object *terrainObj = new Object("Terrain");

	terrainObj->addProp(Hash::getHash("seed"), new Prop<unsigned long long>(seed));
	terrainObj->addProp(Hash::getHash("dim"), new v2Prop(vector2(4000, 720)));

	terrainObj->addProp(Hash::getHash("Terrain"), new dummyProp());

	objectManager->addObject(terrainObj);
	

};

#include "../factory/playerCreator.h"
#include "../factory/boundaryCreator.h"
#include "../factory/dummyCreator.h"
#include "../factory/bulletCreator.h"

void gameState::_initFactory(){

	
	this->objFactory.attachObjectCreator(Hash::getHash("dummy"),
				new dummyCreator(this->viewProc));

	this->objFactory.attachObjectCreator(Hash::getHash("player"),
				new playerCreator(this->viewProc));

	this->objFactory.attachObjectCreator(Hash::getHash("boundary"),
				new boundaryCreator(this->viewProc));

	this->objFactory.attachObjectCreator(Hash::getHash("bullet"),
				new bulletCreator(this->viewProc));
};


void gameState::_createPlayer(vector2 playerInitPos){

	playerCreator *creator = (playerCreator*)objFactory.getCreator(
		Hash::getHash("player"));

	cameraData camera;
	camera.enabled = true;
	camera.maxCoord = vector2(1280 * 2, 720 * 3);
	camera.maxMoveAmt = vector2(30, 60);
	camera.boxHalfW = 360;
	camera.boxHalfH = 300;

	creator->setCameraData(camera);

	Object *playerObj = creator->createObject(playerInitPos);

	//players handlers--------------------------------------------------
	WSADHandlerData WSADdata;
	WSADdata.left = sf::Keyboard::Key::A;
	WSADdata.right = sf::Keyboard::Key::D;
	WSADdata.up = sf::Keyboard::Key::W;

	WSADdata.objMoveData = playerObj->getProp<moveData>(Hash::getHash("moveData"));
	WSADdata.physicsData = playerObj->getProp<phyData>(Hash::getHash("phyData"));

	
	this->playerMoveHandler = new WSADHandler(this->eventManager, WSADdata);
	

	objectManager->addObject(playerObj);
	
};

void gameState::_createDummy(){


	dummyCreator *creator = (dummyCreator*)objFactory.getCreator(
		Hash::getHash("dummy"));

	
	creator->setRadius(1.0f);

	vector2 randPos = vector2(400, 200);
	
	randPos *= viewProc->getRender2GameScale();
	Object *dummy = creator->createObject(randPos);


	objectManager->addObject(dummy);

	/*
	renderData render;
	phyData phy;
	
	Object *dummy = new Object("dummy");

	vector2 *pos = dummy->getProp<vector2>(Hash::getHash("position"));
	*pos = viewProc->getRender2GameScale() * 
		vector2(700 + rand() % 1280, rand() % 600);

	//physics-----------------------------------


	//renderer----------------------------------------------------------------	
	sf::Shape *shape = new sf::CircleShape(10);
	shape->setFillColor(sf::Color::Blue);

	Renderer shapeRenderer(shape);
	render.addRenderer(shapeRenderer);
	

	//final---------------------------------
	dummy->addProp(Hash::getHash("renderData"), new Prop<renderData>(render));
	objectManager->addObject(dummy);
	*/

};


void gameState::_generateBoundary(vector2 levelDim){

	boundaryCreator *creator = (boundaryCreator*)objFactory.getCreator(
		Hash::getHash("boundary"));

	creator->setBoundaryThickness(1.0f);
	creator->setDimensions(levelDim);

	Object *boundary = creator->createObject(vector2(0, 0));


	objectManager->addObject(boundary);

}

void gameState::_createEnemies(){

	bulletCreator *creator = (bulletCreator *)objFactory.getCreator(
		Hash::getHash("bullet"));

	creator->setRadius(0.8f);
	vector2 pos = vector2(400, 400);
	
	creator->setBeginVel(vector2(0, 4.8));

	creator->setEnemyCollision(Hash::getHash("dummy"));

	pos *= viewProc->getRender2GameScale();
	Object *obj = creator->createObject(pos);


	objectManager->addObject(obj);

};
