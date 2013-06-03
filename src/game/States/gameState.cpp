#pragma once
#include "gameState.h"
#include "../../core/Object.h"
#include "../../core/renderUtil.h"

#include "../ObjProcessors/groundMoveProcessor.h"
#include "../ObjProcessors/cameraProcessor.h"


void gameState::_generateTerrain(unsigned long long seed, vector2 playerInitPos){
	Object *terrainObj = new Object("Terrain");

	terrainObj->addProp(Hash::getHash("seed"), new Prop<unsigned long long>(seed));
	terrainObj->addProp(Hash::getHash("dim"), new v2Prop(vector2(4000, 720)));

	terrainObj->addProp(Hash::getHash("Terrain"), new dummyProp());

	objectManager->addObject(terrainObj);

};

void gameState::_createPlayer(vector2 playerInitPos){

	phyData phy;
	renderData render;
	moveData move;
	cameraData camera;

	//phy---------------------------------------------------------------------
	Object *playerObj = new Object("player");

	vector2 *pos = playerObj->getProp<vector2>(Hash::getHash("position"));
	*pos = playerInitPos;


	
	phy.bodyDef.type = b2_dynamicBody;


	//b2PolygonShape playerBoundingBox; 
	//playerBoundingBox.SetAsBox(1.0f, 1.0f);

	b2CircleShape playerBoundingBox;
	playerBoundingBox.m_radius = 1.0f;


	b2FixtureDef playerFixtureDef;
	playerFixtureDef.shape = &playerBoundingBox;
	playerFixtureDef.friction = 1.0;

	phy.fixtureDef.push_back(playerFixtureDef);


	//renderer----------------------------------------------------------------	
	sf::Shape *playerSFMLShape = renderUtil::createShape(&playerBoundingBox, 
		viewProc->getGame2RenderScale());

	playerSFMLShape->setFillColor(sf::Color::Green);

	Renderer playerShapeRenderer(playerSFMLShape);
	
	render.addRenderer(playerShapeRenderer);
	
	//movement-----------------------------------------------------------
	move.strength = vector2(60, 60);
	move.stopStrength = vector2(4, 0);

	//camera---------------------------------------------------------------
	camera.enabled = true;
	camera.maxCoord = vector2(1280 * 2, 720 * 3);
	camera.maxMoveAmt = vector2(30, 60);
	camera.boxHalfW = 360;
	camera.boxHalfH = 300;

	//final creation--------------------------------------------------------
	playerObj->addProp(Hash::getHash("phyData"), new Prop<phyData>(phy));
	playerObj->addProp(Hash::getHash("renderData"), new Prop<renderData>(render));
	playerObj->addProp(Hash::getHash("moveData"), new Prop<moveData>(move));
	playerObj->addProp(Hash::getHash("cameraData"), new Prop<cameraData>(camera));
	objectManager->addObject(playerObj);

	//players handlers--------------------------------------------------
	WSADHandlerData WSADdata;
	WSADdata.left = sf::Keyboard::Key::A;
	WSADdata.right = sf::Keyboard::Key::D;
	WSADdata.up = sf::Keyboard::Key::W;

	WSADdata.objMoveData = playerObj->getProp<moveData>(Hash::getHash("moveData"));

	this->playerMoveHandler = new WSADHandler(this->eventManager, WSADdata);

};

void gameState::_createDummy(){
	renderData render;

	
	Object *dummy = new Object("dummy");

	vector2 *pos = dummy->getProp<vector2>(Hash::getHash("position"));
	*pos = viewProc->getRender2GameScale() * vector2(rand() % 700, rand() % 600);

	//renderer----------------------------------------------------------------	
	sf::Shape *shape = new sf::CircleShape(10);
	shape->setFillColor(sf::Color::Blue);

	Renderer shapeRenderer(shape);
	render.addRenderer(shapeRenderer);
	//final---------------------------------
	dummy->addProp(Hash::getHash("renderData"), new Prop<renderData>(render));
	objectManager->addObject(dummy);


};


void gameState::_generateBoundary(vector2 levelDim){


	vector2 bottomLeft = vector2(0, 0);
	vector2 topRight = levelDim;
	float thickness = 4.0f;

	phyData physicsData;
	renderData render;

	Object *boundaryObject = new Object("boundary");

	vector2 *pos = boundaryObject->getProp<vector2>(Hash::getHash("position"));
	*pos = levelDim * 0.5;

	physicsData.bodyDef.type = b2_staticBody;

	{
	//BOTTOM---------------------------------------------------------
	b2PolygonShape bottom; 
	vector2 bottomCenter = vector2(0, -levelDim.y / 2.0);//vector2(levelDim.x / 2.0, 0);
	bottom.SetAsBox(levelDim.x / 2.0, thickness, bottomCenter, 0);

	b2FixtureDef bottomFixtureDef;
	bottomFixtureDef.shape = &bottom;
	bottomFixtureDef.friction = 1.0;

	physicsData.fixtureDef.push_back(bottomFixtureDef);

	sf::Shape *shape = renderUtil::createShape(&bottom, 
		viewProc->getGame2RenderScale());
	shape->setFillColor(sf::Color::Blue);
	Renderer renderer(shape);
	render.addRenderer(renderer);

	}
	
	{
	//TOP---------------------------------------------------------
	b2PolygonShape top; 
	vector2 topCenter = vector2(0, levelDim.y / 2.0);//vector2(levelDim.x / 2.0, levelDim.y / 2.0);
	top.SetAsBox(levelDim.x / 2.0, thickness, topCenter, 0);

	b2FixtureDef topFixtureDef;
	topFixtureDef.shape = &top;
	topFixtureDef.friction = 2.0;

	physicsData.fixtureDef.push_back(topFixtureDef);


	sf::Shape *shape = renderUtil::createShape(&top, 
	viewProc->getGame2RenderScale());
	shape->setFillColor(sf::Color::Blue);
	Renderer renderer(shape);
	render.addRenderer(renderer);

	}


	{
	//LEFT---------------------------------------------------------
	b2PolygonShape left; 
	vector2 leftCenter = vector2(-levelDim.x / 2.0, 0);
	left.SetAsBox(thickness, levelDim.y / 2.0, leftCenter, 0);

	b2FixtureDef leftFixtureDef;
	leftFixtureDef.shape = &left;
	leftFixtureDef.friction = 1.0;

	physicsData.fixtureDef.push_back(leftFixtureDef);

	sf::Shape *shape = renderUtil::createShape(&left, 
		viewProc->getGame2RenderScale());
	shape->setFillColor(sf::Color::Blue);
	Renderer renderer(shape);
	render.addRenderer(renderer);
	}

	{
	//RIGHT---------------------------------------------------------
	b2PolygonShape right; 
	vector2 rightCenter = vector2(levelDim.x / 2.0, 0);
	right.SetAsBox(thickness, levelDim.y / 2.0, rightCenter, 0);

	b2FixtureDef rightFixtureDef;
	rightFixtureDef.shape = &right;
	rightFixtureDef.friction = 1.0;

	physicsData.fixtureDef.push_back(rightFixtureDef);

	sf::Shape *shape = renderUtil::createShape(&right, 
		viewProc->getGame2RenderScale());
	shape->setFillColor(sf::Color::Blue);
	Renderer renderer(shape);
	render.addRenderer(renderer);
	}

	boundaryObject->addProp(Hash::getHash("phyData"), new Prop<phyData>(physicsData));
	boundaryObject->addProp(Hash::getHash("renderData"), new Prop<renderData>(render));
	objectManager->addObject(boundaryObject);

}

void gameState::_createEnemies(){};
