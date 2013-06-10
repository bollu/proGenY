#pragma once
#include "objectFactory.h"
#include "../../core/Process/viewProcess.h"

class playerCreator : public objectCreator{
private:
	viewProcess *viewProc;

	cameraData camData;

public:
	playerCreator(viewProcess *_viewProc) : viewProc(_viewProc){}
	
	void setCameraData(cameraData &camData){
		this->camData = camData;
		this->camData.enabled = true;
		
	}

	Object *createObject(vector2 playerInitPos) const{

		Object *playerObj = new Object("player");
		
		vector2 *pos = playerObj->getProp<vector2>(Hash::getHash("position"));
		*pos = playerInitPos;


		phyData phy;
		renderData render;
		moveData move;
		
		//physics------------------------------------------------------------
		phy.collisionType = Hash::getHash("player");
		phy.bodyDef.type = b2_dynamicBody;

		b2CircleShape *playerBoundingBox = new b2CircleShape();
		playerBoundingBox->m_radius = 1.0f;

		b2FixtureDef playerFixtureDef;
		playerFixtureDef.shape = playerBoundingBox;
		playerFixtureDef.friction = 0.0;
		playerFixtureDef.restitution = 0.0;

		phy.fixtureDef.push_back(playerFixtureDef);


	//renderer---------------------------------------------------------------
		sf::Shape *playerSFMLShape = renderUtil::createShape(playerBoundingBox, 
			viewProc->getGame2RenderScale());

		playerSFMLShape->setFillColor(sf::Color::Green);

		Renderer playerShapeRenderer(playerSFMLShape);

		render.addRenderer(playerShapeRenderer);

	//movement-----------------------------------------------------------
		move.xVel = 40;
		move.xAccel = 2;
		move.movementDamping = vector2(0.04, 0.0);
		move.jumpRange = viewProc->getRender2GameScale() * 512;
		move.jumpHeight = viewProc->getRender2GameScale() * 128;
		
	//camera---------------------------------------------------------------
		

	//final creation--------------------------------------------------------
	util::msgLog("adding phyData");
		playerObj->addProp(Hash::getHash("phyData"), 
			new Prop<phyData>(phy));

		playerObj->addProp(Hash::getHash("renderData"), 
			new Prop<renderData>(render));

		playerObj->addProp(Hash::getHash("moveData"), 
			new Prop<moveData>(move));

		playerObj->addProp(Hash::getHash("cameraData"), 
			new Prop<cameraData>(this->camData));


		return playerObj;
	};
};