#pragma once
#include "objectFactory.h"
#include "../../core/Process/viewProcess.h"
#include "../defines/renderingLayers.h"

class playerCreator : public objectCreator{
private:
	viewProcess *viewProc;

	CameraData camData;

public:
	playerCreator(viewProcess *_viewProc) : viewProc(_viewProc){}
	
	void setCameraData(CameraData &camData){
		this->camData = camData;
		this->camData.enabled = true;
		
	}

	Object *createObject(vector2 playerInitPos) const{

		Object *playerObj = new Object("player");
		
		vector2 *pos = playerObj->getProp<vector2>(Hash::getHash("position"));
		*pos = playerInitPos;


		phyData phy;
		RenderData renderData;
		MoveData move;
		
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
			viewProc);

		playerSFMLShape->setFillColor(sf::Color::Green);

		shapeRenderNode *playerShapeRenderer = new shapeRenderNode(playerSFMLShape, renderingLayers::action);

		renderData.addRenderer(playerShapeRenderer);

	//movement-----------------------------------------------------------
		move.xVel = 60;
		move.xAccel = 3;
		move.movementDamping = vector2(0.05, 0.0);
		move.jumpRange = viewProc->getRender2GameScale() * 256;
		move.jumpHeight = viewProc->getRender2GameScale() * 128;
		
	//camera---------------------------------------------------------------
		

	//final creation--------------------------------------------------------
		playerObj->addProp(Hash::getHash("phyData"), 
			new Prop<phyData>(phy));

		playerObj->addProp(Hash::getHash("RenderData"), 
			new Prop<RenderData>(renderData));

		playerObj->addProp(Hash::getHash("MoveData"), 
			new Prop<MoveData>(move));

		playerObj->addProp(Hash::getHash("CameraData"), 
			new Prop<CameraData>(this->camData));


		return playerObj;
	};
};