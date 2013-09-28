#pragma once
#include "objectFactory.h"
#include "../../core/Rendering/viewProcess.h"
#include "../defines/renderingLayers.h"

class playerCreator : public objectCreator{
private:
	viewProcess *viewProc;

	cameraData camData;

public:
	playerCreator(viewProcess *_viewProc) : viewProc(_viewProc){}
	
	void Init(cameraData &camData){
		this->camData = camData;
		this->camData.enabled = true;
		
	}

	Object *createObject(vector2 playerInitPos) const{

		Object *playerObj = new Object("player");
		
		vector2 *pos = playerObj->getPrimitive<vector2>(Hash::getHash("position"));
		*pos = playerInitPos;


		phyData phy;
		renderData render;
		groundMoveData move;
		
		//physics------------------------------------------------------------
		phy.collisionType = Hash::getHash("player");
		phy.bodyDef.type = b2_dynamicBody;

		b2CircleShape *playerBoundingBox = new b2CircleShape();
		playerBoundingBox->m_radius = 1.0f;

		b2FixtureDef fixtureDef;
		fixtureDef.shape = playerBoundingBox;
		fixtureDef.friction = 0.0;
		fixtureDef.restitution = 0.0;

		phy.fixtureDef.push_back(fixtureDef);


	//renderNode---------------------------------------------------------------
		sf::Shape *SFMLShape = renderUtil::createShape(playerBoundingBox, 
			viewProc);
		
		SFMLShape->setFillColor(sf::Color::Green);

		shapeRenderNode *renderNode = new shapeRenderNode(SFMLShape, renderingLayers::action);

		render.addRenderer(renderNode);

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

		playerObj->addProp(Hash::getHash("renderData"), 
			new Prop<renderData>(render));

		playerObj->addProp(Hash::getHash("groundMoveData"), 
			new Prop<groundMoveData>(move));

		playerObj->addProp(Hash::getHash("cameraData"), 
			new Prop<cameraData>(this->camData));


		return playerObj;
	};
};
