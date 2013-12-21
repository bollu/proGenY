#pragma once
#include "objectFactory.h"
#include "../../core/Rendering/viewProcess.h"
#include "../ObjProcessors/OffsetProcessor.h"

#include "../../core/componentSys/processor/RenderProcessor.h"
#include "../../core/componentSys/processor/PhyProcessor.h"
#include "../../core/Rendering/renderUtil.h"

#include "../defines/renderingLayers.h"


class enemyCreator : public objectCreator{
private:
	viewProcess *viewProc;
	const Hash *enemyCollision;

public:

	enemyCreator(viewProcess *_viewProc) : viewProc(_viewProc){}


	Object *createObject(vector2 _pos) const{
		RenderData render;
		
		Object *enemy = new Object("enemy");

		vector2 *pos = enemy->getPrimitive<vector2>("position");
		*pos = _pos;


		//physics------------------------------------------------------------
		b2BodyDef bodyDef;
		bodyDef.type = b2_dynamicBody;
		bodyDef.gravityScale = 0.0f;

		b2CircleShape enemyBoundingBox;
		enemyBoundingBox.m_radius = 1.0f;

		b2FixtureDef fixtureDef;
		fixtureDef.shape = &enemyBoundingBox;
		fixtureDef.friction = 0.0;
		fixtureDef.restitution = 0.0;

		//phy.fixtureDef.push_back(fixtureDef);
		PhyData phy = PhyProcessor::createPhyData(&bodyDef, &fixtureDef); 
		phy.collisionType = Hash::getHash("enemy");
	

	//SFMLShape--------------------------------------------------------------------
		sf::Shape *SFMLShape = renderUtil::createShape(&enemyBoundingBox, 
			viewProc);

		SFMLShape->setFillColor(sf::Color::White);
		SFMLShape->setOutlineColor(sf::Color::Black);
		SFMLShape->setOutlineThickness(-3.0);

		shapeRenderNode* renderNode = new shapeRenderNode(SFMLShape, renderingLayers::aboveAction);
		render.addRenderer(renderNode);
	
	
		//final---------------------------------
		enemy->addProp(Hash::getHash("RenderData"), 
			new Prop<RenderData>(render));
		enemy->addProp(Hash::getHash("PhyData"), 
			new Prop<PhyData>(phy));
	
		return enemy;
	};
};
