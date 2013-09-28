#pragma once
#include "objectFactory.h"
#include "../../core/Rendering/viewProcess.h"
#include "../ObjProcessors/offsetProcessor.h"

#include "../../core/componentSys/processor/renderProcessor.h"
#include "../../core/componentSys/processor/phyProcessor.h"
#include "../../core/Rendering/renderUtil.h"

#include "../defines/renderingLayers.h"


class enemyCreator : public objectCreator{
private:
	viewProcess *viewProc;
	const Hash *enemyCollision;

public:

	enemyCreator(viewProcess *_viewProc) : viewProc(_viewProc){}


	Object *createObject(vector2 _pos) const{
		renderData render;
		phyData phy;

		Object *enemy = new Object("enemy");

		vector2 *pos = enemy->getPrimitive<vector2>("position");
		*pos = _pos;


		//physics------------------------------------------------------------
		phy.collisionType = Hash::getHash("enemy");
		phy.bodyDef.type = b2_dynamicBody;
		phy.bodyDef.gravityScale = 0.0f;

		b2CircleShape *boundingBox = new b2CircleShape();
		boundingBox->m_radius = 1.0f;

		b2FixtureDef fixtureDef;
		fixtureDef.shape = boundingBox;
		fixtureDef.friction = 0.0;
		fixtureDef.restitution = 0.0;

		phy.fixtureDef.push_back(fixtureDef);


	//SFMLShape--------------------------------------------------------------------
		sf::Shape *SFMLShape = renderUtil::createShape(boundingBox, 
			viewProc);

		SFMLShape->setFillColor(sf::Color::White);
		SFMLShape->setOutlineColor(sf::Color::Black);
		SFMLShape->setOutlineThickness(-3.0);

		shapeRenderNode* renderNode = new shapeRenderNode(SFMLShape, renderingLayers::aboveAction);
		render.addRenderer(renderNode);
	
	
		//final---------------------------------
		enemy->addProp(Hash::getHash("renderData"), 
			new Prop<renderData>(render));
		enemy->addProp(Hash::getHash("phyData"), 
			new Prop<phyData>(phy));
	
		return enemy;
	};
};
