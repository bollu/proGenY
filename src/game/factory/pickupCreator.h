#pragma once
#include "objectFactory.h"
#include "../../core/Rendering/viewProcess.h"
#include "../ObjProcessors/PickupProcessor.h"
#include "../../core/componentSys/processor/RenderProcessor.h"
#include "../../core/componentSys/processor/PhyProcessor.h"
#include "../../core/Rendering/renderUtil.h"
#include "../defines/renderingLayers.h"

class pickupCreator : public objectCreator{
private:
	viewProcess *viewProc;

	PickupData pickup;
	float radius;

public:

	pickupCreator(viewProcess *_viewProc) : viewProc(_viewProc), radius(0){}
	
	void Init(PickupData data, float radius){
		this->pickup = data;
		this->radius = radius;
	}

	Object *createObject(vector2 _pos) const{
		Object *obj = new Object("bullet");

		vector2 *pos = obj->getPrimitive<vector2>(Hash::getHash("position"));
		*pos = _pos;

		//physics------------------------------------------------------------
		b2BodyDef bodyDef;
		bodyDef.type = b2_staticBody;

		b2CircleShape pickupBoundingBox;
		pickupBoundingBox.m_radius = 1.0f;

		b2FixtureDef fixtureDef;
		fixtureDef.shape = &pickupBoundingBox;
		fixtureDef.friction = 1.0;
		fixtureDef.restitution = 0.0;

		PhyData phy = PhyProcessor::createPhyData(&bodyDef, &fixtureDef);
		phy.collisionType = Hash::getHash("dummy");


		//renderer------------------------------------
		float game2RenderScale = this->viewProc->getGame2RenderScale();
		sf::Shape *SFMLShape  = new sf::CircleShape(this->radius * game2RenderScale,4);
		SFMLShape->setFillColor(sf::Color::Red);

		_RenderNode renderNode(SFMLShape, renderingLayers::action);
		RenderData renderData = RenderProcessor::createRenderData(&renderNode);
	
		//final---------------------------------
		obj->addProp(Hash::getHash("RenderData"), 
			new Prop<RenderData>(renderData));
		obj->addProp(Hash::getHash("PhyData"), 
			new Prop<PhyData>(phy));
		obj->addProp(Hash::getHash("PickupData"), 
			new Prop<PickupData>(this->pickup));
		
		return obj;
	};
};
