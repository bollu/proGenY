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
		RenderData render;
		PhyData phy;
		
		Object *obj = new Object("bullet");

		vector2 *pos = obj->getPrimitive<vector2>(Hash::getHash("position"));
		*pos = _pos;

		//physics------------------------------------------------------------
		phy.collisionType = Hash::getHash("pickup");
		phy.bodyDef.type = b2_staticBody;

		
		b2CircleShape *shape = new b2CircleShape();
		shape->m_radius = this->radius;
		b2FixtureDef fixtureDef;
		fixtureDef.shape = shape;
		fixtureDef.friction = 0.0;
		fixtureDef.restitution = 0.0;
		fixtureDef.isSensor = true;

		phy.fixtureDef.push_back(fixtureDef);


		//renderer------------------------------------
		float game2RenderScale = this->viewProc->getGame2RenderScale();
		sf::Shape *sfShape = new sf::CircleShape(this->radius * game2RenderScale,
										4);
		

		sfShape->setFillColor(sf::Color::Red);

		shapeRenderNode* renderer = new shapeRenderNode(sfShape);
		render.addRenderer(renderer);
		
	
		//final---------------------------------
		obj->addProp(Hash::getHash("RenderData"), 
			new Prop<RenderData>(render));
		obj->addProp(Hash::getHash("PhyData"), 
			new Prop<PhyData>(phy));
		obj->addProp(Hash::getHash("PickupData"), 
			new Prop<PickupData>(this->pickup));
		
		return obj;
	};
};
